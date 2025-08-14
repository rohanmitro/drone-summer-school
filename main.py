#!/usr/bin/env python3
"""
Animal Image Classifier (PyTorch, transfer learning)
===================================================

Goal
----
Train a scalable image classifier for animals (e.g., antelope, zebra), and easily extend
it later by just adding a new folder of images (e.g., lions/, elephants/).

Highlights
----------
- Uses torchvision pretrained backbones (default: ResNet18).
- Infers classes from directory names (no hardcoding).
- Works with either a single dataset folder (auto split) **or** an explicit
  `train/val/test` directory structure.
- Data augmentation + normalization matched to pretrained weights.
- Handles class imbalance via optional `--balance sampler`.
- Mixed precision training for speed on GPU.
- Early stopping + ReduceLROnPlateau scheduler.
- Saves: best model checkpoint (`model.pt`), label mapping (`labels.json`),
  training configuration (`config.json`), and metrics (`metrics.json`).
- Command-line subcommands: `train`, `evaluate`, `predict-file`, `predict-folder`.

Dataset Layout
--------------
Option A — single folder (auto split):
    dataset/
        antelope/
            img001.jpg
            ...
        zebra/
            img101.jpg
            ...

Option B — explicit splits:
    dataset/
        train/
            antelope/*.jpg
            zebra/*.jpg
        val/
            antelope/*.jpg
            zebra/*.jpg
        test/
            antelope/*.jpg
            zebra/*.jpg

Add new classes later by simply creating a new subfolder (e.g., `lion/`) with images,
then run `train` again. The script will discover the new class automatically and
produce a new model + label map.

Quickstart
----------
1) Install deps (CUDA optional):
   pip install torch torchvision pillow tqdm

2) Train (auto-split 80/10/10):
   python animal_classifier.py train \
       --data-root ./dataset \
       --out-dir ./runs/antelopes-vs-zebras \
       --epochs 15 --batch-size 32 --lr 3e-4

3) Predict on a single image:
   python animal_classifier.py predict-file \
       --checkpoint ./runs/antelopes-vs-zebras/model.pt \
       --labels ./runs/antelopes-vs-zebras/labels.json \
       --image ./some_photo.jpg

4) Predict on a folder of images:
   python animal_classifier.py predict-folder \
       --checkpoint ./runs/antelopes-vs-zebras/model.pt \
       --labels ./runs/antelopes-vs-zebras/labels.json \
       --folder ./photos_to_check \
       --out-csv ./predictions.csv

Notes
-----
- For best results, aim for at least ~200 images per class, varied in lighting,
  angle, background. More is better.
- Images can be jpg/png/jpeg; non-images are ignored.
- If you add new animals (classes), **retrain** by running the `train` command again.
  (This script does not implement incremental learning with memory; it retrains
  a new model that includes the expanded label set.)

"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights

import pdb

# -----------------------------
# Utilities
# -----------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class TrainConfig:
    data_root: str
    out_dir: str
    model: str = "resnet18"  # [resnet18, efficientnet_b0]
    epochs: int = 15
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 5  # early stopping patience by val loss
    num_workers: int = 4
    seed: int = 42
    img_size: int = 224
    balance: str = "none"  # [none, sampler, weights]
    freeze_backbone: bool = False  # if True, only train final classifier layer
    auto_split: bool = True  # If no train/val/test dirs, auto-split from single folder
    val_split: float = 0.1
    test_split: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Data setup
# -----------------------------

def discover_splits(root: Path, auto_split: bool) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Return (train_dir, val_dir, test_dir). If they don't exist and auto_split=True,
    we'll return (root, None, None) and handle split later with random indices.
    """
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    if train_dir.exists():
        return train_dir, (val_dir if val_dir.exists() else None), (test_dir if test_dir.exists() else None)
    else:
        if auto_split:
            return root, None, None
        else:
            raise FileNotFoundError(
                "Expected data_root/train (and optionally val/test) folders, or enable --auto-split"
            )


def build_datasets(cfg: TrainConfig):
    root = Path(cfg.data_root)
    train_dir, val_dir, test_dir = discover_splits(root, cfg.auto_split)

    # Choose weights to get the right normalization + transforms
    if cfg.model == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        # ImageNet normalization values
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif cfg.model == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT
        # ImageNet normalization values
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError("Unsupported model. Choose 'resnet18' or 'efficientnet_b0'.")

    # Augmentations for training (resize handled inside base_transforms for these weights)
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize(int(cfg.img_size * 1.14)),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if train_dir is not None and val_dir is None and test_dir is None:
        # Single folder — auto split
        full = datasets.ImageFolder(train_dir, transform=train_tfms)
        num_total = len(full)
        indices = list(range(num_total))
        random.shuffle(indices)

        n_test = int(cfg.test_split * num_total)
        n_val = int(cfg.val_split * num_total)
        n_train = num_total - n_val - n_test

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_ds = Subset(full, train_idx)
        # swap to eval transforms for val/test
        full_eval = datasets.ImageFolder(train_dir, transform=eval_tfms)
        val_ds = Subset(full_eval, val_idx)
        test_ds = Subset(full_eval, test_idx)
        class_to_idx = full.class_to_idx
    else:
        # Explicit split dirs
        train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
        val_ds = datasets.ImageFolder(val_dir if val_dir else train_dir, transform=eval_tfms)
        test_ds = datasets.ImageFolder(test_dir if test_dir else val_dir if val_dir else train_dir, transform=eval_tfms)
        class_to_idx = train_ds.dataset.class_to_idx if isinstance(train_ds, Subset) else train_ds.class_to_idx

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return train_ds, val_ds, test_ds, class_to_idx, idx_to_class


def make_loaders(train_ds, val_ds, test_ds, cfg: TrainConfig, class_to_idx: Dict[str, int]):
    sampler = None
    if cfg.balance == "sampler":
        # WeightedRandomSampler to balance classes by sampling inversely to frequency
        targets = []
        if isinstance(train_ds, Subset):
            ds: datasets.ImageFolder = train_ds.dataset  # type: ignore
            for i in train_ds.indices:  # type: ignore
                targets.append(ds.samples[i][1])
        else:
            targets = [y for _, y in train_ds.samples]
        class_counts = np.bincount(targets)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = [class_weights[t] for t in targets]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# -----------------------------
# Model setup
# -----------------------------

def build_model(num_classes: int, backbone: str = "resnet18", freeze_backbone: bool = False) -> nn.Module:
    if backbone == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        m = models.resnet18(weights=weights)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, num_classes)
        if freeze_backbone:
            for name, p in m.named_parameters():
                if not name.startswith("fc"):
                    p.requires_grad = False
        return m
    elif backbone == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT
        m = models.efficientnet_b0(weights=weights)
        in_feats = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feats, num_classes)
        if freeze_backbone:
            for name, p in m.named_parameters():
                if not name.startswith("classifier.1"):
                    p.requires_grad = False
        return m
    else:
        raise ValueError("Unsupported backbone")


# -----------------------------
# Training / Evaluation Loops
# -----------------------------

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, scaler=None, train: bool = True):
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    epoch_acc = 0.0
    n = 0

    with torch.set_grad_enabled(train):
        for images, targets in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if train and scaler is not None:
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, targets)

            bs = images.size(0)
            epoch_loss += loss.item() * bs
            epoch_acc += accuracy_from_logits(outputs, targets) * bs
            n += bs

    return epoch_loss / n, epoch_acc / n


def train_model(cfg: TrainConfig):
    seed_everything(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Data
    train_ds, val_ds, test_ds, class_to_idx, idx_to_class = build_datasets(cfg)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, cfg, class_to_idx)

    # Save label map
    labels_path = Path(cfg.out_dir) / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(idx_to_class, f, indent=2)

    # Model
    num_classes = len(idx_to_class)
    model = build_model(num_classes, cfg.model, cfg.freeze_backbone).to(cfg.device)

    # Loss (optionally class-weighted)
    class_weights_t = None
    if cfg.balance == "weights":
        # compute weights inversely proportional to class frequency on train set
        counts = np.zeros(num_classes, dtype=np.int64)
        if isinstance(train_ds, Subset):
            base: datasets.ImageFolder = train_ds.dataset  # type: ignore
            for i in train_ds.indices:  # type: ignore
                _, y = base.samples[i]
                counts[y] += 1
        else:
            for _, y in train_ds.samples:
                counts[y] += 1
        w = 1.0 / np.maximum(counts, 1)
        w = w * (num_classes / w.sum())  # normalize a bit
        class_weights_t = torch.tensor(w, dtype=torch.float32, device=cfg.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    # Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device.startswith("cuda")))

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, cfg.device, scaler, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, cfg.device, scaler=None, train=False)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | Val: loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {
                'model_state': model.state_dict(),
                'num_classes': num_classes,
                'backbone': cfg.model,
                'idx_to_class': idx_to_class,
            }
            torch.save(best_state, Path(cfg.out_dir) / "model.pt")
            print("Saved new best checkpoint → model.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print("Early stopping: no improvement")
                break

    # Save history + config
    with open(Path(cfg.out_dir) / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(Path(cfg.out_dir) / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Final test evaluation (on best checkpoint if available)
    ckpt_path = Path(cfg.out_dir) / "model.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=cfg.device)
        model = build_model(len(state['idx_to_class']), state['backbone'], freeze_backbone=False)
        model.load_state_dict(state['model_state'])
        model.to(cfg.device)

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, cfg.device, scaler=None, train=False)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")


# -----------------------------
# Prediction helpers
# -----------------------------

def load_model_for_inference(checkpoint: str, device: str):
    state = torch.load(checkpoint, map_location=device)
    backbone = state['backbone']
    idx_to_class = state['idx_to_class']
    num_classes = len(idx_to_class)
    model = build_model(num_classes, backbone, freeze_backbone=False)
    model.load_state_dict(state['model_state'])
    model.to(device)
    model.eval()
    # get matching eval transforms
    if backbone == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        weights = EfficientNet_B0_Weights.DEFAULT
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    eval_tfms = transforms.Compose([
        transforms.Resize(int(224 * 1.14)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    

    return model, idx_to_class, eval_tfms


def predict_image(model, tfms, image_path: Path, device: str, idx_to_class: Dict[int, str], topk: int = 3):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    x = tfms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        values, indices = probs.topk(min(topk, probs.numel()))
    preds = [(idx_to_class[i.item()], float(values[j].item())) for j, i in enumerate(indices)]
    return preds


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Animal Image Classifier")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Train
    p_tr = sub.add_parser("train", help="Train a new model")
    p_tr.add_argument("--data-root", type=str, required=True, help="Path to dataset root")
    p_tr.add_argument("--out-dir", type=str, required=True, help="Where to save artifacts")
    p_tr.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "efficientnet_b0"]) 
    p_tr.add_argument("--epochs", type=int, default=15)
    p_tr.add_argument("--batch-size", type=int, default=32)
    p_tr.add_argument("--lr", type=float, default=3e-4)
    p_tr.add_argument("--weight-decay", type=float, default=1e-4)
    p_tr.add_argument("--patience", type=int, default=5)
    p_tr.add_argument("--num-workers", type=int, default=4)
    p_tr.add_argument("--seed", type=int, default=42)
    p_tr.add_argument("--img-size", type=int, default=224)
    p_tr.add_argument("--balance", type=str, default="none", choices=["none", "sampler", "weights"],
                      help="Handle class imbalance via sampler or loss weights")
    p_tr.add_argument("--freeze-backbone", action="store_true")
    p_tr.add_argument("--auto-split", action="store_true", help="Auto-split a single folder into train/val/test")
    p_tr.add_argument("--val-split", type=float, default=0.1)
    p_tr.add_argument("--test-split", type=float, default=0.1)
    p_tr.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate
    p_ev = sub.add_parser("evaluate", help="Evaluate on a test folder")
    p_ev.add_argument("--checkpoint", type=str, required=True)
    p_ev.add_argument("--labels", type=str, required=False, help="Optional labels.json (unused if inside checkpoint)")
    p_ev.add_argument("--test-root", type=str, required=True, help="Folder with class subfolders to test on")
    p_ev.add_argument("--batch-size", type=int, default=64)
    p_ev.add_argument("--num-workers", type=int, default=4)
    p_ev.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Predict a single file
    p_pf = sub.add_parser("predict-file", help="Predict a single image")
    p_pf.add_argument("--checkpoint", type=str, required=True)
    p_pf.add_argument("--labels", type=str, required=False)
    p_pf.add_argument("--image", type=str, required=True)
    p_pf.add_argument("--topk", type=int, default=3)
    p_pf.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Predict a folder
    p_pd = sub.add_parser("predict-folder", help="Predict all images in a folder (writes CSV)")
    p_pd.add_argument("--checkpoint", type=str, required=True)
    p_pd.add_argument("--labels", type=str, required=False)
    p_pd.add_argument("--folder", type=str, required=True)
    p_pd.add_argument("--out-csv", type=str, required=True)
    p_pd.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def cmd_train(args):
    cfg = TrainConfig(
        data_root=args.data_root,
        out_dir=args.out_dir,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
        img_size=args.img_size,
        balance=args.balance,
        freeze_backbone=args.freeze_backbone,
        auto_split=args.auto_split,
        val_split=args.val_split,
        test_split=args.test_split,
        device=args.device,
    )
    print("Config:\n" + json.dumps(asdict(cfg), indent=2))
    train_model(cfg)


def cmd_evaluate(args):
    device = args.device
    model, idx_to_class, eval_tfms = load_model_for_inference(args.checkpoint, device)

    test_ds = datasets.ImageFolder(args.test_root, transform=eval_tfms)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)  # dummy for API comp

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, scaler=None, train=False)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

    # Confusion matrix
    n_classes = len(test_ds.classes)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.no_grad():
        for images, targets in tqdm(test_loader, leave=False):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            for t, p in zip(targets.view(-1), preds.view(-1)):
                cm[t.long().item(), p.long().item()] += 1
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)


def cmd_predict_file(args):
    device = args.device
    model, idx_to_class, tfms = load_model_for_inference(args.checkpoint, device)
    preds = predict_image(model, tfms, Path(args.image), device, idx_to_class, topk=args.topk)
    print(json.dumps({"image": args.image, "predictions": preds}, indent=2))


def cmd_predict_folder(args):
    import csv
    device = args.device
    model, idx_to_class, tfms = load_model_for_inference(args.checkpoint, device)

    folder = Path(args.folder)
    paths = [p for p in folder.iterdir() if p.is_file() and is_image_file(p)]
    os.makedirs(Path(args.out_csv).parent, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["image", "top1_label", "top1_prob", "top2_label", "top2_prob", "top3_label", "top3_prob"]
        writer.writerow(header)
        for p in tqdm(paths):
            preds = predict_image(model, tfms, p, device, idx_to_class, topk=3)
            flat = [p.as_posix()]
            # ensure fixed width of 3 preds
            for i in range(3):
                if i < len(preds):
                    flat.extend([preds[i][0], f"{preds[i][1]:.6f}"])
                else:
                    flat.extend(["", ""])
            writer.writerow(flat)
    print(f"Wrote predictions → {args.out_csv}")


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "evaluate":
        cmd_evaluate(args)
    elif args.cmd == "predict-file":
        cmd_predict_file(args)
    elif args.cmd == "predict-folder":
        cmd_predict_folder(args)
