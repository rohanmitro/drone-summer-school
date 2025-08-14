#!/usr/bin/env python3
"""
All-in-One Animal Detector for Grassland Terrain
------------------------------------------------
1. Trains a lightweight classifier (MobileNetV2) on your dataset folders.
2. Saves the model to 'animal_classifier.h5'.
3. Runs terrain-specific detection + classification on a test image.
"""

import os
import cv2
import numpy as np
import ssl
import urllib.request

# Fix SSL certificate verification issue
ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# === TRAINING ===
def train_classifier(dataset_dir, model_path="animal_classifier.h5", epochs=10):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        dataset_dir, target_size=(224,224), batch_size=16,
        class_mode='categorical', subset='training'
    )
    val_gen = train_datagen.flow_from_directory(
        dataset_dir, target_size=(224,224), batch_size=16,
        class_mode='categorical', subset='validation'
    )

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    preds = Dense(train_gen.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=preds)

    # Freeze backbone for speed
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")

    return model, list(train_gen.class_indices.keys())


# === ENHANCED DETECTION + CLASSIFICATION ===
def detect_and_classify_terrain(image_path, model, class_names, min_area=800, edge_thresh=0.03, top_k=3, min_score=100000):
    """
    Enhanced detection using advanced CV techniques + deep learning classification.
    
    Args:
        image_path: Path to input image
        model: Trained classification model
        class_names: List of class names
        min_area: Minimum contour area to consider
        edge_thresh: Minimum edge density threshold (0-1)
        top_k: Maximum number of detections to keep
        min_score: Minimum detection score threshold (default: 100000)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not load image {image_path}")
        return
    output_img = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. Mask green grass (same thresholds as run_detection.py)
    green_lower = np.array([35, 40, 40], dtype=np.uint8)
    green_upper = np.array([85, 255, 255], dtype=np.uint8)
    grass_mask = cv2.inRange(hsv, green_lower, green_upper)

    # 2. Invert to get non-green objects
    non_green_mask = cv2.bitwise_not(grass_mask)

    # 3. ENHANCED: Better brightness filter (from run_detection.py)
    v_channel = hsv[:,:,2]
    _, bright_mask = cv2.threshold(v_channel, 60, 255, cv2.THRESH_BINARY)

    # 4. Combine masks
    mask = cv2.bitwise_and(non_green_mask, bright_mask)

    # 5. ENHANCED: Better morphological cleanup (from run_detection.py)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 6. NEW: Edge map for enhanced detection (from run_detection.py)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # 7. Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"No contours found in {os.path.basename(image_path)}")
        cv2.imshow("No Detections", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 8. NEW: Advanced candidate scoring and filtering
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # NEW: Reject green patches explicitly (from run_detection.py)
        roi_hsv = hsv[y:y+h, x:x+w]
        roi_green = cv2.inRange(roi_hsv, green_lower, green_upper)
        green_ratio = float(np.count_nonzero(roi_green)) / max(1, (w * h))
        if green_ratio > 0.55:  # mostly grass -> skip
            continue

        # NEW: Edge density filter (from run_detection.py)
        roi_edges = edges[y:y+h, x:x+w]
        edge_ratio = roi_edges.mean() / 255.0
        if edge_ratio < edge_thresh:
            continue

        # NEW: Calculate detection score
        score = area * (0.5 + edge_ratio)
        
        # Filter out low-score detections
        if score < min_score:
            continue
            
        candidates.append((score, (x, y, w, h)))

    if not candidates:
        print(f"No suitable detections found in {os.path.basename(image_path)}")
        cv2.imshow("No Valid Detections", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 9. NEW: Keep top-K candidates by score
    candidates.sort(reverse=True, key=lambda t: t[0])
    top_candidates = candidates[:min(top_k, len(candidates))]

    print(f"Found {len(top_candidates)} high-quality detections")

    # 10. ENHANCED: Classify and draw results
    for i, (score, (x, y, w, h)) in enumerate(top_candidates):
        # Extract and classify crop
        crop = img[y:y+h, x:x+w]
        crop_resized = cv2.resize(crop, (224,224))
        crop_array = preprocess_input(np.expand_dims(crop_resized, axis=0).astype(np.float32))
        
        pred = model.predict(crop_array, verbose=0)
        label = class_names[np.argmax(pred)]
        conf = np.max(pred)

        # Color-code by confidence (green/red only)
        if conf > 0.6:
            color = (0, 255, 0)  # Green for high confidence
        else:
            color = (0, 0, 255)  # Red for low confidence

        # Draw enhanced bounding box
        cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 3)
        
        # Enhanced label with score info
        edge_ratio = (score / area) - 0.5  # Back-calculate edge ratio
        label_text = f"{label} {conf:.2f} (S:{score:.0f})"
        
        # Calculate text position with background
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_x = x
        text_y = y - 10 if y - 10 > text_height else y + text_height + 10
        
        # Draw text background
        cv2.rectangle(output_img, (text_x, text_y - text_height - baseline), 
                     (text_x + text_width, text_y + baseline), color, -1)
        
        # Draw text
        cv2.putText(output_img, label_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        print(f"Detection {i+1}: {label} (conf: {conf:.3f}, score: {score:.1f}, area: {area}, edges: {edge_ratio:.3f})")

    cv2.imshow("Enhanced Detections", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# === TESTING FUNCTIONS ===
def test_single_image(image_path, model_path="animal_classifier.h5", min_area=800, edge_thresh=0.03, top_k=3, min_score=10000):
    """
    Test the enhanced detector on a single image using pre-trained model.
    
    Args:
        image_path: Path to the image to test
        model_path: Path to the saved model (default: animal_classifier.h5)
        min_area: Minimum area for detection
        edge_thresh: Edge density threshold
        top_k: Maximum detections to show
        min_score: Minimum detection score threshold (default: 10000)
    """
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file {model_path} not found. Please train the model first.")
        return False
    
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file {image_path} not found.")
        return False
    
    print(f"[INFO] Loading pre-trained model from {model_path}")
    model = load_model(model_path)
    
    # Infer class names from model (antelope=0, zebra=1 typically)
    class_names = ['antelope', 'zebra']  # These match the dataset folder names
    
    print(f"[INFO] Testing image: {image_path}")
    detect_and_classify_terrain(image_path, model, class_names, min_area, edge_thresh, top_k, min_score)
    return True


def train_and_test(dataset_dir="dataset", test_image=None, epochs=10):
    """
    Train a new model and optionally test it on an image.
    
    Args:
        dataset_dir: Directory containing training data
        test_image: Optional image to test after training
        epochs: Number of training epochs
    """
    print("[INFO] Training new model...")
    model, class_names = train_classifier(dataset_dir, epochs=epochs)
    
    if test_image:
        print(f"[INFO] Testing on: {test_image}")
        detect_and_classify_terrain(test_image, model, class_names)
    
    return model, class_names


# === MAIN ===
if __name__ == "__main__":
    import sys
    
    # Check if model exists
    MODEL_PATH = "animal_classifier.h5"
    
    if len(sys.argv) > 1:
        # Command line usage: python animal_detector.py <image_path>
        image_path = sys.argv[1]
        print(f"[INFO] Testing image from command line: {image_path}")
        
        if os.path.exists(MODEL_PATH):
            test_single_image(image_path, MODEL_PATH)
        else:
            print(f"[INFO] No pre-trained model found. Training first...")
            model, class_names = train_classifier("dataset", epochs=10)
            detect_and_classify_terrain(image_path, model, class_names)
    
    elif os.path.exists(MODEL_PATH):
        # Model exists, test with default image
        TEST_IMAGE = "test/zebra+antelope1.jpg"  # Default test image
        print(f"[INFO] Using existing model to test: {TEST_IMAGE}")
        test_single_image(TEST_IMAGE, MODEL_PATH)
    
    else:
        # No model exists, train new one
        print("[INFO] No pre-trained model found. Training new model...")
        DATASET_DIR = "dataset"
        TEST_IMAGE = "test/zebra+antelope1.jpg"
        train_and_test(DATASET_DIR, TEST_IMAGE, epochs=10)
