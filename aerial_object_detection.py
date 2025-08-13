import cv2
import numpy as np

def detect_multiple_objects(image_path: str, min_area: int = 500, output_path: str = 'multiple_objects_detected.jpg'):
    """
    Detects multiple objects in an image and draws bounding boxes around them.

    Args:
        image_path (str): The file path for the input image.
        min_area (int): Minimum contour area to consider as a valid object (default: 1000).
    """
    # 1. Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # --- Pre-processing ---

    # 2. Convert to grayscale and apply a blur to reduce noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- Detection using Edges and Contours ---

    # 3. Use the Canny algorithm to find prominent edges
    canny_edges = cv2.Canny(blurred, 50, 150)

    # 4. Find all contours from the edge map
    contours, _ = cv2.findContours(canny_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours were found. Detection failed.")
        return

    # 5. Filter contours by area to find valid objects
    valid_objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            valid_objects.append(contour)

    if not valid_objects:
        print(f"No objects found with area greater than {min_area} pixels.")
        return

    print(f"Found {len(valid_objects)} objects")

    # --- Output Generation ---

    # 6. Draw bounding boxes for all valid objects
    output_image = image.copy()
    
    for i, contour in enumerate(valid_objects):
        # Calculate the bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw the bounding box with a different color for each object
        color = (0, 255, 0) if i == 0 else (255, 0, 0) if i == 1 else (0, 0, 255)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 3)
        
        # Add object number label
        cv2.putText(output_image, f"Object {i+1}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        print(f"Object {i+1}: Bounding box at ({x}, {y}) with size {w}x{h}")

    # 7. Save the final image
    cv2.imwrite(output_path, output_image)

    print("Detection complete! Check the output file: 'multiple_objects_detected.jpg'.")


if __name__ == '__main__':
    # Add system arguments to pass the file path.
    import argparse
    parser = argparse.ArgumentParser(description='Detect multiple objects in an image.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--min_area', type=int, default=100, help='Minimum area of the object to be detected')
    parser.add_argument('--output_path', type=str, default='multiple_objects_detected.jpg', help='Path to the output image')
    
    args = parser.parse_args()
    detect_multiple_objects(args.image_path, args.min_area, args.output_path)