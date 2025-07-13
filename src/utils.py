"""
Utility functions for the face detection project
"""

import cv2
import os
from typing import List, Tuple

def create_dataset_structure(base_path: str, person_names: List[str]):
    """
    Create directory structure for training dataset

    Args:
        base_path: Base directory path
        person_names: List of person names
    """
    os.makedirs(base_path, exist_ok=True)
    for name in person_names:
        person_dir = os.path.join(base_path, name)
        os.makedirs(person_dir, exist_ok=True)
        print(f"Created directory: {person_dir}")

def capture_training_images(person_name: str, dataset_path: str, num_images: int = 20):
    """
    Capture training images for a person using webcam

    Args:
        person_name: Name of the person
        dataset_path: Base dataset folder path
        num_images: Number of images to capture
    """
    save_dir = os.path.join(dataset_path, person_name)
    if not os.path.exists(save_dir):
        print(f"Error: Directory {save_dir} does not exist.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Capturing {num_images} images for '{person_name}'")
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow(f"Capturing for {person_name} - Press 'q' to quit", frame)

        image_path = os.path.join(save_dir, f"{count+1:03d}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Saved image: {image_path}")
        count += 1

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Image capture complete.")

def visualize_detection_results(image, faces: List[Tuple[int, int, int, int]], window_name="Detected Faces"):
    """
    Draw rectangles around detected faces and show the image

    Args:
        image: Input image
        faces: List of face bounding boxes (x, y, w, h)
        window_name: Window title
    """
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, 'Face', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
