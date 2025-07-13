"""
Face Recognition Demo
Complete example showing face detection and recognition
"""

import cv2
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.face_recognition_system import FaceRecognitionSystem
from src.video_processor import VideoProcessor
from src.utils import capture_training_images, create_dataset_structure

def setup_recognition_demo():
    """Setup a complete face recognition demo"""

    print("=== Face Recognition Demo Setup ===")

    # Create dataset directory
    dataset_path = "demo_dataset"
    
    print("\n1. Setting up dataset structure...")
    person_names = []

    while True:
        name = input("Enter person name (or 'done' to finish): ").strip()
        if name.lower() == "done":
            break
        elif name == "":
            print("Invalid name. Please enter a valid name.")
        else:
            person_names.append(name)

    if not person_names:
        print("No names entered. Exiting.")
        return

    create_dataset_structure(dataset_path, person_names)

    print("\n2. Capturing training images...")
    for name in person_names:
        print(f"\nCapturing images for: {name}")
        capture_training_images(name, dataset_path)

    print("\n3. Initializing face recognition system...")
    frs = FaceRecognitionSystem()
    frs.train_from_dataset(dataset_path)

    print("\n4. Starting real-time face recognition...")
    vp = VideoProcessor(face_recognizer=frs)
    vp.process_webcam()

if __name__ == "__main__":
    setup_recognition_demo()
