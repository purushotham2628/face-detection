"""
Image Face Detection Example
Example showing face detection on static images
"""

import cv2
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.face_detector import FaceDetector
from src.utils import visualize_detection_results

def detect_faces_in_image(image_path):
    """
    Detect faces in a static image
    
    Args:
        image_path: Path to the image file
    """
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Initialize detector
    detector = FaceDetector('haar')
    
    # Detect faces
    faces = detector.detect_faces(image)
        print("No valid image provided")