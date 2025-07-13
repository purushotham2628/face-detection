"""
Main application for Face Detection and Recognition System
"""

import cv2
import os
import argparse
from src.face_detector import FaceDetector
from src.face_recognition_system import FaceRecognitionSystem
from src.video_processor import VideoProcessor
from src.utils import (
    create_dataset_structure, 
    capture_training_images, 
    visualize_detection_results
)

def main():
    parser = argparse.ArgumentParser(description="Face Detection and Recognition System")
    parser.add_argument('--mode', choices=['detect', 'webcam', 'train', 'recognize', 'capture'], 
                        default='webcam', help='Operation mode')
    parser.add_argument('--input', help='Input image/video path')
    parser.add_argument('--output', help='Output path')
    parser.add_argument('--dataset', default='dataset', help='Dataset directory path')
    parser.add_argument('--method', choices=['haar', 'dnn'], default='haar', 
                        help='Face detection method')
    parser.add_argument('--person', help='Person name for training data capture')
    parser.add_argument('--num-images', type=int, default=20, help='Number of training images to capture')
    
    args = parser.parse_args()
    
    if args.mode == 'detect':
        # Single image face detection
        if not args.input:
            print("Please provide input image path with --input")
            return

        detector = FaceDetector(args.method)
        image = cv2.imread(args.input)
        
        if image is None:
            print(f"Could not load image: {args.input}")
            return
        
        faces = detector.detect_faces(image)
        print(f"Detected {len(faces)} faces using {args.method} method")
        
        visualize_detection_results(image, faces)

    elif args.mode == 'webcam':
        # Real-time webcam processing
        detector = FaceDetector(args.method)
        frs = FaceRecognitionSystem()
        frs.detector = detector  # Inject detector
        vp = VideoProcessor(face_recognizer=frs)
        vp.process_webcam()

    elif args.mode == 'train':
        # Train model from dataset
        frs = FaceRecognitionSystem()
        frs.train_from_dataset(args.dataset)
        print("Model training complete.")

    elif args.mode == 'recognize':
        # Recognize faces using webcam
        frs = FaceRecognitionSystem()
        frs.train_from_dataset(args.dataset)
        vp = VideoProcessor(face_recognizer=frs)
        vp.process_webcam()

    elif args.mode == 'capture':
        # Capture training data
        if not args.person:
            print("Please provide person name with --person")
            return
        create_dataset_structure(args.dataset, [args.person])
        capture_training_images(args.person, args.dataset, args.num_images)

if __name__ == "__main__":
    main()
