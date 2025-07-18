"""
Face Recognition System Module
Handles training and recognition of specific individuals using LBPH algorithm
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional
from .face_detector import FaceDetector

class FaceRecognitionSystem:
    def __init__(self):
        """
        Initialize the face recognition system
        """
        self.detector = FaceDetector('haar')  # Default to Haar for speed
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.is_trained = False
        self.label_to_name = {}  # Maps numeric labels to person names
        self.name_to_label = {}  # Maps person names to numeric labels
        
    def train_from_dataset(self, dataset_path: str):
        """
        Train the recognition system from a dataset directory
        
        Dataset structure should be:
        dataset/
        ├── person1/
        │   ├── 001.jpg
        │   ├── 002.jpg
        │   └── ...
        ├── person2/
        │   ├── 001.jpg
        │   └── ...
        
        Args:
            dataset_path: Path to the dataset directory
        """
        if not os.path.exists(dataset_path):
            print(f"Dataset path {dataset_path} does not exist")
            return False
            
        faces = []
        labels = []
        current_label = 0
        
        print("Loading training data...")
        
        # Process each person's directory
        for person_name in os.listdir(dataset_path):
            person_dir = os.path.join(dataset_path, person_name)
            
            if not os.path.isdir(person_dir):
                continue
                
            print(f"Processing images for: {person_name}")
            
            # Map person name to numeric label
            self.label_to_name[current_label] = person_name
            self.name_to_label[person_name] = current_label
            
            # Process each image in person's directory
            image_count = 0
            for image_file in os.listdir(person_dir):
                if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                image_path = os.path.join(person_dir, image_file)
                image = cv2.imread(image_path)
                
                if image is None:
                    continue
                    
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Detect faces in the training image
                detected_faces = self.detector.detect_faces(image)
                
                # Use the largest face (assuming it's the main subject)
                if len(detected_faces) > 0:
                    # Sort by area and take the largest
                    detected_faces = sorted(detected_faces, key=lambda x: x[2]*x[3], reverse=True)
                    x, y, w, h = detected_faces[0]
                    
                    # Extract face region
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Resize to standard size for consistency
                    face_roi = cv2.resize(face_roi, (200, 200))
                    
                    faces.append(face_roi)
                    labels.append(current_label)
                    image_count += 1
            
            print(f"  Loaded {image_count} images for {person_name}")
            current_label += 1
        
        if len(faces) == 0:
            print("No training faces found!")
            return False
            
        print(f"Training with {len(faces)} face images...")
        
        # Train the recognizer
        self.recognizer.train(faces, np.array(labels))
        self.is_trained = True
        
        print("Training completed successfully!")
        print(f"Trained to recognize {len(self.label_to_name)} people:")
        for label, name in self.label_to_name.items():
            print(f"  - {name}")
            
        return True
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Recognize a face from an image
        
        Args:
            face_image: Grayscale face image (numpy array)
            
        Returns:
            Tuple of (person_name, confidence_score)
            Returns ("Unknown", confidence) if not recognized
        """
        if not self.is_trained:
            return "Untrained", 0.0
            
        # Resize face to match training size
        face_resized = cv2.resize(face_image, (200, 200))
        
        # Predict
        label, confidence = self.recognizer.predict(face_resized)
        
        # Lower confidence means better match
        # Threshold can be adjusted based on your needs
        confidence_threshold = 50  # Adjust this value
        
        if confidence < confidence_threshold:
            person_name = self.label_to_name.get(label, "Unknown")
            return person_name, confidence
        else:
            return "Unknown", confidence
    
    def save_model(self, model_path: str):
        """
        Save the trained model to file
        
        Args:
            model_path: Path to save the model
        """
        if not self.is_trained:
            print("No trained model to save")
            return False
            
        try:
            self.recognizer.save(model_path)
            
            # Save label mappings
            import pickle
            mappings = {
                'label_to_name': self.label_to_name,
                'name_to_label': self.name_to_label
            }
            
            mapping_path = model_path.replace('.yml', '_mappings.pkl')
            with open(mapping_path, 'wb') as f:
                pickle.dump(mappings, f)
                
            print(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path: str):
        """
        Load a previously trained model
        
        Args:
            model_path: Path to the saved model
        """
        try:
            self.recognizer.read(model_path)
            
            # Load label mappings
            import pickle
            mapping_path = model_path.replace('.yml', '_mappings.pkl')
            
            if os.path.exists(mapping_path):
                with open(mapping_path, 'rb') as f:
                    mappings = pickle.load(f)
                    self.label_to_name = mappings['label_to_name']
                    self.name_to_label = mappings['name_to_label']
            
            self.is_trained = True
            print(f"Model loaded from {model_path}")
            print(f"Can recognize: {list(self.label_to_name.values())}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_training_stats(self) -> dict:
        """
        Get statistics about the trained model
        
        Returns:
            Dictionary with training statistics
        """
        if not self.is_trained:
            return {"status": "not_trained"}
            
        return {
            "status": "trained",
            "num_people": len(self.label_to_name),
            "people": list(self.label_to_name.values()),
            "detection_method": self.detector.method
        }