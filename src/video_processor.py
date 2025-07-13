"""
Video Processing Module
Handles real-time face detection and recognition in video streams
"""

import cv2
import numpy as np
import time
from typing import Optional
from .face_recognition_system import FaceRecognitionSystem

class VideoProcessor:
    def __init__(self, face_recognizer: Optional[FaceRecognitionSystem] = None):
        """
        Initialize video processor
        Args:
            face_recognizer: Optional face recognition system
        """
        self.face_recognizer = face_recognizer
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for face detection/recognition

        Args:
            frame: Input frame
        Returns:
            Processed frame with annotations
        """
        if self.face_recognizer is None:
            return frame

        # Detect faces
        faces = self.face_recognizer.detector.detect_faces(frame)

        # Process each detected face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for (x, y, w, h) in faces:
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract face ROI for recognition
            face_roi = gray[y:y + h, x:x + w]

            # Recognize face
            if self.face_recognizer.is_trained:
                name, confidence = self.face_recognizer.recognize_face(face_roi)
                label = f"{name} ({confidence:.1f})"
            else:
                label = "Untrained"

            # Display name and confidence
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # FPS calculation
        self.fps_counter += 1
        elapsed_time = time.time() - self.fps_start_time
        if elapsed_time > 1.0:
            self.current_fps = self.fps_counter / elapsed_time
            self.fps_counter = 0
            self.fps_start_time = time.time()

        cv2.putText(frame, f"FPS: {self.current_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def process_webcam(self):
        """
        Start webcam and process each frame for face detection/recognition
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow("Face Recognition", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Video processing completed.")
