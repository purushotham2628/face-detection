"""
Face Detection Module
Handles face detection using OpenCV's Haar Cascades and DNN models
"""

import cv2
import numpy as np
from typing import List, Tuple
import os

class FaceDetector:
    def __init__(self, method='haar'):
        """
        Initialize face detector with specified method

        Args:
            method (str): Detection method - 'haar' or 'dnn'
        """
        self.method = method
        self.face_cascade = None
        self.net = None

        if method == 'haar':
            self._load_haar_cascade()
        elif method == 'dnn':
            self._load_dnn_model()
        else:
            raise ValueError(f"Unsupported method: {method}")

    def _load_haar_cascade(self):
        haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(haar_path):
            raise FileNotFoundError(f"Haar cascade not found at {haar_path}")
        self.face_cascade = cv2.CascadeClassifier(haar_path)

    def _load_dnn_model(self):
        proto_path = 'models/deploy.prototxt'
        model_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
        if not os.path.exists(proto_path) or not os.path.exists(model_path):
            raise FileNotFoundError("DNN model files not found.")
        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image using the selected method.

        Args:
            image: The input image (numpy array)

        Returns:
            A list of bounding boxes: (x, y, w, h)
        """
        if self.method == 'haar':
            return self._detect_faces_haar(image)
        elif self.method == 'dnn':
            return self._detect_faces_dnn(image)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _detect_faces_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return faces

    def _detect_faces_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0), swapRB=False, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                boxes.append((x, y, x1 - x, y1 - y))
        return boxes
