# 🔧 Source Code Documentation

This directory contains the core modules that power the face detection and recognition system. Each module has a specific responsibility and works together to create a complete solution.

## 📁 Module Overview

### 🎯 face_detector.py
**Purpose**: Handles face detection using different algorithms

**Key Features**:
- Supports Haar Cascades (fast, lightweight)
- Supports DNN models (more accurate)
- Returns bounding boxes for detected faces
- Easy to switch between methods

**Usage Example**:
```python
from face_detector import FaceDetector

# Create detector
detector = FaceDetector('haar')  # or 'dnn'

# Detect faces in image
faces = detector.detect_faces(image)
print(f"Found {len(faces)} faces")
```

### 🧠 face_recognition_system.py
**Purpose**: Recognizes specific people from detected faces

**Key Features**:
- LBPH (Local Binary Pattern Histogram) algorithm
- Train from image datasets
- Returns person name and confidence score
- Handles unknown faces gracefully

**Usage Example**:
```python
from face_recognition_system import FaceRecognitionSystem

# Create and train system
frs = FaceRecognitionSystem()
frs.train_from_dataset('dataset/')

# Recognize a face
name, confidence = frs.recognize_face(face_image)
```

### 📹 video_processor.py
**Purpose**: Processes video streams and webcam feeds

**Key Features**:
- Real-time face detection and recognition
- FPS calculation and display
- Handles webcam input
- Draws bounding boxes and labels

**Usage Example**:
```python
from video_processor import VideoProcessor

# Create processor with recognition
vp = VideoProcessor(face_recognizer=frs)

# Start webcam processing
vp.process_webcam()
```

### 🛠 utils.py
**Purpose**: Helper functions for common tasks

**Key Features**:
- Dataset creation and management
- Training image capture
- Result visualization
- File system operations

**Usage Example**:
```python
from utils import capture_training_images, create_dataset_structure

# Create dataset folders
create_dataset_structure('dataset', ['Alice', 'Bob'])

# Capture training images
capture_training_images('Alice', 'dataset', num_images=25)
```

## 🔄 How Modules Work Together

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   main.py       │───▶│  video_processor │───▶│  face_detector  │
│  (Entry Point)  │    │   (Orchestrator) │    │   (Detection)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │face_recognition │───▶│     utils       │
                       │    (AI Brain)   │    │   (Helpers)     │
                       └─────────────────┘    └─────────────────┘
```

## 🎛 Configuration Options

### Face Detection Methods

#### Haar Cascades
```python
detector = FaceDetector('haar')
```
- **Pros**: Fast, lightweight, works on older hardware
- **Cons**: Less accurate, sensitive to lighting
- **Best for**: Real-time applications, resource-constrained systems

#### DNN (Deep Neural Network)
```python
detector = FaceDetector('dnn')
```
- **Pros**: More accurate, robust to variations
- **Cons**: Slower, requires more processing power
- **Best for**: High accuracy requirements, good hardware

### Recognition Parameters

#### Training Dataset Size
- **Minimum**: 10 images per person
- **Recommended**: 20-30 images per person
- **Optimal**: 50+ images with varied lighting/angles

#### Confidence Threshold
```python
# In face_recognition_system.py, adjust threshold
if confidence < 50:  # Lower = more strict
    return "Unknown", confidence
```

## 🐛 Common Issues & Solutions

### Import Errors
```python
# Problem: ModuleNotFoundError
# Solution: Add src to Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
```

### Memory Issues
```python
# Problem: High memory usage with video
# Solution: Resize frames before processing
frame = cv2.resize(frame, (640, 480))
```

### Performance Optimization
```python
# Skip frames for better performance
frame_count = 0
if frame_count % 3 == 0:  # Process every 3rd frame
    faces = detector.detect_faces(frame)
frame_count += 1
```

## 🔬 Advanced Customization

### Adding New Detection Methods
1. Extend `FaceDetector` class
2. Add new method in `_load_model()` function
3. Implement detection logic
4. Update method selection

### Custom Recognition Algorithms
1. Create new recognizer class
2. Implement `train()` and `predict()` methods
3. Replace LBPH with your algorithm
4. Test with your dataset

### Multi-threading for Performance
```python
import threading
from queue import Queue

# Process frames in separate thread
def process_frame_thread(frame_queue, result_queue):
    while True:
        frame = frame_queue.get()
        # Process frame
        result_queue.put(processed_frame)
```

## 📊 Performance Metrics

### Typical Performance (on modern laptop)
- **Haar Detection**: 30-60 FPS
- **DNN Detection**: 15-30 FPS
- **Recognition**: 20-40 FPS
- **Memory Usage**: 50-200 MB

### Optimization Tips
1. **Reduce frame size**: `cv2.resize(frame, (320, 240))`
2. **Skip frames**: Process every 2nd or 3rd frame
3. **Use threading**: Separate detection and display
4. **Optimize parameters**: Adjust `scaleFactor` and `minNeighbors`

## 🧪 Testing Your Changes

### Unit Testing
```python
# Test face detection
def test_face_detection():
    detector = FaceDetector('haar')
    # Load test image
    image = cv2.imread('test_face.jpg')
    faces = detector.detect_faces(image)
    assert len(faces) > 0, "Should detect at least one face"
```

### Integration Testing
```python
# Test full pipeline
def test_recognition_pipeline():
    # Train system
    frs = FaceRecognitionSystem()
    frs.train_from_dataset('test_dataset')
    
    # Test recognition
    test_image = cv2.imread('test_person.jpg')
    name, confidence = frs.recognize_face(test_image)
    assert name != "Unknown", "Should recognize trained person"
```

## 📈 Extending the System

### Add Emotion Detection
```python
# Install additional library
# pip install fer

from fer import FER
emotion_detector = FER()

# In video processing loop
emotions = emotion_detector.detect_emotions(frame)
```

### Add Age/Gender Detection
```python
# Use additional OpenCV models
age_net = cv2.dnn.readNet('age_net.caffemodel', 'age_deploy.prototxt')
gender_net = cv2.dnn.readNet('gender_net.caffemodel', 'gender_deploy.prototxt')
```

### Database Integration
```python
import sqlite3

# Store recognition results
def save_recognition_result(name, timestamp, confidence):
    conn = sqlite3.connect('recognition_log.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO recognitions (name, timestamp, confidence)
        VALUES (?, ?, ?)
    ''', (name, timestamp, confidence))
    conn.commit()
    conn.close()
```

---

**💡 Pro Tip**: Start with the basic functionality and gradually add features. Each module is designed to be independent, making it easy to modify or extend specific parts without affecting the entire system.