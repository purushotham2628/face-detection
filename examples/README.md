# 🎯 Examples & Tutorials

This directory contains practical examples to help you understand and use the face detection system. Each example focuses on a specific aspect and builds upon the previous ones.

## 📚 Learning Path

### 🟢 Beginner Examples
Start here if you're new to computer vision or this project.

### 🟡 Intermediate Examples  
Try these after you understand the basics.

### 🔴 Advanced Examples
For experienced users who want to extend the system.

---

## 🟢 Beginner Examples

### 1. 📄 basic_detection.py
**What it does**: Simple face detection using your webcam

**Learning objectives**:
- Understand basic face detection
- Learn OpenCV video capture
- See how to draw bounding boxes

**How to run**:
```bash
cd examples
python basic_detection.py
```

**What you'll see**:
- Green rectangles around detected faces
- Face count displayed on screen
- Real-time processing from your webcam

**Key concepts**:
```python
# Initialize detector
detector = FaceDetector('haar')

# Detect faces
faces = detector.detect_faces(frame)

# Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
```

**Try this**: 
- Move closer/farther from camera
- Turn your head different angles
- Try with multiple people

---

### 2. 📄 image_detection.py
**What it does**: Detect faces in static images (photos)

**Learning objectives**:
- Process image files instead of video
- Understand different image formats
- Learn result visualization

**How to run**:
```bash
cd examples
python image_detection.py path/to/your/photo.jpg
```

**What you'll see**:
- Your photo with detected faces highlighted
- Number of faces found
- Confidence scores (if using DNN method)

**Key concepts**:
```python
# Load image from file
image = cv2.imread(image_path)

# Detect faces
faces = detector.detect_faces(image)

# Visualize results
visualize_detection_results(image, faces)
```

**Try this**:
- Use family photos
- Try group photos vs individual portraits
- Compare Haar vs DNN methods

---

## 🟡 Intermediate Examples

### 3. 📄 recognition_demo.py
**What it does**: Complete face recognition system setup

**Learning objectives**:
- Create training datasets
- Train recognition models
- Perform real-time recognition

**How to run**:
```bash
cd examples
python recognition_demo.py
```

**What happens**:
1. **Setup**: Creates dataset folders
2. **Capture**: Takes training photos for each person
3. **Training**: Builds recognition model
4. **Recognition**: Tests real-time recognition

**Interactive process**:
```
=== Face Recognition Demo Setup ===

1. Setting up dataset structure...
Enter person name (or 'done' to finish): Alice
Enter person name (or 'done' to finish): Bob
Enter person name (or 'done' to finish): done

2. Capturing training images...
Capturing images for: Alice
[Webcam opens - smile and move your head slightly]

3. Training model...
[Processing images and creating model]

4. Testing recognition...
[Webcam opens with name labels]
```

**Key concepts**:
```python
# Create dataset structure
create_dataset_structure(dataset_path, person_names)

# Capture training images
for name in person_names:
    capture_training_images(name, dataset_path)

# Train recognition system
frs = FaceRecognitionSystem()
frs.train_from_dataset(dataset_path)

# Start real-time recognition
vp = VideoProcessor(face_recognizer=frs)
vp.process_webcam()
```

**Pro tips**:
- Capture images with different expressions
- Vary lighting conditions during capture
- Include photos with/without glasses
- Take some from different angles

---

## 🔴 Advanced Examples

### 4. Custom Detection Parameters
**File**: Create `advanced_detection.py`

**What it teaches**: Fine-tuning detection parameters

```python
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.face_detector import FaceDetector

def advanced_detection_demo():
    """Demonstrate parameter tuning for better detection"""
    
    # Create detector
    detector = FaceDetector('haar')
    
    # Custom parameters for different scenarios
    scenarios = {
        'close_up': {'scaleFactor': 1.05, 'minNeighbors': 3},
        'group_photo': {'scaleFactor': 1.1, 'minNeighbors': 5},
        'distant': {'scaleFactor': 1.3, 'minNeighbors': 7}
    }
    
    cap = cv2.VideoCapture(0)
    current_scenario = 'close_up'
    
    print("Press 1, 2, 3 to switch scenarios, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get current parameters
        params = scenarios[current_scenario]
        
        # Detect with custom parameters
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=params['scaleFactor'],
            minNeighbors=params['minNeighbors']
        )
        
        # Draw results
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display info
        cv2.putText(frame, f"Scenario: {current_scenario}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Advanced Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_scenario = 'close_up'
        elif key == ord('2'):
            current_scenario = 'group_photo'
        elif key == ord('3'):
            current_scenario = 'distant'
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    advanced_detection_demo()
```

### 5. Performance Benchmarking
**File**: Create `benchmark.py`

```python
import cv2
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.face_detector import FaceDetector

def benchmark_detection_methods():
    """Compare performance of different detection methods"""
    
    methods = ['haar', 'dnn']
    results = {}
    
    # Load test image
    test_image = cv2.imread('test_image.jpg')  # Add your test image
    if test_image is None:
        print("Please add a test_image.jpg file")
        return
    
    for method in methods:
        print(f"\nTesting {method.upper()} method...")
        detector = FaceDetector(method)
        
        # Warm up
        for _ in range(5):
            detector.detect_faces(test_image)
        
        # Benchmark
        start_time = time.time()
        iterations = 100
        
        for i in range(iterations):
            faces = detector.detect_faces(test_image)
            if i == 0:  # Store first result
                results[method] = {
                    'faces_detected': len(faces),
                    'avg_time': 0
                }
        
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        results[method]['avg_time'] = avg_time
        
        print(f"Average time: {avg_time*1000:.2f}ms")
        print(f"Faces detected: {results[method]['faces_detected']}")
        print(f"Estimated FPS: {1/avg_time:.1f}")
    
    # Compare results
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    
    for method, data in results.items():
        print(f"{method.upper()}:")
        print(f"  Time per frame: {data['avg_time']*1000:.2f}ms")
        print(f"  Estimated FPS: {1/data['avg_time']:.1f}")
        print(f"  Faces detected: {data['faces_detected']}")
        print()

if __name__ == "__main__":
    benchmark_detection_methods()
```

---

## 🎯 Practice Exercises

### Exercise 1: Modify Basic Detection
**Goal**: Change the bounding box color and add face numbering

**Tasks**:
1. Change rectangle color from green to blue
2. Add numbers above each detected face
3. Display total face count in window title

**Hint**: Look at the `cv2.rectangle()` and `cv2.putText()` functions

### Exercise 2: Save Detection Results
**Goal**: Save images with detected faces

**Tasks**:
1. Add a key press to save current frame
2. Save images with timestamp in filename
3. Only save frames that have detected faces

**Hint**: Use `cv2.imwrite()` and `datetime` module

### Exercise 3: Multi-Method Comparison
**Goal**: Show Haar and DNN results side by side

**Tasks**:
1. Create two detectors (Haar and DNN)
2. Split screen to show both results
3. Display method name and face count for each

**Hint**: Use `numpy.hstack()` to combine images horizontally

---

## 🚀 Next Steps

After completing these examples, you can:

1. **Modify the main application** with your improvements
2. **Create your own examples** for specific use cases
3. **Contribute back** to the project with new examples
4. **Build applications** using this as a foundation

### Ideas for Your Own Examples
- **Security camera**: Motion detection + face recognition
- **Photo organizer**: Batch process photo collections
- **Attendance system**: Log recognized faces with timestamps
- **Emotion detector**: Add emotion recognition to faces
- **Age estimator**: Predict age from detected faces

---

## 🆘 Getting Help

### Common Issues

**Problem**: Examples don't run
```bash
# Solution: Make sure you're in the right directory
cd face-detection-system/examples
python basic_detection.py
```

**Problem**: Import errors
```bash
# Solution: Install requirements first
pip install -r ../requirements.txt
```

**Problem**: Camera not working
```bash
# Solution: Check camera permissions and close other apps
# Try different camera index in VideoCapture(1) instead of VideoCapture(0)
```

### Where to Ask Questions
1. **GitHub Issues**: For bugs and feature requests
2. **Discussions**: For general questions and ideas
3. **Stack Overflow**: Tag with `opencv` and `face-detection`

---

**🎉 Congratulations!** You now have a solid understanding of how face detection and recognition work. Use these examples as building blocks for your own computer vision projects!