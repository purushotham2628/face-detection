# ⚡ Quick Start Guide

Get up and running with face detection in under 5 minutes!

## 🎯 What You'll Achieve
By the end of this guide, you'll have:
- ✅ Face detection working on your webcam
- ✅ Your own face recognition system
- ✅ Understanding of how it all works

## 📋 Prerequisites
- Python 3.7 or higher
- A webcam (built-in or external)
- 5 minutes of your time

## 🚀 Step-by-Step Setup

### Step 1: Install Dependencies (2 minutes)
```bash
# Clone or download this project
# Navigate to the project folder
cd face-detection-system

# Install required packages
pip install -r requirements.txt
```

**What's happening?** This installs OpenCV, NumPy, and other libraries needed for computer vision.

### Step 2: Test Basic Detection (1 minute)
```bash
python main.py --mode webcam
```

**What you should see:**
- Your webcam opens
- Green rectangles appear around faces
- FPS counter in the corner

**Controls:**
- Press `q` to quit
- Move around to test detection

### Step 3: Create Your Recognition System (2 minutes)

#### 3a. Capture Your Photos
```bash
python main.py --mode capture --person YourName --num-images 25
```

**What happens:**
- Webcam opens for 25 seconds
- Takes photos automatically
- Saves them in `dataset/YourName/` folder

**Tips for better photos:**
- Look directly at camera
- Slight head movements (left, right, up, down)
- Keep good lighting
- Try different expressions

#### 3b. Train the System
```bash
python main.py --mode train
```

**What happens:**
- Processes your photos
- Creates a recognition model
- Takes about 10-30 seconds

#### 3c. Test Recognition
```bash
python main.py --mode recognize
```

**What you should see:**
- Your name appears above your face
- Confidence score (lower = better match)
- "Unknown" for other people

## 🎉 Success! What Now?

### Add More People
```bash
# Capture photos for family/friends
python main.py --mode capture --person Mom --num-images 25
python main.py --mode capture --person Dad --num-images 25

# Retrain with new people
python main.py --mode train

# Test with multiple people
python main.py --mode recognize
```

### Try Different Detection Methods
```bash
# Use DNN (more accurate, slower)
python main.py --mode webcam --method dnn

# Compare with Haar (faster, less accurate)
python main.py --mode webcam --method haar
```

### Process Photos Instead of Webcam
```bash
# Detect faces in a photo
python main.py --mode detect --input path/to/photo.jpg
```

## 🔧 Troubleshooting

### Camera Won't Open
```bash
# Try different camera index
# Edit main.py, change VideoCapture(0) to VideoCapture(1)
```

### Poor Recognition
```bash
# Capture more photos with better lighting
python main.py --mode capture --person YourName --num-images 50

# Retrain
python main.py --mode train
```

### Slow Performance
```bash
# Use Haar method (faster)
python main.py --mode webcam --method haar
```

## 📚 Next Steps

1. **Read the full README.md** for detailed explanations
2. **Try the examples/** folder for more advanced features
3. **Modify the code** to add your own features
4. **Share your results** and contribute back!

## 🆘 Need Help?

- **Check the main README.md** for detailed documentation
- **Look at examples/** for code samples
- **Open an issue** on GitHub if you find bugs
- **Read src/README.md** for technical details

---

**🎊 Congratulations!** You now have a working face detection and recognition system. Time to explore and build something amazing!