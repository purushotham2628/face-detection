# 👤 Face Detection and Recognition System

A Python-based real-time face detection and recognition system using OpenCV, designed for image and video inputs. Built to support both classical methods (Haar/DNN) and extendable to advanced AI applications.

---

## 🚀 Features

- Real-time face detection using Haar and DNN models  
- Face recognition using local dataset and LBPH algorithm  
- Dataset creation via webcam  
- Modular codebase for easy experimentation and extension

---

## 📁 Use Cases

### 💻 Human-Computer Interaction
- Personalized Interfaces: Adapt UI based on recognized users  
- Emotion Recognition (future): Detect emotions and expressions  
- Gesture Control (future): Combine with gesture-based control  

---

## 🔧 How to Run

python main.py --mode detect --input path/to/image.jpg  
python main.py --mode webcam  
python main.py --mode capture --person John  
python main.py --mode train  
python main.py --mode recognize

---

## 🛠 Troubleshooting

1. Camera Not Detected  
- Check camera permissions  
- Try different camera indices (0, 1, 2...)  
- Ensure no other app is using the camera

2. Poor Detection Accuracy  
- Improve lighting conditions  
- Switch between haar and dnn methods  
- Adjust detection thresholds  

3. Recognition Not Working  
- Make sure model is trained  
- Check if dataset is large and well-lit  
- Verify correct folder structure  

4. Performance Issues  
- Lower frame/image resolution  
- Use lightweight detection models  
- Close background apps  

---

## 🤝 Contributing

1. Fork the repository  
2. Create a feature branch  
3. Make your changes  
4. Add tests (if applicable)  
5. Submit a Pull Request

---

## 📜 License

This project is licensed under the MIT License – see the LICENSE file for details.

---

## 🙏 Acknowledgments

- OpenCV  
- dlib  
- face_recognition  
- NumPy  
- Matplotlib

---

## 🌟 Future Enhancements

- Deep learning models (CNN-based face detection)  
- Multi-face tracking  
- Age and gender prediction  
- Emotion recognition  
- 3D face modeling  
- Mobile and edge deployment support
