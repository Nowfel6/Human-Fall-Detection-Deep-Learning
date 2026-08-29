# 🧍‍♂️ Real-Time Human Fall Detection using Deep Learning

An advanced Spatio-Temporal Deep Learning project that detects human falls in real-time. Instead of using wearable sensors, this project utilizes **Computer Vision (Google MediaPipe)** to extract 3D skeletal data from video feeds and processes them using **Sequence Models (LSTM, GRU, Bi-LSTM)**.

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)

## 🚀 Project Overview
The objective of this project is to build a highly accurate, non-intrusive fall detection system for elderly care and surveillance. Falls are continuous actions, not static images. Therefore, the system extracts body landmarks over a series of frames and feeds them into Recurrent Neural Networks equipped with "memory" to understand the downward trajectory and physics of a fall.

---

## 🧠 Data Engineering Pipeline (Creating Custom Dataset)
One of the major achievements of this project is the custom data extraction pipeline. Instead of relying on pre-processed CSV files, the dataset was generated from raw video footage.

1. **Source Data:** Raw videos of normal daily activities (walking, sitting) and sudden falls were collected from the public **[UR Fall Detection Dataset](http://fenix.ur.edu.pl/~mkepski/ds/uf.html)** and other CCTV compilations.
2. **Spatial Extraction (`Newbie_HFD.py`):** An automated OpenCV script loops through the raw `.mp4` files. For every frame, **MediaPipe Holistic** extracts 33 critical body joints (132 coordinates: X, Y, Z, Visibility).
3. **Temporal Sequencing:** To capture the "transition" of a fall, the extracted keypoints are packaged into **60-frame Sliding Windows** (approx. 2 seconds of action).
4. **Final 3D Shape:** The generated Numpy array translates raw videos into a mathematical 3D tensor: `[Samples, Time-Steps (60), Features (132)]`, perfectly formatted for LSTM ingestion.

---

## 🏗️ Model Architecture & Benchmarking
To find the most optimized "Brain" for the system, three different Sequence Models were built, trained, and compared (`Train_Compare_All_Model.py`). 

*   **LSTM:** Standard baseline sequence model.
*   **GRU:** A faster, streamlined version of LSTM with fewer gates.
*   **Bi-LSTM (Bidirectional LSTM):** Processes the 60-frame sequence both forwards and backwards, providing maximum context for sudden impacts.

### 📊 Performance Evaluation
*(See the performance charts generated during training below)*

*   **Comparison Chart:** `Architecture_Comparison_Chart.png` demonstrates the validation accuracy across the different models.
*   **Learning Curve:** `Learning_Curve.png` proves the model generalized well without severe overfitting, aided by an **Early Stopping** callback (patience=15).

---

## 📂 Repository Structure
```text
Human-Fall-Detection/
├── Live_Detection_Model.py      # Real-time webcam inference script
├── Train_Model.py               # Main training script for the best model
├── Train_Compare_All_Model.py   # Script to benchmark LSTM vs GRU vs Bi-LSTM
├── Newbie_HFD.py                # Automated custom data extraction from raw videos
├── PoseEstimationMin.py         # Basic MediaPipe testing module
├── Fall_Detection_Model.h5      # The final trained AI Model
├── Learning_Curve.png           # Training vs Validation loss graph
├── Comparison_Chart.png         # Model benchmarking bar chart
└── README.md                    # Project documentation
```
*(Note: The raw videos and extracted `.npy` files are ignored in this repository due to size constraints. To replicate the dataset, simply run `Newbie_HFD.py` with your own `.mp4` files).*

---

## 🛠️ How to Run the Project

**1. Clone the repository:**
```bash
git clone https://github.com/YourUsername/Human-Fall-Detection-Deep-Learning.git
cd Human-Fall-Detection-Deep-Learning
```

**2. Install Dependencies:**
```bash
pip install tensorflow==2.15.0 opencv-python mediapipe numpy matplotlib scikit-learn
```

**3. Run Live Detection:**
To test the pre-trained model in real-time using your webcam:
```bash
python Live_Detection_Model.py
```
*Stand in front of the camera. The system will display "NORMAL / SAFE" in green. Simulate a fall, and it will immediately trigger a red "!!! FALL DETECTED !!!" alert.*

---
**Author:** [Your Name / Nowfel]  
**Focus:** Computer Vision, Spatio-Temporal Deep Learning, Data Engineering
