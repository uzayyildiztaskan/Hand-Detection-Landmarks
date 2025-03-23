# 🖐️ Hand Detection with Landmarks

A PyTorch-based deep learning project focused on detecting multiple hands in an image and predicting precise landmark keypoints for each hand. This repository serves as a foundational module for gesture recognition, sign language translation, and human-computer interaction systems.

---

## 🚀 Features

- 🔍 **Multi-hand detection** capability
- 🎯 **21 keypoint landmark localization** per detected hand
- 📦 **Modular dataset loader**, currently tailored for the [FreiHAND dataset](https://lmb.informatik.uni-freiburg.de/projects/freihand/).
- 💾 **Validation & checkpointing** integrated
- 🏆 **Supports augmentation handling and custom training splits**
- 🔥 Designed for **real-time extensions and downstream tasks**

---

## 🗂️ Project Structure
```
hand-detection-landmarks/
├── src/
│   ├── data/
│   │   └── freihand_dataset.py        # Custom Dataset class for FreiHAND
│   ├── models/
│   │   └── hand_landmark_model.py     # Model architecture definition
│   ├── utils/
│   │   └── preprocessing.py           # 3D to 2D keypoint projection, helpers
│   ├── config/
│   └── params.py                      # Hyperparameters & file paths
│   ├── train.py                       # Training loop
├── checkpoints/                       # Saved model checkpoints (not included)
├── dataset/
│   └── FreiHAND_pub_v2/               # Dataset folder (not included)
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🏁 Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/uzayyildiztaskan/Hand-Detection-Landmarks
cd Hand-Detection-Landmarks
```
### 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
### 3️⃣ Prepare Dataset
Download the FreiHAND dataset

Place files in:

```
dataset/FreiHAND_pub_v2/
```
### 4️⃣ Train the Model
```
python -m src.train
```
### 5️⃣ Checkpoints & Validation
Checkpoints are saved in ```/checkpoints/``` when validation loss improves. Configure validation split in ```config/params.py```.

### 5️⃣ Inference
After specifying ```IMAGE_PATH``` and ```CHECKPOINT_PATH``` in ```inference.py``` run the script by
```
python -m src.inference
```

![Sample image inference from dataset](https://github.com/uzayyildiztaskan/Hand-Detection-Landmarks/blob/main/inference_images/Figure_1.png)

Infered model parameters:

* Epochs: ```15```
* Layers Unfreezed: ```6```
* Train Loss: ```10.9017```, Val Loss: ```14.3914```
* Train IoU: ```0.8973```, Val IoU: ```0.8919```
* Train PCK: ```0.9942```, Val PCK: ```0.9826```
* ALPHA: ```1.0```
* BETA: ```1.0```
* Margin of bounding box: ```0```

---

## 📌 Future Work

* Sign language translation pipeline integration

* Real-time inference demo (OpenCV/Streamlit)

---

## 📜 License
MIT License - see [LICENSE](https://github.com/uzayyildiztaskan/Hand-Detection-Landmarks/blob/main/LICENSE) for details.
