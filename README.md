<div style="background-color: rgba(255, 0, 0, 0.1); color: red; padding: 12px; text-align: center; font-weight: bold; border: 2px solid red; border-radius: 6px; font-size: 18px;">
  ⚠️⚠️ <span style="color: black;">🔧 Currently Being Updated</span> ⚠️⚠️
</div>


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
.
└── hand-detection-landmarks/
    ├── config/
    │   └── params
    ├── data/
    │   ├── freihanddataset.py                  # Custom Dataset class for FreiHAND
    │   └── preprocess.py                       # Conversion of 3D keypoints to 2D
    ├── dataset                                 # FreiHAND dataset (not included)
    ├── inference_images                        # Output folder for inference images
    ├── outputs/
    │   ├── checkpoints                         # Saved checkpoints (not included)
    │   └── plots                               # Saved plots from training (not included)
    ├── src/
    │   ├── model/
    │   │   └── landmarkmodel.py                # Model architecture definition
    │   ├── utils/
    │   │   ├── accuracy/
    │   │   │   └── custom_accuracies.py        # IoU and PCB for accuracy metrics
    │   │   └── losses/
    │   │       ├── custom_loss_functions.py    # SmoothL1 and WingLoss
    │   │       └── wing_loss.py                # WingLoss class    
    │   ├── inference.py
    │   └── train.py
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
