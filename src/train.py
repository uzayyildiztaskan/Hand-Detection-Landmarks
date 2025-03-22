import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from data.freihanddataset import FreiHandDataset
from config.params import Config
import numpy as np
from sklearn.model_selection import train_test_split
from data.preprocess import convert_to_2d
from src.model.landmarkmodel import HandLandmarkModel
import argparse
import matplotlib as plt


parser = argparse.ArgumentParser(description='Hand Detection Training Script')

hp = Config()

parser.add_argument('--dataset_path', type=str, default=hp.DATASET_PATH, help='Path to RGB folder')
parser.add_argument('--checkpoint_dir', type=str, default=hp.CHECKPOINT_DIR, help='Path to save checkpoints')

args = parser.parse_args()

hp.DATASET_PATH = args.dataset_path

def detection_loss(pred_bbox, true_bbox):
    return nn.SmoothL1Loss()(pred_bbox, true_bbox)

def keypoint_loss(pred_keypoints, true_keypoints):
    return nn.MSELoss()(pred_keypoints, true_keypoints)

def compute_iou(pred_boxes, true_boxes):
    x1 = torch.max(pred_boxes[:, 0], true_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], true_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], true_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], true_boxes[:, 3])
    
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
    
    union_area = pred_area + true_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    return iou.mean().item()

def compute_pck(pred_keypoints, true_keypoints, threshold=0.05):
    dists = torch.norm(pred_keypoints - true_keypoints, dim=2)
    correct = (dists < threshold).float().mean().item()
    return correct


os.makedirs(hp.CHECKPOINT_DIR, exist_ok=True)

if not os.path.exists(hp.KEYPOINT_ANNOTATION_2D_PATH):
    convert_to_2d(hp)

all_image_files = sorted(os.listdir(hp.RGB_FOLDER_PATH))
all_keypoints = np.tile(np.load(hp.KEYPOINT_ANNOTATION_2D_PATH), (4, 1, 1))

train_files, val_files, train_kps, val_kps = train_test_split(
    all_image_files, all_keypoints, test_size=0.2, random_state=42
)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = FreiHandDataset(
    image_folder=hp.RGB_FOLDER_PATH,
    image_files=train_files,
    keypoints=train_kps,
    transform=transform
)

val_dataset = FreiHandDataset(
    image_folder=hp.RGB_FOLDER_PATH,
    image_files=val_files,
    keypoints=val_kps,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=hp.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=hp.BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HandLandmarkModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)

ALPHA = hp.ALPHA
BETA = hp.BETA
EPOCHS = hp.EPOCHS
CHECKPOINT_DIR = hp.CHECKPOINT_DIR

best_val_loss = float('inf')

best_model_file = None

train_losses, val_losses = [], []
train_ious, val_ious = [], []
train_pcks, val_pcks = [], []

for epoch in range(EPOCHS):
    model.train()
    train_total_loss = 0
    pbar = tqdm(train_loader)
    epoch_train_iou, epoch_train_pck = 0, 0

    for imgs, bboxes, keypoints in pbar:
        imgs, bboxes, keypoints = imgs.to(device), bboxes.to(device), keypoints.to(device)
        
        optimizer.zero_grad()
        pred_bboxes, pred_keypoints = model(imgs)
        
        loss_det = detection_loss(pred_bboxes, bboxes)
        loss_kp = keypoint_loss(pred_keypoints, keypoints)
        loss = ALPHA * loss_det + BETA * loss_kp
        
        loss.backward()
        optimizer.step()
        
        train_total_loss += loss.item()
        epoch_train_iou += compute_iou(pred_bboxes, bboxes)
        epoch_train_pck += compute_pck(pred_keypoints, keypoints)
        
    avg_train_loss = train_total_loss / len(train_loader)
    avg_train_iou = epoch_train_iou / len(train_loader)
    avg_train_pck = epoch_train_pck / len(train_loader)

    train_losses.append(avg_train_loss)
    train_ious.append(avg_train_iou)
    train_pcks.append(avg_train_pck)


    model.eval()
    val_total_loss = 0
    
    epoch_val_iou, epoch_val_pck = 0, 0

    with torch.no_grad():
        for imgs, bboxes, keypoints in val_loader:
            imgs, bboxes, keypoints = imgs.to(device), bboxes.to(device), keypoints.to(device)
            
            pred_bboxes, pred_keypoints = model(imgs)
            
            loss_det = detection_loss(pred_bboxes, bboxes)
            loss_kp = keypoint_loss(pred_keypoints, keypoints)
            loss = ALPHA * loss_det + BETA * loss_kp
            
            val_total_loss += loss.item()
            epoch_val_iou += compute_iou(pred_bboxes, bboxes)
            epoch_val_pck += compute_pck(pred_keypoints, keypoints)

    avg_val_loss = val_total_loss / len(val_loader)
    avg_val_iou = epoch_val_iou / len(val_loader)
    avg_val_pck = epoch_val_pck / len(val_loader)

    val_losses.append(avg_val_loss)
    val_ious.append(avg_val_iou)
    val_pcks.append(avg_val_pck)

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    print(f"Train IoU: {avg_train_iou:.4f}, Val IoU: {avg_val_iou:.4f}")
    print(f"Train PCK: {avg_train_pck:.4f}, Val PCK: {avg_val_pck:.4f}")

    if avg_val_loss < best_val_loss:
        print(f"Validation loss improved! Saving checkpoint...")
        best_model_file = f"checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss
        }, os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pt'))
        best_val_loss = avg_val_loss

os.makedirs('outputs/plots', exist_ok=True)

epochs_range = range(1, EPOCHS + 1)

plt.figure()
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Val Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.savefig('outputs/plots/loss_plot.png')

plt.figure()
plt.plot(epochs_range, train_ious, label='Train IoU')
plt.plot(epochs_range, val_ious, label='Val IoU')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.title('Training & Validation IoU')
plt.savefig('outputs/plots/iou_plot.png')

plt.figure()
plt.plot(epochs_range, train_pcks, label='Train PCK')
plt.plot(epochs_range, val_pcks, label='Val PCK')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('PCK')
plt.title('Training & Validation PCK')
plt.savefig('outputs/plots/pck_plot.png')

print("Plots saved to outputs/plots/")
    
print(f"Best model: {best_model_file}")