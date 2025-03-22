import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from data.freihanddataset import FreiHandDataset
import config.params as hp
import os
import numpy as np
from sklearn.model_selection import train_test_split
from data.preprocess import convert_to_2d
from model.landmarkmodel import HandLandmarkModel

def detection_loss(pred_bbox, true_bbox):
    return nn.SmoothL1Loss()(pred_bbox, true_bbox)

def keypoint_loss(pred_keypoints, true_keypoints):
    return nn.MSELoss()(pred_keypoints, true_keypoints)


os.makedirs(hp.CHECKPOINT_DIR, exist_ok=True)

if not os.path.exists(hp.KEYPOINT_ANNOTATION_2D_PATH):
    convert_to_2d()

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

for epoch in range(EPOCHS):
    model.train()
    train_total_loss = 0
    pbar = tqdm(train_loader)
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
        pbar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    avg_train_loss = train_total_loss / len(train_loader)

    model.eval()
    val_total_loss = 0
    
    with torch.no_grad():
        for imgs, bboxes, keypoints in val_loader:
            imgs, bboxes, keypoints = imgs.to(device), bboxes.to(device), keypoints.to(device)
            
            pred_bboxes, pred_keypoints = model(imgs)
            
            loss_det = detection_loss(pred_bboxes, bboxes)
            loss_kp = keypoint_loss(pred_keypoints, keypoints)
            loss = ALPHA * loss_det + BETA * loss_kp
            
            val_total_loss += loss.item()
    
    avg_val_loss = val_total_loss / len(val_loader)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
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
    
print(f"Best model: {best_model_file}")