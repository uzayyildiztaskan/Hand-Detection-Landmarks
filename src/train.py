import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from data.freihanddataset import FreiHandDataset
from config.params import Config
import numpy as np
from sklearn.model_selection import train_test_split
from data.preprocess import convert_to_2d
from src.model.landmarkmodel import HandLandmarkModel
from src.utils.losses.custom_loss_functions import detection_loss, keypoint_loss
from src.utils.accuracy.custom_accuracies import compute_iou, compute_pck
import argparse
import matplotlib as plt
from torch.optim.lr_scheduler import StepLR


parser = argparse.ArgumentParser(description='Hand Detection Training Script')

hp = Config()

parser.add_argument('--dataset_path', type=str, default=hp.DATASET_PATH, help='Path to RGB folder')
parser.add_argument('--checkpoint_dir', type=str, default=hp.CHECKPOINT_DIR, help='Path to save checkpoints')

args = parser.parse_args()

hp.DATASET_PATH = args.dataset_path


def get_optimizer(model):
    return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hp.LEARNING_RATE)


def progressive_unfreeze(epoch, model, step=4, layers_per_step=1, start_epoch=3, total_layers_to_unfreeze = 6):
   
    if epoch < start_epoch or (epoch != 0 and epoch != start_epoch and (epoch - start_epoch) % step != 0):
        return
    
    feature_layers = list(model.feature_extractor.children())

    count = 0
    for i in reversed(range(len(feature_layers) - total_layers_to_unfreeze, len(feature_layers))):
        layer = feature_layers[i]
        if any(not p.requires_grad for p in layer.parameters()):
            for param in layer.parameters():
                param.requires_grad = True
            count += 1
        if count >= layers_per_step:
            break
    
    print(f"Epoch {epoch}: Unfroze {count} new layers in feature extractor.")



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
    transform=transform,
    original_size=(224, 224),
    target_size=(256, 256)
)

val_dataset = FreiHandDataset(
    image_folder=hp.RGB_FOLDER_PATH,
    image_files=val_files,
    keypoints=val_kps,
    transform=transform,
    original_size=(224, 224),
    target_size=(256, 256)
)

train_loader = DataLoader(train_dataset, batch_size=hp.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=hp.BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HandLandmarkModel().to(device)

for param in model.feature_extractor.parameters():
    param.requires_grad = False

optimizer = optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

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

    progressive_unfreeze(epoch, model, layers_per_step=2, start_epoch=3)
    optimizer = get_optimizer(model)

    model.train()
    train_total_loss = 0
    pbar = tqdm(train_loader)
    epoch_train_iou, epoch_train_pck = 0, 0

    debug_var = 0

    for imgs, bboxes, keypoints in pbar:
        imgs, bboxes, keypoints = imgs.to(device), bboxes.to(device), keypoints.to(device)
        
        optimizer.zero_grad()
        pred_bboxes, pred_keypoints = model(imgs)

        pred_bboxes_scaled = pred_bboxes * 256
        pred_keypoints_scaled = pred_keypoints * 256
        
        loss_det = detection_loss(pred_bboxes_scaled, bboxes)
        loss_kp = keypoint_loss(pred_keypoints_scaled, keypoints)
        loss = ALPHA * loss_det + BETA * loss_kp
        
        loss.backward()
        optimizer.step()
        
        train_total_loss += loss.item()
        epoch_train_iou += compute_iou(pred_bboxes_scaled, bboxes)
        epoch_train_pck += compute_pck(pred_keypoints_scaled, keypoints)

        debug_var += 1

        if debug_var % 3256 == 0:
            print("\n=======================\n", pred_bboxes_scaled[0])
            print("\n=======================\n", bboxes[0])
            print("\n=======================\n", pred_keypoints_scaled[0])
            print("\n=======================\n", keypoints[0])
        
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

            pred_bboxes_scaled = pred_bboxes * 256
            pred_keypoints_scaled = pred_keypoints * 256
            
            loss_det = detection_loss(pred_bboxes_scaled, bboxes)
            loss_kp = keypoint_loss(pred_keypoints_scaled, keypoints)
            loss = ALPHA * loss_det + BETA * loss_kp
            
            val_total_loss += loss.item()
            epoch_val_iou += compute_iou(pred_bboxes_scaled, bboxes)
            epoch_val_pck += compute_pck(pred_keypoints_scaled, keypoints)

    scheduler.step()

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