import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from config.params import Config
import numpy as np
from sklearn.model_selection import train_test_split
from data.process_annotations import filter_annotations
from data.dataset_download import download_filtered_dataset
from src.model.landmarkmodel import HandLandmarkModel
from src.utils.losses.custom_loss_function import detection_loss
from src.utils.accuracy.custom_accuracies import compute_iou, compute_pck
from data.wholebodydataset import COCOWholeBodyHandDataset
import argparse
import matplotlib as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import torch.nn.functional as F


def main():

    parser = argparse.ArgumentParser(description='Hand Detection Training Script')

    hp = Config()

    parser.add_argument('--dataset_path', type=str, default=hp.DATASET_PATH, help='Path to the dataset folder')
    parser.add_argument('--checkpoint_output_dir', type=str, default=hp.CHECKPOINT_DIR, help='Path to save checkpoints')
    parser.add_argument('--saved_model_file', type=str, default=None, help='Path to saved model checkpoint file to resume training')

    args = parser.parse_args()

    hp.DATASET_PATH = args.dataset_path
    hp.CHECKPOINT_DIR = args.checkpoint_output_dir
    model_path = args.saved_model_file
    
    checkpoint = torch.load(model_path) if model_path else None

    os.makedirs(hp.CHECKPOINT_DIR, exist_ok=True)

    if not os.path.exists(hp.FILTERED_ANNOTATIONS_PATH):
        filter_annotations(hp)

    if not os.path.exists(hp.IMAGES_PATH):
        download_filtered_dataset(hp)

    with open(hp.FILTERED_ANNOTATIONS_PATH, 'r') as f:
        full_data = json.load(f)

    images = os.listdir(hp.IMAGES_PATH)

    filtered_images = [os.path.splitext(img)[0] for img in images if os.path.splitext(img)[0] in full_data]
    hand_labels = [int(full_data[os.path.splitext(img)[0]]['has_valid_hands']) for img in filtered_images]

    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42, stratify=hand_labels)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = COCOWholeBodyHandDataset(
        image_folder=hp.IMAGES_PATH,
        image_files=train_images,
        annotation_file=hp.FILTERED_ANNOTATIONS_PATH,
        transform=transform,
        target_size=(512, 512)
    )

    val_dataset = COCOWholeBodyHandDataset(
        image_folder=hp.IMAGES_PATH,
        image_files=val_images,
        annotation_file=hp.FILTERED_ANNOTATIONS_PATH,
        transform=transform,
        target_size=(512, 512)
    )

    train_loader = DataLoader(train_dataset, batch_size=hp.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=hp.BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HandLandmarkModel().to(device)

    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    else:
        optimizer = optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    EPOCHS = hp.EPOCHS
    CHECKPOINT_DIR = hp.CHECKPOINT_DIR

    best_val_loss = checkpoint['val_loss'] if checkpoint else float('inf')

    best_model_file = model_path if checkpoint else None

    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    train_pcks, val_pcks = [], []

    start_epoch = checkpoint['epoch'] if checkpoint else 0

    for epoch in range(start_epoch, EPOCHS):

        model.train()
        train_total_loss = 0
        pbar = tqdm(train_loader)
        epoch_train_iou, epoch_train_pck = 0, 0

        debug_var = 0

        for imgs, confidences, bboxes, keypoints in pbar:
            imgs, confidences, bboxes, keypoints = imgs.to(device), confidences.to(device), bboxes.to(device), keypoints.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs)

            pred_confidences = preds[..., 0:1]
            pred_bboxes = preds[..., 1:5]
            pred_keypoints = preds[..., 5:]

            pred_bboxes_scaled = pred_bboxes * 512
            pred_keypoints_scaled = pred_keypoints * 512

            preds = torch.cat([pred_confidences, pred_bboxes_scaled, pred_keypoints_scaled], dim=-1)

            loss, loss_det = detection_loss(preds, confidences, bboxes, keypoints)
            
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
            for imgs, confidences, bboxes, keypoints in val_loader:
                imgs, confidences, bboxes, keypoints = imgs.to(device), confidences.to(device), bboxes.to(device), keypoints.to(device)
                
                pred_confidences, pred_bboxes, pred_keypoints = model(imgs)

                pred_bboxes_scaled = pred_bboxes * 512
                pred_keypoints_scaled = pred_keypoints * 512

                preds = torch.cat([pred_confidences, pred_bboxes_scaled, pred_keypoints_scaled], dim=-1)

                loss, loss_det = detection_loss(preds, confidences, bboxes, keypoints)
                
                val_total_loss += loss.item()
                epoch_val_iou += compute_iou(pred_bboxes_scaled, bboxes)
                epoch_val_pck += compute_pck(pred_keypoints_scaled, keypoints)

        avg_val_loss = val_total_loss / len(val_loader)
        avg_val_iou = epoch_val_iou / len(val_loader)
        avg_val_pck = epoch_val_pck / len(val_loader)

        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)
        val_pcks.append(avg_val_pck)

        scheduler.step(avg_val_loss)

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
                'scheduler_state_dict': scheduler.state_dict(),
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

if __name__ == '__main__':
    main()