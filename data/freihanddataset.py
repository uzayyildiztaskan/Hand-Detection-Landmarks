import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class FreiHandDataset(Dataset):
    def __init__(self, image_folder, image_files, keypoints, transform=None):
        self.image_folder = image_folder
        self.image_files = image_files
        self.keypoints = keypoints
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        keypoints = torch.tensor(self.keypoints[idx], dtype=torch.float32)
        
        x_min = torch.min(keypoints[:, 0])
        y_min = torch.min(keypoints[:, 1])
        x_max = torch.max(keypoints[:, 0])
        y_max = torch.max(keypoints[:, 1])
        bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
        
        if self.transform:
            img = self.transform(img)
        
        return img, bbox, keypoints

