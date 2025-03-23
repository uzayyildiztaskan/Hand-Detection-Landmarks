import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class FreiHandDataset(Dataset):
    def __init__(self, image_folder, image_files, keypoints, transform=None, original_size=(224, 224), target_size=(256, 256)):
        self.image_folder = image_folder
        self.image_files = image_files
        self.keypoints = keypoints
        self.transform = transform
        
        self.original_width, self.original_height = original_size
        self.target_width, self.target_height = target_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        kp = self.keypoints[idx].copy()

        scale_x = self.target_width / self.original_width
        scale_y = self.target_height / self.original_height

        kp[..., 0] *= scale_x
        kp[..., 1] *= scale_y

        x_min = np.min(kp[:, 0])
        y_min = np.min(kp[:, 1])
        x_max = np.max(kp[:, 0])
        y_max = np.max(kp[:, 1])
        
        margin = 0.05
        bbox = np.array([x_min - margin, y_min - margin, x_max + margin, y_max + margin], dtype=np.float32)     

        if self.transform:
            image = self.transform(image)

        kp_tensor = torch.tensor(kp, dtype=torch.float32)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        return image, bbox_tensor, kp_tensor
