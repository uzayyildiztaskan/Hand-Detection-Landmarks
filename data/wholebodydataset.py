import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json

class COCOWholeBodyHandDataset(Dataset):
    def __init__(self, image_folder, annotation_file, transform=None, target_size=(256, 256)):
        self.image_folder = image_folder
        self.transform = transform
        self.target_width, self.target_height = target_size
        
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)['annotations']
        
        self.image_id_to_file = {}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        annotation = self.annotations[idx]
        
        boxes = []
        keypoints = []

        if annotation['lefthand_valid']:
            lh_box = annotation['lefthand_box']
            lh_kpts = annotation['lefthand_kpts']
            boxes.append(lh_box)
            keypoints.append(lh_kpts)
        
        if annotation['righthand_valid']:
            rh_box = annotation['righthand_box']
            rh_kpts = annotation['righthand_kpts']
            boxes.append(rh_box)
            keypoints.append(rh_kpts)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            keypoints = torch.zeros((0, 21, 3), dtype=torch.float32)  # (num_hands, num_keypoints, 3)

        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            keypoints = torch.tensor(keypoints, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, boxes, keypoints

