import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json

## Custom dataset class for handling COCO dataset with filtered annotations ##

class COCOWholeBodyHandDataset(Dataset):
    def __init__(self, image_folder, image_files, annotation_file, transform=None, target_size=(512, 512), grid_size = 32, num_keypoints = 21):
        self.image_folder = image_folder
        self.image_files = image_files
        self.transform = transform
        self.target_width, self.target_height = target_size
        
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.features_per_anchor = 1 + 4 + (3 * num_keypoints)  # conf, bbox, keypoints((x, y, v) for each)
        self.num_anchors = 2  # left + right hand

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, image_file_name)
        image = Image.open(img_path).convert('RGB')
        image_id = os.path.splitext(image_file_name)[0]

        annotation = self.annotations[image_id]

        # Scaling proportions of the image for the bbox and keypoint coordinate resizing
        scale_x = self.target_width / annotation['width']
        scale_y = self.target_height / annotation['height']

        S = self.grid_size
        target = torch.zeros((S, S, self.num_anchors, self.features_per_anchor), dtype=torch.float32)

        # Iterate through each person in the image
        for person_hands in annotation.values():

            for anchor_idx, (valid_key, box_key, kpts_key) in enumerate([
                ('left_hand_valid', 'lefthand_box', 'lefthand_kpts'),
                ('right_hand_valid', 'righthand_box', 'righthand_kpts')
            ]):
                
                if not person_hands[valid_key]:
                    continue

                x, y, w, h = person_hands[box_key]

                cx = (x + w / 2) * scale_x
                cy = (y + h / 2) * scale_y
                w *= scale_x
                h *= scale_y

                # Corresponding grid cell
                cell_w = self.target_width / S
                cell_h = self.target_height / S
                cell_x = int(cx / cell_w)
                cell_y = int(cy / cell_h)

                if cell_x >= S or cell_y >= S:
                    continue  # skip bad coords

                target[cell_y, cell_x, anchor_idx, 0] = 1.0  # confidence
                target[cell_y, cell_x, anchor_idx, 1:5] = torch.tensor([cx, cy, w, h], dtype=torch.float32)
                
                kpts = person_hands[kpts_key]
                keypoints = []

                # Keypoints and validity of keypoints extraction
                for i in range(0, len(kpts), 3):
                    x_kp = kpts[i] * scale_x
                    y_kp = kpts[i + 1] * scale_y
                    valid_kp = kpts[i + 2]
                    keypoints.extend([x_kp, y_kp, valid_kp])
                target[cell_y, cell_x, anchor_idx, 5:] = torch.tensor(keypoints, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        confidences = target[:, :, :, 0]
        bboxes = target[:, :, :, 1:5]
        keypoints = target[:, :, :, 5:]

        return image, confidences, bboxes, keypoints

