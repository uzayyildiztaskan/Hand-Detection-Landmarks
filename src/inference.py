import torch
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.model.landmarkmodel import HandLandmarkModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = 'outputs/checkpoints/checkpoint_epoch_15.pt'
IMAGE_PATH = 'dataset/FreiHAND_pub_v2/training/rgb/00003168.jpg'
IMG_SIZE = 512

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

model = HandLandmarkModel().to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

image = Image.open(IMAGE_PATH).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred_bbox, pred_keypoints = model(input_tensor)

pred_bbox = pred_bbox.squeeze().cpu().numpy() * IMG_SIZE
pred_keypoints = pred_keypoints.squeeze().cpu().numpy() * IMG_SIZE


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9,10), (10,11), (11,12),     # Middle
    (0,13), (13,14), (14,15), (15,16),    # Ring
    (0,17), (17,18), (18,19), (19,20)     # Pinky
]


img = input_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
img = (img * 0.5 + 0.5) * 255
img = img.astype(np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img = cv2.resize(img, (512, 512))

x_min, y_min, x_max, y_max = pred_bbox.astype(int)
cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

for x, y in pred_keypoints:
    cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)

for connection in HAND_CONNECTIONS:
    start_idx, end_idx = connection
    x_start, y_start = pred_keypoints[start_idx]
    x_end, y_end = pred_keypoints[end_idx]
    cv2.line(img, (int(x_start), int(y_start)), (int(x_end), int(y_end)), (255, 0, 0), 2)

plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()