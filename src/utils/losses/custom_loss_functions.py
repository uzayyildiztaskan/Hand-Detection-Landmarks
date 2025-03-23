import torch.nn as nn
from src.utils.losses.wing_loss import WingLoss

keypoint_loss_fn = WingLoss(omega=10.0, epsilon=2.0)

def detection_loss(pred_bbox, true_bbox):
    return nn.SmoothL1Loss()(pred_bbox, true_bbox)

def keypoint_loss(pred_keypoints, true_keypoints):
    return keypoint_loss_fn(pred_keypoints, true_keypoints)