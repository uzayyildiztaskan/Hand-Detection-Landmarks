import torch

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
    correct = (dists < (threshold*256)).float().mean().item()
    return correct