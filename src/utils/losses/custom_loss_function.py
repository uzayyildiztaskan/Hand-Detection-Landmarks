import torch.nn.functional as F
import torch

def detection_loss(preds, confidences, bboxes, keypoints, num_keypoints=21):
    """
    preds: [B, S, S, num_anchors, output_features] model output
    confidences: [B, S, S, num_anchors]
    bboxes: [B, S, S, num_anchors, 4]
    keypoints: [B, S, S, num_anchors, num_keypoints * 3]
    """

    pred_conf = preds[..., 0]
    pred_bboxes = preds[..., 1:5]
    pred_keypoints = preds[..., 5:]

    ### 1. Confidence loss
    conf_loss = F.binary_cross_entropy_with_logits(pred_conf, confidences)

    ### 2. BBox loss (only where there's an object)
    object_mask = confidences > 0  # shape: [B, S, S, num_anchors]

    bbox_loss = F.smooth_l1_loss(pred_bboxes[object_mask], bboxes[object_mask])

    ### 3. Keypoints loss (only for visible keypoints)
    pred_kpts = pred_keypoints.view(pred_keypoints.shape[0], pred_keypoints.shape[1], pred_keypoints.shape[2], pred_keypoints.shape[3], num_keypoints, 3)
    gt_kpts = keypoints.view(keypoints.shape[0], keypoints.shape[1], keypoints.shape[2], keypoints.shape[3], num_keypoints, 3)

    pred_xy = pred_kpts[..., :2]  # (x, y)
    gt_xy = gt_kpts[..., :2]
    gt_v = gt_kpts[..., 2]        # visibility

    visibility_mask = gt_v > 0  # Only consider visible keypoints

    if visibility_mask.sum() > 0:
        keypoint_loss = F.smooth_l1_loss(pred_xy[visibility_mask], gt_xy[visibility_mask])
    else:
        keypoint_loss = torch.tensor(0.0, device=preds.device)

    ### 4. Total loss
    total_loss = (1.0 * conf_loss) + (5.0 * bbox_loss) + (2.0 * keypoint_loss)

    return total_loss, {
        "conf_loss": conf_loss.item(),
        "bbox_loss": bbox_loss.item(),
        "keypoint_loss": keypoint_loss.item()
    }