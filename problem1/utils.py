import torch
import numpy as np

# utils.py
import torch


def generate_anchors(feature_map_sizes, anchor_scales, image_size=224):
    """
    Generate anchors for multiple feature maps.

    Args:
        feature_map_sizes: List[(H, W)] for each feature map
        anchor_scales: List[List[int/float]], side-length (pixels) for each fmap
        image_size: Input image size (assume square, e.g., 224)

    Returns:
        anchors_list: List[Tensor], each [H*W*num_anchors, 4] (x1,y1,x2,y2)
    """
    assert len(feature_map_sizes) == len(anchor_scales), \
        "feature_map_sizes and anchor_scales must have the same length"

    anchors_list = []
    for (H, W), scales in zip(feature_map_sizes, anchor_scales):
        # stride per cell (allow non-square fmap)
        stride_y = float(image_size) / float(H)
        stride_x = float(image_size) / float(W)

        ys = (torch.arange(H, dtype=torch.float32) + 0.5) * stride_y
        xs = (torch.arange(W, dtype=torch.float32) + 0.5) * stride_x
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # [H,W], [H,W]

        # Only 1:1 aspect ratio as required; scale is side length in pixels
        per_scale = []
        for s in scales:
            s = float(s)
            w = torch.full_like(gx, s)
            h = torch.full_like(gy, s)
            x1 = gx - 0.5 * w
            y1 = gy - 0.5 * h
            x2 = gx + 0.5 * w
            y2 = gy + 0.5 * h

            # (optional) clamp into image bounds (safe for training)
            x1 = x1.clamp(min=0.0, max=image_size - 1.0)
            y1 = y1.clamp(min=0.0, max=image_size - 1.0)
            x2 = x2.clamp(min=0.0, max=image_size - 1.0)
            y2 = y2.clamp(min=0.0, max=image_size - 1.0)

            boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [H,W,4]
            per_scale.append(boxes)

        # stack across anchors -> [A,H,W,4] -> [H,W,A,4] -> [H*W*A,4]
        anchors = torch.stack(per_scale, dim=0).permute(1, 2, 0, 3).reshape(-1, 4)
        anchors_list.append(anchors)

    return anchors_list


def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: Tensor [N, 4] in (x1,y1,x2,y2)
        boxes2: Tensor [M, 4] in (x1,y1,x2,y2)

    Returns:
        iou: Tensor [N, M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    # [N,1,4] & [1,M,4]
    b1 = boxes1.unsqueeze(1)  # [N,1,4]
    b2 = boxes2.unsqueeze(0)  # [1,M,4]

    inter_x1 = torch.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = torch.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = torch.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = torch.minimum(b1[..., 3], b2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
    iou = inter_area / (union + 1e-6)
    return iou


def match_anchors_to_targets(anchors, target_boxes, target_labels,
                             pos_threshold=0.5, neg_threshold=0.3):
    """
    Match anchors to ground truth boxes for ONE image.

    Args:
        anchors: Tensor [A,4]
        target_boxes: Tensor [T,4]
        target_labels: Tensor [T] (class indices 0..C-1)
        pos_threshold: IoU >= this -> positive
        neg_threshold: IoU <  this -> negative (others are ignore)

    Returns:
        matched_labels: Tensor [A] (0=background, 1..C=classes+1)
        matched_boxes:  Tensor [A,4]
        pos_mask:       Bool Tensor [A]
        neg_mask:       Bool Tensor [A]
    """
    A = anchors.shape[0]
    device = anchors.device

    # Default outputs
    matched_boxes = torch.zeros((A, 4), dtype=torch.float32, device=device)
    matched_labels = torch.zeros((A,), dtype=torch.long, device=device)  # 0 = background
    pos_mask = torch.zeros((A,), dtype=torch.bool, device=device)
    neg_mask = torch.ones((A,), dtype=torch.bool, device=device)  # default all neg, will fix below

    # No targets -> all negatives
    if target_boxes is None or target_boxes.numel() == 0:
        return matched_labels, matched_boxes, pos_mask, neg_mask

    # IoU matrix [A,T]
    ious = compute_iou(anchors, target_boxes.to(device))
    iou_max, gt_idx = ious.max(dim=1)  # per-anchor best GT

    # Initial masks
    pos_mask = iou_max >= pos_threshold
    neg_mask = iou_max < neg_threshold
    ignore_mask = ~(pos_mask | neg_mask)

    # Force at least one anchor per GT (bipartite-style)
    # For each GT, find its best anchor and mark positive.
    best_anchor_for_each_gt = ious.argmax(dim=0)  # [T]
    pos_mask[best_anchor_for_each_gt] = True
    neg_mask[best_anchor_for_each_gt] = False
    ignore_mask[best_anchor_for_each_gt] = False
    gt_idx[best_anchor_for_each_gt] = torch.arange(target_boxes.shape[0], device=device)

    # Fill matched boxes/labels for positives
    if pos_mask.any():
        assigned = gt_idx[pos_mask]
        matched_boxes[pos_mask] = target_boxes[assigned].to(device)
        # labels in dataset are 0..C-1; required output is 1..C (0=background)
        matched_labels[pos_mask] = (target_labels[assigned].to(device) + 1).long()

    # Negatives keep label 0 and boxes as zeros; ignore are neither pos nor neg
    return matched_labels, matched_boxes, pos_mask, neg_mask