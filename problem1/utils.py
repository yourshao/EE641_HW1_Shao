import torch
import numpy as np

def generate_anchors(feature_map_sizes, anchor_scales, image_size=224):
    """
    Generate anchors for multiple feature maps.

    Args:
        feature_map_sizes: List of (H, W) tuples for each feature map
        anchor_scales: List of lists, scales for each feature map
        image_size: Input image size

    Returns:
        anchors: List of tensors, each of shape [H*W*num_anchors, 4]
                 in [x1, y1, x2, y2] format
    """
    # For each feature map:
    # 1. Create grid of anchor centers
    # 2. Generate anchors with specified scales and ratios
    # 3. Convert to absolute coordinates
    pass

def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: Tensor of shape [N, 4]
        boxes2: Tensor of shape [M, 4]

    Returns:
        iou: Tensor of shape [N, M]
    """
    pass

def match_anchors_to_targets(anchors, target_boxes, target_labels,
                             pos_threshold=0.5, neg_threshold=0.3):
    """
    Match anchors to ground truth boxes.

    Args:
        anchors: Tensor of shape [num_anchors, 4]
        target_boxes: Tensor of shape [num_targets, 4]
        target_labels: Tensor of shape [num_targets]
        pos_threshold: IoU threshold for positive anchors
        neg_threshold: IoU threshold for negative anchors

    Returns:
        matched_labels: Tensor of shape [num_anchors]
                       (0: background, 1-N: classes)
        matched_boxes: Tensor of shape [num_anchors, 4]
        pos_mask: Boolean tensor indicating positive anchors
        neg_mask: Boolean tensor indicating negative anchors
    """
    pass