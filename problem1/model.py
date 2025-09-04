import torch
import torch.nn as nn

class MultiScaleDetector(nn.Module):
    def __init__(self, num_classes=3, num_anchors=3):
        """
        Initialize the multi-scale detector.

        Args:
            num_classes: Number of object classes (not including background)
            num_anchors: Number of anchors per spatial location
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Feature extraction backbone
        # Extract features at 3 different scales

        # Detection heads for each scale
        # Each head outputs: [batch, num_anchors * (4 + 1 + num_classes), H, W]
        pass

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, 3, 224, 224]

        Returns:
            List of 3 tensors (one per scale), each containing predictions
            Shape: [batch, num_anchors * (5 + num_classes), H, W]
            where 5 = 4 bbox coords + 1 objectness score
        """
        pass