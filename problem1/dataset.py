import torch
from torch.utils.data import Dataset
from PIL import Image
import json

class ShapeDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        """
        Initialize the dataset.

        Args:
            image_dir: Path to directory containing images
            annotation_file: Path to COCO-style JSON annotations
            transform: Optional transform to apply to images
        """
        self.image_dir = image_dir
        self.transform = transform
        # Load and parse annotations
        # Store image paths and corresponding annotations
        pass

    def __len__(self):
        """Return the total number of samples."""
        pass

    def __getitem__(self, idx):
        """
        Return a sample from the dataset.

        Returns:
            image: Tensor of shape [3, H, W]
            targets: Dict containing:
                - boxes: Tensor of shape [N, 4] in [x1, y1, x2, y2] format
                - labels: Tensor of shape [N] with class indices (0, 1, 2)
        """
        pass