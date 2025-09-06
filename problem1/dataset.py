import os

import numpy as np
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

        # 读取 COCO JSON
        with open(annotation_file, "r") as f:
            coco = json.load(f)

        # 类别映射：优先用名字 circle/square/triangle -> 0/1/2
        name2idx = {"circle": 0, "square": 1, "triangle": 2}
        self.catid_to_idx = {}
        for c in coco.get("categories", []):
            cid = c["id"]
            name = str(c.get("name", "")).strip().lower()
            if name in name2idx:
                self.catid_to_idx[cid] = name2idx[name]
        # 若没有名字映射，则假定 category_id 已经是 0/1/2
        # （在 __getitem__ 时若找不到映射，会尝试 int(category_id)）

        # 建立 image_id -> 标注列表
        anns_by_img = {}
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            anns_by_img.setdefault(img_id, []).append(ann)

        # 预构建样本列表
        self.samples = []
        for img_info in coco.get("images", []):
            img_id = img_info["id"]
            file_name = img_info["file_name"]
            path = os.path.join(self.image_dir, file_name)

            width = img_info.get("width", 224)
            height = img_info.get("height", 224)

            boxes = []
            labels = []
            for a in anns_by_img.get(img_id, []):
                # COCO: bbox = [x, y, w, h]
                x, y, w, h = a["bbox"]
                x1 = float(x)
                y1 = float(y)
                x2 = float(x) + float(w)
                y2 = float(y) + float(h)

                # clamp 到图像范围（稳妥）
                x1 = max(0.0, min(x1, width - 1))
                y1 = max(0.0, min(y1, height - 1))
                x2 = max(0.0, min(x2, width - 1))
                y2 = max(0.0, min(y2, height - 1))

                if x2 <= x1 or y2 <= y1:
                    continue  # 跳过退化框

                cat = a["category_id"]
                if cat in self.catid_to_idx:
                    label = self.catid_to_idx[cat]
                else:
                    # 尝试直接转成 0/1/2
                    label = int(cat)

                boxes.append([x1, y1, x2, y2])
                labels.append(label)

            boxes_np = np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
            labels_np = np.asarray(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

            self.samples.append(
                {
                    "path": path,
                    "boxes": boxes_np,
                    "labels": labels_np,
                }
            )

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return a sample from the dataset.

        Returns:
            image: Tensor of shape [3, H, W]
            targets: Dict containing:
                - boxes: Tensor of shape [N, 4] in [x1, y1, x2, y2] format
                - labels: Tensor of shape [N] with class indices (0, 1, 2)
        """
        sample = self.samples[idx]

        # 读图（PIL -> RGB）
        img = Image.open(sample["path"]).convert("RGB")

        # 可选变换：支持返回 PIL 或直接返回 Tensor
        if self.transform is not None:
            img_t = self.transform(img)
            if not isinstance(img_t, torch.Tensor):
                arr = np.asarray(img_t, dtype=np.float32)  # [H,W,3]
                img_t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        else:
            arr = np.asarray(img, dtype=np.float32)  # [H,W,3]
            img_t = torch.from_numpy(arr).permute(2, 0, 1).float().contiguous() / 255.0

        boxes_t = torch.from_numpy(sample["boxes"]).to(dtype=torch.float32)
        labels_t = torch.from_numpy(sample["labels"]).to(dtype=torch.int64)

        targets = {
            "boxes": boxes_t,
            "labels": labels_t,
        }
        return img_t, targets

