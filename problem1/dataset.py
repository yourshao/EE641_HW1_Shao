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
            annotation_file: Path to JSON annotations (bbox as [x1,y1,x2,y2])
            transform: Optional transform to apply to images
        """
        self.image_dir = image_dir
        self.transform = transform

        # 读取 JSON
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
                # ==== 这里改为读取 xyxy ====
                bx = a["bbox"]
                if not (isinstance(bx, (list, tuple)) and len(bx) == 4):
                    continue  # 跳过异常框

                x1, y1, x2, y2 = map(float, bx)

                # 若出现顺序颠倒，做纠正（稳妥处理）
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1

                # clamp 到图像范围
                x1 = max(0.0, min(x1, width - 1))
                y1 = max(0.0, min(y1, height - 1))
                x2 = max(0.0, min(x2, width - 1))
                y2 = max(0.0, min(y2, height - 1))

                # 跳过退化框
                if x2 <= x1 or y2 <= y1:
                    continue

                # 处理类别
                cat = a["category_id"]
                if cat in self.catid_to_idx:
                    label = self.catid_to_idx[cat]
                else:
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
        return len(self.samples)

    def __getitem__(self, idx):
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

