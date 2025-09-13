import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class KeypointDataset(Dataset):
    def __init__(self, image_dir, annotation_file, output_type='heatmap',
                 heatmap_size=64, sigma=2.0):

        assert output_type in ('heatmap', 'regression')
        self.image_dir = image_dir
        self.output_type = output_type
        self.heatmap_size = int(heatmap_size)
        self.sigma = float(sigma)

        self.input_h, self.input_w = 128, 128

        with open(annotation_file, 'r') as f:
            data = json.load(f)

        images = data.get('images', [])
        self.keypoint_names = data.get('keypoint_names', None)

        self.samples = []
        for rec in images:
            fname = rec.get('file_name')
            kps = rec.get('keypoints')
            if fname is None or kps is None:
                continue
            kps = np.asarray(kps, dtype=np.float32)  # (K,2)
            self.samples.append({
                'path': os.path.join(self.image_dir, fname),
                'keypoints': kps
            })

        if len(self.samples) == 0:
            raise RuntimeError(f'No valid samples parsed from {annotation_file}')

    def __len__(self):
        return len(self.samples)

    # ============== Heatmap  ==============
    def generate_heatmap(self, keypoints, height, width):
        """
        keypoints: (K,2) in (x,y) of heatmap coordinate system
        return:   torch.FloatTensor (K, H, W)
        """
        K = keypoints.shape[0]
        heatmaps = np.zeros((K, height, width), dtype=np.float32)

        yy, xx = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing='ij'
        )

        s2 = 2.0 * (self.sigma ** 2)

        for i, (x, y) in enumerate(keypoints):
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            if x < 0 or y < 0 or x > width - 1 or y > height - 1:
                continue
            g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / s2)
            g /= (g.max() + 1e-8)
            heatmaps[i] = g

        return torch.from_numpy(heatmaps)

    def _load_image_128(self, path):

        img = Image.open(path).convert('L')
        W0, H0 = img.size  # PIL: (W,H)
        img = img.resize((self.input_w, self.input_h), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr[None, ...]  # [1,H,W]
        return torch.from_numpy(arr), H0, W0

    def __getitem__(self, idx):
        rec = self.samples[idx]
        img_t, H0, W0 = self._load_image_128(rec['path'])  # [1,128,128]

        kps = rec['keypoints'].astype(np.float32).copy()  # (K,2)
        sx = self.input_w / max(W0, 1e-6)
        sy = self.input_h / max(H0, 1e-6)
        kps[:, 0] *= sx
        kps[:, 1] *= sy

        if self.output_type == 'heatmap':

            sx_hm = self.heatmap_size / self.input_w
            sy_hm = self.heatmap_size / self.input_h
            kps_hm = kps.copy()
            kps_hm[:, 0] *= sx_hm
            kps_hm[:, 1] *= sy_hm
            target = self.generate_heatmap(kps_hm, self.heatmap_size, self.heatmap_size)  # (K,H,W)
            return img_t, target  # image: [1,128,128], target: [K,64,64]

        else:  # 'regression'

            kp_norm = kps.copy()
            kp_norm[:, 0] = np.clip(kp_norm[:, 0] / self.input_w, 0.0, 1.0)
            kp_norm[:, 1] = np.clip(kp_norm[:, 1] / self.input_h, 0.0, 1.0)
            target = torch.from_numpy(kp_norm.reshape(-1))  # [2K]
            return img_t, target



