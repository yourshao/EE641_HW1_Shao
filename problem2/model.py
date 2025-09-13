import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ------------------------------
# 基础块：Conv -> BN -> ReLU
# ------------------------------
def conv_block(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ------------------------------
# Encoder
# ------------------------------
class Encoder(nn.Module):
    """
    input:  [B,C,128,128]
    output:  f1@[B,32,64,64], f2@[B,64,32,32], f3@[B,128,16,16], f4@[B,256,8,8]
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x); f1 = self.pool1(x)      # [B,32,64,64]
        x = self.conv2(f1); f2 = self.pool2(x)     # [B,64,32,32]
        x = self.conv3(f2); f3 = self.pool3(x)     # [B,128,16,16]
        x = self.conv4(f3); f4 = self.pool4(x)     # [B,256, 8, 8]
        return f1, f2, f3, f4


# ------------------------------
# Heatmap Head
# ------------------------------
class HeatmapHead(nn.Module):
    def __init__(self, num_keypoints: int):
        super().__init__()
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(32, num_keypoints, kernel_size=1, stride=1, padding=0)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor, f4: torch.Tensor) -> torch.Tensor:
        x = self.deconv4(f4)               # [B,128,16,16]
        x = torch.cat([x, f3], dim=1)      # [B,256,16,16]

        x = self.deconv3(x)                # [B,64,32,32]
        x = torch.cat([x, f2], dim=1)      # [B,128,32,32]

        x = self.deconv2(x)                # [B,32,64,64]
        heatmaps = self.final(x)           # [B,K,64,64]
        return heatmaps


# ------------------------------
# Regression Head
# ------------------------------
class RegressionHead(nn.Module):
    def __init__(self, num_keypoints: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.do1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 64)
        self.do2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(64, num_keypoints * 2)

    def forward(self, f4: torch.Tensor) -> torch.Tensor:
        z = self.gap(f4).flatten(1)        # [B,256]
        z = self.do1(F.relu(self.fc1(z)))
        z = self.do2(F.relu(self.fc2(z)))
        coords = torch.sigmoid(self.fc3(z))  # [B,2K] ∈ [0,1]
        return coords


# ------------------------------
# HeatmapNet (独立模型)
# ------------------------------
class HeatmapNet(nn.Module):
    def __init__(self, num_keypoints: int = 5, in_channels: int = 1):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.hm_head = HeatmapHead(num_keypoints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1, f2, f3, f4 = self.encoder(x)
        return self.hm_head(f1, f2, f3, f4)


# ------------------------------
# RegressionNet (独立模型)
# ------------------------------
class RegressionNet(nn.Module):
    def __init__(self, num_keypoints: int = 5, in_channels: int = 1):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.reg_head = RegressionHead(num_keypoints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, f4 = self.encoder(x)
        return self.reg_head(f4)
