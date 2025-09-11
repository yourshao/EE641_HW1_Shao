import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- 基础块：Conv -> BN -> ReLU ----------
def conv_block(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """
    共享编码器（两个网络都用）：
    输入: [B,1,128,128]
    输出: f1@[B,32,64,64], f2@[B,64,32,32], f3@[B,128,16,16], f4@[B,256,8,8]
    """
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(1, 32)    # 128 -> 128
        self.pool1 = nn.MaxPool2d(2)      # 128 -> 64

        self.conv2 = conv_block(32, 64)   # 64 -> 64
        self.pool2 = nn.MaxPool2d(2)      # 64 -> 32

        self.conv3 = conv_block(64, 128)  # 32 -> 32
        self.pool3 = nn.MaxPool2d(2)      # 32 -> 16

        self.conv4 = conv_block(128, 256) # 16 -> 16
        self.pool4 = nn.MaxPool2d(2)      # 16 -> 8

    def forward(self, x):
        x = self.conv1(x); f1 = self.pool1(x)  # [B,32,64,64]
        x = self.conv2(f1); f2 = self.pool2(x) # [B,64,32,32]
        x = self.conv3(f2); f3 = self.pool3(x) # [B,128,16,16]
        x = self.conv4(f3); f4 = self.pool4(x) # [B,256,8,8]
        return f1, f2, f3, f4


class HeatmapNet(nn.Module):
    """
    输出热力图 [B, K, 64, 64]
    反卷积配置严格按你的规格：
      Deconv4: 256->128, 8->16
      +concat f3 (128) -> 256
      Deconv3: 256->64, 16->32
      +concat f2 (64)  -> 128
      Deconv2: 128->32, 32->64
      Final:  32->K (1x1 conv, 无激活)
    """
    def __init__(self, num_keypoints=5):
        super().__init__()
        self.num_keypoints = num_keypoints

        # 共享编码器
        self.encoder = Encoder()

        # 反卷积：使用 kernel_size=2, stride=2 精确上采样到 2×
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

    def forward(self, x):
        # 编码
        f1, f2, f3, f4 = self.encoder(x)   # sizes: 64,32,16,8

        # 解码 + 跳连
        x = self.deconv4(f4)               # [B,128,16,16]
        x = torch.cat([x, f3], dim=1)      # [B,256,16,16]

        x = self.deconv3(x)                # [B,64,32,32]
        x = torch.cat([x, f2], dim=1)      # [B,128,32,32]

        x = self.deconv2(x)                # [B,32,64,64]

        heatmaps = self.final(x)           # [B,K,64,64] (无激活，配 MSE/Huber)
        return heatmaps


class RegressionNet(nn.Module):
    """
    直接坐标回归 [B, 2K]，Sigmoid 归一化到 [0,1]
    头部：
      GAP -> FC(256->128)+ReLU+Dropout(0.5)
          -> FC(128->64)+ReLU+Dropout(0.5)
          -> FC(64->2K)+Sigmoid
    """
    def __init__(self, num_keypoints=5):
        super().__init__()
        self.num_keypoints = num_keypoints

        # 共享编码器
        self.encoder = Encoder()

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 全连接头
        self.fc1 = nn.Linear(256, 128)
        self.do1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 64)
        self.do2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(64, num_keypoints * 2)

    def forward(self, x):
        # 仅用到最深特征 f4（[B,256,8,8]）
        _, _, _, f4 = self.encoder(x)
        z = self.gap(f4).flatten(1)        # [B,256]

        z = F.relu(self.fc1(z))
        z = self.do1(z)
        z = F.relu(self.fc2(z))
        z = self.do2(z)

        coords = torch.sigmoid(self.fc3(z))  # [B, 2K] in [0,1]
        return coords
