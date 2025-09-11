import torch
import torch.nn as nn


class MultiScaleDetector(nn.Module):
    """
    Backbone (224x224 输入)：
      Block1: Conv(3->32, s=1) -> BN -> ReLU -> Conv(32->64, s=2) -> BN -> ReLU   -> 112x112
      Block2: Conv(64->128, s=2) -> BN -> ReLU                                     -> 56x56   (Scale 1)
      Block3: Conv(128->256, s=2) -> BN -> ReLU                                    -> 28x28   (Scale 2)
      Block4: Conv(256->512, s=2) -> BN -> ReLU                                    -> 14x14   (Scale 3)

    Detection Head（每个尺度）：
      3x3 Conv(通道不变, s=1) + BN + ReLU -> 1x1 Conv(输出 A*(5+C))
    """
    def __init__(self, num_classes=3, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.pred_ch = num_anchors * (5 + num_classes)
        #（可选）记录每个尺度对应的步幅
        self.strides = [4, 8, 16]  # 224->56->28->14

        # -------- Backbone: stride=2 的卷积做下采样 --------
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # -------- Detection Heads: 3x3(保通道) -> 1x1(到 A*(5+C)) --------
        def make_head(in_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, self.pred_ch, kernel_size=1, stride=1, padding=0, bias=True),
            )

        self.head_s1 = make_head(128)  # 56x56
        self.head_s2 = make_head(256)  # 28x28
        self.head_s3 = make_head(512)  # 14x14

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B, 3, 224, 224]
        return:
          p1: [B, A*(5+C), 56, 56]
          p2: [B, A*(5+C), 28, 28]
          p3: [B, A*(5+C), 14, 14]
        """
        x = self.block1(x)   # [B, 64, 112,112]
        f1 = self.block2(x)  # [B,128, 56, 56] -> Scale 1
        f2 = self.block3(f1) # [B,256, 28, 28] -> Scale 2
        f3 = self.block4(f2) # [B,512, 14, 14] -> Scale 3

        p1 = self.head_s1(f1)
        p2 = self.head_s2(f2)
        p3 = self.head_s3(f3)
        return [p1, p2, p3]
