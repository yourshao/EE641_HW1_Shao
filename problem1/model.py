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
        self.pred_ch = num_anchors * (5 + num_classes)  # 4 bbox + 1 obj + C classes

        # ---------------- Backbone (4 blocks) ----------------
        # Block: Conv -> BN -> ReLU -> MaxPool(2x2, stride=2)
        # 输入 [B,3,224,224] 经过四次下采样到 112/56/28/14
        self.block1 = self._make_block(3,   64)   # -> [B,64,112,112]
        self.block2 = self._make_block(64,  128)  # -> [B,128,56,56]  (Scale 1)
        self.block3 = self._make_block(128, 256)  # -> [B,256,28,28]  (Scale 2)
        self.block4 = self._make_block(256, 512)  # -> [B,512,14,14]  (Scale 3)

        # ---------------- Detection heads (per scale) ----------------
        # 每个 head: 3x3 Conv(保持通道) -> 1x1 Conv(输出 num_anchors*(5+num_classes))
        self.head_s1 = self._make_head(128)
        self.head_s2 = self._make_head(256)
        self.head_s3 = self._make_head(512)

        self._init_weights()

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def _make_head(self, in_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, self.pred_ch, kernel_size=1, stride=1, padding=0),
        )

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
        Args:
            x: [B, 3, 224, 224]

        Returns:
            List[Tensor]: [p1, p2, p3]
              - p1: [B, A*(5+C), 56, 56]   (Scale 1)
              - p2: [B, A*(5+C), 28, 28]   (Scale 2)
              - p3: [B, A*(5+C), 14, 14]   (Scale 3)
        """
        x = self.block1(x)     # [B,64,112,112]
        f1 = self.block2(x)    # [B,128,56,56]   -> Scale 1
        f2 = self.block3(f1)   # [B,256,28,28]   -> Scale 2
        f3 = self.block4(f2)   # [B,512,14,14]   -> Scale 3

        p1 = self.head_s1(f1)  # [B, A*(5+C), 56, 56]
        p2 = self.head_s2(f2)  # [B, A*(5+C), 28, 28]
        p3 = self.head_s3(f3)  # [B, A*(5+C), 14, 14]

        return [p1, p2, p3]