import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

# --------------------------
# Detect Head v1
# --------------------------

class Detect(nn.Module):
    def __init__(self, in_channels, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        self.reg_branch = nn.Sequential(
                    Conv(in_channels, in_channels, k=3),
                    Conv(in_channels, in_channels, k=3),
                    nn.Conv2d(in_channels, 5, 1)  #regression + objness
                )
        self.cls_branch = nn.Sequential(
                    Conv(in_channels, in_channels, k=3),
                    Conv(in_channels, in_channels, k=3),
                    nn.Conv2d(in_channels, num_classes , 1)  #cls
                )
    def forward(self, feature):
        reg = self.reg_branch(feature)
        cls = self.cls_branch(feature)
        out = torch.cat([reg, cls], dim=1)
        return out


class DetectHead(nn.Module):
    """
    Detect Head: mỗi scale có 2 Conv3x3 (Ultralytics Conv)
    final conv ra num_classes + 5
    """
    def __init__(self, in_channels, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        self.heads = nn.ModuleList()
        for c in in_channels:
            self.heads.append(Detect(in_channels=c, num_classes=num_classes))

    def forward(self, features):
        preds = []
        for f, h in zip(features, self.heads):
            preds.append(h(f))
        return preds
