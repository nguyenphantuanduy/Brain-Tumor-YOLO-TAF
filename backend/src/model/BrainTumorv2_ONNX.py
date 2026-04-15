import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from src.model.DetectHead import DetectHead


class BrainTumorv2_ONNX(nn.Module):
    def __init__(self, num_classes=4, pretrained_weights="yolov8l.pt"):
        super().__init__()

        # ---------------- backbone ----------------
        self.extend_1 = Conv(1, 1, k=3)
        self.extend_2 = Conv(1, 1, k=3)

        yolo_model = YOLO(pretrained_weights)

        self.backbone_block1 = nn.Sequential(*yolo_model.model.model[:5])
        self.backbone_block2 = nn.Sequential(*yolo_model.model.model[5:7])
        self.backbone_block3 = nn.Sequential(*yolo_model.model.model[7:10])

        # ---------------- neck ----------------
        self.neck = nn.ModuleList()
        for i in range(10, 22):
            self.neck.append(yolo_model.model.model[i])

        # ---------------- head ----------------
        in_channels = [256, 512, 512]
        self.head = DetectHead(in_channels, num_classes)

    # --------------------------------------------------
    # helper: safe concat for ONNX graph
    # --------------------------------------------------
    def _cat(self, a, b, dim=1):
        return torch.cat((a, b), dim=dim)

    # --------------------------------------------------
    def forward(self, x):

        # -------- input expand --------
        if x.shape[1] == 1:
            f1 = self.extend_1(x)
            f2 = self.extend_2(f1)
            x = torch.cat((x, f1, f2), dim=1)

        # -------- backbone --------
        f1 = self.backbone_block1(x)
        f2 = self.backbone_block2(f1)
        f3 = self.backbone_block3(f2)

        # -------- neck --------

        # P3
        up1 = self.neck[0](f3)
        cat1 = self._cat(up1, f2)
        p3 = self.neck[2](cat1)

        # P2
        up2 = self.neck[3](p3)
        cat2 = self._cat(up2, f1)
        p2 = self.neck[5](cat2)

        # N3
        down1 = self.neck[6](p2)
        cat3 = self._cat(down1, p3)
        n3 = self.neck[8](cat3)

        # N4
        down2 = self.neck[9](n3)
        cat4 = self._cat(down2, f3)
        n4 = self.neck[11](cat4)

        # -------- head --------
        y = self.head([p2, n3, n4])

        return y