import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from src.model.DetectHead import DetectHead

class BrainTumorv1(nn.Module):
    def __init__(self, num_classes=4, pretrained_weights="yolov8l.pt"):
        super().__init__()

        self.extend_1 = Conv(1, 1, k=3)
        self.extend_2 = Conv(1, 1, k=3)
        yolo_model = YOLO(pretrained_weights)
        self.backbone_block1 = nn.Sequential(*yolo_model.model.model[:5])   # block 0–4
        self.backbone_block2 = nn.Sequential(*yolo_model.model.model[5:7])   # block 5-6
        self.backbone_block3 = nn.Sequential(*yolo_model.model.model[7:10])   # block 7-9

        self.neck = nn.ModuleList()
        for i in range(10, 22):  # nếu neck gồm 12 block (0→11)
            self.neck.append(yolo_model.model.model[i])
        
        in_channels = [256, 512, 512]
        self.head = DetectHead(in_channels, num_classes)
        self.freeze_backbone_neck()

    def freeze_backbone_neck(self):
        for param in list(self.backbone_block1.parameters()) + \
                    list(self.backbone_block2.parameters()) + \
                    list(self.backbone_block3.parameters()) + \
                    list(self.neck.parameters()):
            param.requires_grad = False
        print("Backbone và Neck đã được freeze (đóng băng).")

    def forward(self, x):
        # --- Backbone ---
        if x.shape[1] == 1:
            f1 = self.extend_1(x)
            f2 = self.extend_2(f1)
            x = torch.cat([x, f1, f2], dim=1)

        feature_1 = self.backbone_block1(x)  # shallow
        feature_2 = self.backbone_block2(feature_1)  # mid
        feature_3 = self.backbone_block3(feature_2)  # deep

        # --- Neck ---
        # 0: Upsample, 1: Concat, 2: C2f
        up1 = self.neck[0](feature_3)
        cat1 = self.neck[1]([up1, feature_2])
        p3 = self.neck[2](cat1)

        # 3: Upsample, 4: Concat, 5: C2f
        up2 = self.neck[3](p3)
        cat2 = self.neck[4]([up2, feature_1])
        p2 = self.neck[5](cat2)

        # 6: Downsample, 7: Concat, 8: C2f
        down1 = self.neck[6](p2)
        cat3 = self.neck[7]([down1, p3])
        n3 = self.neck[8](cat3)

        # 9: Downsample, 10: Concat, 11: C2f
        down2 = self.neck[9](n3)
        cat4 = self.neck[10]([down2, feature_3])
        n4 = self.neck[11](cat4)

        # Return các feature map output cuối cùng (3 scale cho head)

        # --- Head ---
        y = self.head([p2, n3, n4])
        return y
