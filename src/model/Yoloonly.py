
# 3. Import các module
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
from src.config import *
from src.preprocessing import *
from ultralytics import YOLO
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random
from src.config import *
from src.preprocessing import *
import numpy as np
import sys

class modelYoloonly():
    def __init__(self, pretrained_weights="yolov8l.pt"):
        super().__init__()
        self.model = YOLO(pretrained_weights)
        # Thay số class
        self.model.model.nc = 4
        self.model.model.names = ["Glioma", "Meningioma", "NoTumor", "Pituitary"]

        # Hàm reset parameters
        def reset_recursive(m):
            if len(list(m.children())) == 0 and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            else:
                for child in m.children():
                    reset_recursive(child)

        # Backbone blocks 0–9
        backbone_blocks = self.model.model.model[:10]

        # Freeze: 0,1,3
        for idx in [0,1,3]:
            for param in backbone_blocks[idx].parameters():
                param.requires_grad = False

        # Fine-tune: 4,5
        for idx in [4,5]:
            for param in backbone_blocks[idx].parameters():
                param.requires_grad = True

        # Reset/train lại: 2,6,7,8,9
        for idx in [2,6,7,8,9]:
            block = backbone_blocks[idx]
            for param in block.parameters():
                param.requires_grad = True
            reset_recursive(block)

        # Neck 10–21
        neck_blocks = self.model.model.model[10:22]
        for block in neck_blocks:
            for param in block.parameters():
                param.requires_grad = True
            reset_recursive(block)

        print("Backbone + Neck đã được thiết lập theo yêu cầu (freeze, fine-tune, reset)")

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run():
    SEED = 42
    set_seed(SEED)

    # --- Tạo generator cho DataLoader ---
    g = torch.Generator()
    g.manual_seed(SEED)

    # --- Cấu hình ---
    yaml_data = "data/brain_tumor.yaml"  # file YAML bạn đã tạo với train/val/test
    NUM_EPOCHS = 100
    BATCH_SIZE = 8

    # --- Khởi tạo model ---
    model_wrapper = modelYoloonly()
    model = model_wrapper.model  # YOLO object

    
    # --- Huấn luyện ---
    results = model.train(
    data=yaml_data,
    epochs=NUM_EPOCHS,
    batch=BATCH_SIZE,
    patience=50,
    save=True,
    project="experiments",
    name="BrainTumor_yoloonly",
    exist_ok=True,
    workers=0,          # để reproducible
    augment=False,       # tắt augmentation
    imgsz = 640,
    device='cuda:0'
    )


    # --- Đánh giá cuối cùng trên test set ---
    # YOLOv8 sẽ đọc 'test:' trong YAML
    metrics = model.val(data="data/total_test_list.txt", imgsz = 640, device='cuda:0', batch=BATCH_SIZE) # validate trên val, sẽ tự dùng test nếu bạn truyền test=None
    print("Metrics trên test set:")
    print(f"Precision: {metrics.box.precision:.4f}")
    print(f"Recall   : {metrics.box.recall:.4f}")
    print(f"mAP@0.5  : {metrics.box.map:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map50_95:.4f}")

    
