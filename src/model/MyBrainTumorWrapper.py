import torch
import torch.nn as nn
import torch.optim as optim
from src.model.BrainTumorWrapper import BrainTumorWrapper
from ultralytics.utils.loss import VarifocalLoss
from src.config import *
from src.model.LossWrapper import *
class MyBrainTumorWrapper:
    def __init__(self, model, device=None, CKPT_PATH=None, img_size=img_size, num_classes=4):
        self.wrapper = BrainTumorWrapper(model = model, device=device, CKPT_PATH=CKPT_PATH, num_classes = num_classes, img_size = img_size)
        self.num_classes = num_classes
        self.img_size = img_size
        # ======= Training phases =======
        # Loss functions
        loss_mse = nn.MSELoss()
        loss_ce = CEWrapper()
        loss_bce = nn.BCEWithLogitsLoss()
        # from src.losses import CIoULoss, GIoULoss, VarifocalLoss
        loss_ciou = CIoULossWrapper(device, img_size)
        loss_giou = GIoULossWrapper(device, img_size)
        loss_vfl = VFLWrapper(device, num_classes)

        # Phase 1: Warm-up
        optimizer_warmup = optim.SGD(params=model.parameters(), lr=1e-3)
        scheduler_warmup = optim.lr_scheduler.LinearLR(
        optimizer_warmup, start_factor=0.1, end_factor=1.0, total_iters=100
        )

        # Phase 2: Acceleration
        optimizer_accel = optim.SGD(params=model.parameters(), lr=1e-2, momentum=0.9)
        scheduler_accel = None  # giữ lr cố định

        # Phase 3: Sustain
        optimizer_sustain = optim.AdamW(params=model.parameters(), lr=1e-4)
        scheduler_sustain = optim.lr_scheduler.CosineAnnealingLR(optimizer_sustain, T_max=50)

        # Training phases dict
        self.training_phases = {
            "Warm-up": {
            "loss_reg": loss_mse,
            "loss_cls": loss_ce,
            "loss_objness": loss_bce,
            "optimizer": optimizer_warmup,
            "scheduler": scheduler_warmup,
        },
        "Acceleration": {
            "loss_reg": loss_ciou,
            "loss_cls": loss_vfl,
            "loss_objness": loss_bce,
            "optimizer": optimizer_accel,
            "scheduler": scheduler_accel,
        },
        "Sustain": {
            "loss_reg": loss_giou,
            "loss_cls": loss_vfl,
            "loss_objness": loss_bce,
            "optimizer": optimizer_sustain,
            "scheduler": scheduler_sustain,
        }
    }

    def fit(self, train_loader, val_loader=None, epochs=10, patience=3, mode = "Warm-up"):
        self.wrapper.compile(optimizer = self.training_phases[mode]["optimizer"], 
                            reg_loss = self.training_phases[mode]["loss_reg"], 
                            cls_loss = self.training_phases[mode]["loss_cls"],
                            objness_loss= self.training_phases[mode]["loss_objness"],
                            scheduler = self.training_phases[mode]["scheduler"])
        self.wrapper.fit(train_loader, val_loader, epochs, patience)

    def evaluate(self, val_loader, verbose=True):
        return self.wrapper.evaluate(val_loader, verbose)

    def _load_checkpoint(self, path):
        self.wrapper._load_checkpoint(path)

    def save_model(self, path, epoch=None):
        self.wrapper.save_model(path, epoch)

    def predict(self, path):
        return self.wrapper.predict(path)
    
    def img_predict(self, path, class_names=class_names):
        self.wrapper.img_predict(path, class_names)

