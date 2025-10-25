import torch
import torch.nn as nn
from torchvision.ops import box_iou, distance_box_iou_loss, generalized_box_iou_loss
from ultralytics.utils.loss import VarifocalLoss
from src.utils import yolo_to_xyxy  # phải trả về (N,4) nếu input (x: N, y: N, w: N, h: N)

# ================== CIoU Loss ==================
class CIoULossWrapper(nn.Module):
    def __init__(self, device, img_size):
        super().__init__()
        self.device = device
        self.img_size = img_size

    def forward(self, pred, target):
        """
        pred, target: (4, N) -> x,y,w,h
        """
        pred = pred.clone()
        target = target.clone()

        pred_xyxy = yolo_to_xyxy(pred[0,:], pred[1,:], pred[2,:], pred[3,:], self.img_size).to(self.device)
        target_xyxy = yolo_to_xyxy(target[0,:], target[1,:], target[2,:], target[3,:], self.img_size).to(self.device)

        # Clamp giá trị để tránh box lỗi
        pred_xyxy = torch.clamp(pred_xyxy, min=0.0)
        target_xyxy = torch.clamp(target_xyxy, min=0.0)

        diou_loss = distance_box_iou_loss(pred_xyxy, target_xyxy, reduction='none')
        iou = box_iou(pred_xyxy, target_xyxy).diag().clamp(0, 1)  # tránh >1

        w_pred = (pred_xyxy[:,2] - pred_xyxy[:,0]).clamp(min=1e-6)
        h_pred = (pred_xyxy[:,3] - pred_xyxy[:,1]).clamp(min=1e-6)
        w_target = (target_xyxy[:,2] - target_xyxy[:,0]).clamp(min=1e-6)
        h_target = (target_xyxy[:,3] - target_xyxy[:,1]).clamp(min=1e-6)

        # Aspect ratio penalty
        atan_pred = torch.atan(w_pred / h_pred)
        atan_target = torch.atan(w_target / h_target)
        v = (4 / (torch.pi ** 2)) * (atan_target - atan_pred) ** 2

        alpha = v / (1 - iou + v + 1e-7)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

        ciou_loss = diou_loss + alpha * v
        ciou_loss = torch.nan_to_num(ciou_loss, nan=0.0, posinf=0.0, neginf=0.0)

        return ciou_loss.mean()


# ================== GIoU Loss ==================
class GIoULossWrapper(nn.Module):
    def __init__(self, device, img_size):
        super().__init__()
        self.device = device
        self.img_size = img_size

    def forward(self, pred, target):
        """
        pred, target: (4, N)
        """
        if target.numel() == 0:
            return torch.tensor(0., device=pred.device)
        pred = pred.clone()
        target = target.clone()
        pred_xyxy = yolo_to_xyxy(pred[0,:], pred[1,:], pred[2,:], pred[3,:], self.img_size).to(self.device)
        target_xyxy = yolo_to_xyxy(target[0,:], target[1,:], target[2,:], target[3,:], self.img_size).to(self.device)

        loss = generalized_box_iou_loss(pred_xyxy, target_xyxy, reduction='none')  # (N,)
        return loss.mean()

import torch
import torch.nn as nn
from ultralytics.utils.loss import VarifocalLoss

# ================== VFL Loss ==================
class VFLWrapper(nn.Module):
    def __init__(self, device, num_classes):
        super().__init__()
        self.vfl = VarifocalLoss().to(device)
        self.device = device
        self.num_classes = num_classes

    def forward(self, pred, target):
        """
        pred: (N, num_classes) hoặc (num_classes, N)
        target: (N,) hoặc one-hot (N, num_classes) hoặc (num_classes, N)
        """
        # Chuẩn hóa pred sang (N, num_classes)
        if pred.dim() == 2 and pred.shape[1] != self.num_classes:
            pred = pred.T

        # target -> one-hot (N, num_classes)
        if target.dim() == 1 or (target.dim() == 2 and target.shape[1] == 1):
            t = torch.zeros_like(pred)
            t.scatter_(1, target.long().unsqueeze(1), 1.0)
        else:
            t = target.clone().to(pred.device, dtype=pred.dtype)
            if t.shape[0] == self.num_classes:  # (num_classes, N) -> (N, num_classes)
                t = t.T

        loss = self.vfl(pred, label=t, gt_score=t)
        return loss.mean()


class CEWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, pred, target):
        """
        pred: (N, num_classes) hoặc (num_classes, N)
        target: (N,), one-hot (N, num_classes) hoặc (num_classes, N)
        """
        # Chuẩn hóa pred sang (N, num_classes)
        if pred.dim() == 2 and pred.shape[1] != self.num_classes:
            pred = pred.T

        # Chuẩn hóa target sang (N,) với class index
        if target.dim() == 2:
            # Nếu là one-hot (N, num_classes) hoặc (num_classes, N)
            if target.shape[0] == self.num_classes:
                target = target.T
            target = target.argmax(dim=1)
        elif target.dim() == 1:
            target = target.long()
        else:
            raise ValueError(f"Unsupported target shape: {target.shape}")

        return self.ce(pred, target)



# ================== BCE Loss ==================
class BCEWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.num_classes = num_classes

    def forward(self, pred, target):
        """
        pred: (N, num_classes) hoặc (num_classes, N)
        target: one-hot (N, num_classes) hoặc (num_classes, N)
        """
        if pred.dim() == 2 and pred.shape[1] != self.num_classes:
            pred = pred.T
        if target.dim() == 2 and target.shape[0] == self.num_classes:
            target = target.T
        t = target.clone().to(pred.device, dtype=pred.dtype)
        return self.bce(pred, t)


# ================== MSE Loss ==================
class MSEWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mse = nn.MSELoss()
        self.num_classes = num_classes

    def forward(self, pred, target):
        """
        pred: (N, num_classes) hoặc (num_classes, N)
        target: (N, num_classes) hoặc (num_classes, N)
        """
        if pred.dim() == 2 and pred.shape[1] != self.num_classes:
            pred = pred.T
        if target.dim() == 2 and target.shape[0] == self.num_classes:
            target = target.T
        t = target.clone().to(pred.device, dtype=pred.dtype)
        return self.mse(pred, t)

