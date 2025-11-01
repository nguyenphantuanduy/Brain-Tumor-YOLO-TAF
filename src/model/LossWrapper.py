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

        # if torch.isnan(pred_xyxy).any():
        #     print("NaN in pred_xyxy!")
        # if torch.isnan(target_xyxy).any():
        #     print("NaN in target_xyxy!")

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
        # if torch.isnan(ciou_loss).any() or torch.isinf(ciou_loss).any():
        #     print("NaN/Inf in CIoU loss!")
        #     print("diou_loss:", diou_loss)
        #     print("alpha:", alpha)
        #     print("v:", v)

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
    
import torch
import torch.nn as nn
import torchvision.ops as ops

# ================== Focal BCE Wrapper (torchvision) ==================
class FocalBCEWrapperTV(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        num_classes: số class output
        alpha: cân bằng pos/neg
        gamma: focusing parameter
        reduction: 'mean', 'sum' hoặc 'none'
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred: (N, num_classes) hoặc (num_classes, N), logits
        target: one-hot (N, num_classes) hoặc (num_classes, N)
        """
        if pred.dim() == 2 and pred.shape[1] != self.num_classes:
            pred = pred.T
        if target.dim() == 2 and target.shape[0] == self.num_classes:
            target = target.T

        pred = pred.to(target.device, dtype=target.dtype)
        t = target.clone().to(pred.device, dtype=pred.dtype)

        # torchvision.ops.sigmoid_focal_loss dùng logits trực tiếp
        loss = ops.sigmoid_focal_loss(
            inputs=pred,
            targets=t,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction
        )
        return loss



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
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class OneHotCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, target_onehot):
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(target_onehot * log_probs).sum(dim=1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class NegClsLoss(nn.Module):
    def __init__(self, num_classes, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.criterion = OneHotCrossEntropyLoss(reduction=reduction)

    def forward(self, pred):
        if pred.dim() == 2 and pred.shape[1] != self.num_classes:
            pred = pred.T
        batch_size = pred.shape[0]
        # tạo target uniform cùng device với pred
        target_uniform = torch.full(
            (batch_size, self.num_classes),
            1.0 / self.num_classes,
            device=pred.device,
            dtype=pred.dtype
        )
        loss = self.criterion(pred, target_uniform)
        return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class NegClsFocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        num_classes: số lớp
        alpha: hệ số cân bằng (giảm tác động của easy samples)
        gamma: độ nhạy của focal (mặc định 2.0)
        reduction: 'mean' hoặc 'sum'
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred):
        """
        pred: logits shape (N, num_classes) hoặc (num_classes, N)
        => Loss trung tính, ép model không tự tin ở vùng không có GT.
        """
        if pred.dim() == 2 and pred.shape[1] != self.num_classes:
            pred = pred.T

        batch_size = pred.shape[0]

        # target trung tính (đều cho mọi class)
        target_uniform = torch.full(
            (batch_size, self.num_classes),
            1.0 / self.num_classes,
            device=pred.device,
            dtype=pred.dtype
        )

        # softmax để lấy xác suất từng class
        probs = F.softmax(pred, dim=1).clamp(min=1e-8, max=1.0)  # tránh log(0)

        # focal modulation: (1 - p_t)^γ
        focal_weight = torch.pow(1.0 - probs, self.gamma)

        # focal cross-entropy với target uniform
        loss = -self.alpha * focal_weight * target_uniform * torch.log(probs)

        loss = loss.sum(dim=1)  # tổng qua các class

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss




