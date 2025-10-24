import torch
import torch.nn as nn
from torchvision.ops import box_iou, distance_box_iou_loss, generalized_box_iou_loss
from ultralytics.utils.loss import VarifocalLoss
from src.utils import yolo_to_xyxy

# ================== CIoU Loss ==================
class CIoULossWrapper(nn.Module):
    def __init__(self, device, img_size):
        super().__init__()
        self.device = device
        self.img_size = img_size

    def forward(self, pred, target):
        # Clone để tránh view inplace
        pred = pred.clone()
        target = target.clone()

        # Chuyển [x,y,w,h] -> [x1,y1,x2,y2] mà vẫn giữ graph
        pred_xyxy = yolo_to_xyxy(*pred, self.img_size).unsqueeze(0).to(self.device)
        target_xyxy = yolo_to_xyxy(*target, self.img_size).unsqueeze(0).to(self.device)

        # DIoU loss = 1 - DIoU
        diou_loss = distance_box_iou_loss(pred_xyxy, target_xyxy, reduction='mean')

        # IoU chuẩn để tính alpha*v
        iou = box_iou(pred_xyxy, target_xyxy).squeeze(0)

        # Aspect ratio term v
        w_pred = pred_xyxy[0,2] - pred_xyxy[0,0]
        h_pred = pred_xyxy[0,3] - pred_xyxy[0,1]
        w_target = target_xyxy[0,2] - target_xyxy[0,0]
        h_target = target_xyxy[0,3] - target_xyxy[0,1]

        v = (4 / (torch.pi**2)) * (torch.atan(w_target / h_target) - torch.atan(w_pred / h_pred))**2
        alpha = v / (1 - iou + v + 1e-7)

        ciou_loss = diou_loss + alpha * v
        return ciou_loss.mean() if ciou_loss.dim() > 0 else ciou_loss

# ================== GIoU Loss ==================
class GIoULossWrapper(nn.Module):
    def __init__(self, device, img_size):
        super().__init__()
        self.device = device
        self.img_size = img_size

    def forward(self, pred, target):
        pred = pred.clone()
        target = target.clone()
        pred_xyxy = yolo_to_xyxy(*pred, self.img_size).unsqueeze(0).to(self.device)
        target_xyxy = yolo_to_xyxy(*target, self.img_size).unsqueeze(0).to(self.device)
        loss = generalized_box_iou_loss(pred_xyxy, target_xyxy, reduction='mean')
        return loss.mean() if loss.dim() > 0 else loss

# ================== VFL Loss ==================
class VFLWrapper(nn.Module):
    def __init__(self, device, num_classes):
        super().__init__()
        self.vfl = VarifocalLoss().to(device)
        self.device = device
        self.num_classes = num_classes

    def forward(self, pred, target):
        pred = pred.clone()  # tránh view slice

        # Convert target scalar -> one-hot nếu cần
        if isinstance(target, int) or (isinstance(target, torch.Tensor) and target.dim() == 0):
            t = torch.zeros(self.num_classes, device=self.device, dtype=torch.float32)
            t = t.scatter(0, target.long().unsqueeze(0), 1.0)
        else:
            t = target.clone().to(self.device, dtype=torch.float32)

        # Batch dim nếu cần
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(0)

        gt_score = t.clone()
        loss = self.vfl(pred, label=t, gt_score=gt_score)
        return loss.mean() if loss.dim() > 0 else loss



import torch
import torch.nn as nn

class CEWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        # Nếu target float, convert sang long
        if target.dtype != torch.long:
            target = target.long()
        return self.ce(pred, target)

