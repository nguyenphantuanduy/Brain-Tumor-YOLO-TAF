import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics.utils.loss import VarifocalLoss
from torchmetrics.detection.iou import CompleteIoULoss  # Giả sử dùng CIoU, GIoU

