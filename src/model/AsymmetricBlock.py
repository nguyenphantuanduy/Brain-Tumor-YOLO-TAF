import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Trọng số học được
        self.w1 = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.w2 = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.w3 = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
        # Conv cuối cùng + BatchNorm
        self.conv = nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def neg_silu(self, x):
        return F.silu(-x)

    def forward(self, x):
        # --- Tạo các scale ---
        x1 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool2d(x1, kernel_size=3, stride=1, padding=1)
        x3 = F.avg_pool2d(x2, kernel_size=3, stride=1, padding=1)
        
        # --- Pixel-wise tương tác ---
        y1 = x * x1
        y2 = x * x2
        y3 = x * x3
        
        # --- Activation nghịch ---
        z1 = self.neg_silu(y1)
        z2 = self.neg_silu(y2)
        z3 = self.neg_silu(y3)

        # Chuẩn hóa để tránh bùng giá trị
        z1 = torch.tanh(z1)
        z2 = torch.tanh(z2)
        z3 = torch.tanh(z3)
        
        # --- Cộng với x và trọng số học được ---
        t1 = x + self.w1 * z1
        t2 = x + self.w2 * z2
        t3 = x + self.w3 * z3
        
        # --- Concat và conv để tạo output cùng C, H, W ---
        out = torch.cat([t1, t2, t3], dim=1)  # concat kênh
        out = self.conv(out)
        out = self.bn(out)
        out = self.act(out)
        return out
