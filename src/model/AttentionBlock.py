import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.BatchNorm2d(dim)
        # 2 conv3x3 downsample
        self.down1 = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
        # self.down2 = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
        self.act = nn.SiLU()

    def forward(self, x):  # x: (B, C, H, W)
        B, C, H, W = x.shape

        # --- Downsample ---
        x_down = self.down1(x)  # H/2, W/2
        # x_down = self.down2(x_down)  # H/4, W/4

        # --- Flatten cho attention ---
        B, C, h, w = x_down.shape
        seq = x_down.flatten(2).transpose(1, 2)  # (B, h*w, C)
        attn_out, _ = self.attn(seq, seq, seq)

        # --- Unflatten + Upsample về H,W ban đầu ---
        out = attn_out.transpose(1, 2).reshape(B, C, h, w)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        # --- Residual + norm ---
        out = out + x
        out = self.norm(out)
        out = self.act(out)
        return out
