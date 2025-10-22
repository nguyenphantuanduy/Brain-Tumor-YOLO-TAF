# ---- Giả sử bạn đã import Dataset ----

import torch

from torch.utils.data import DataLoader
from src.Dataset import Brain_Tumor_Dataset

dataset = Brain_Tumor_Dataset(item_path="data/raw/val_list.pkl", img_size=640)
dataloader = DataLoader(
    dataset, 
    batch_size=2, 
    shuffle=False, 
    collate_fn=lambda x: tuple(zip(*x))  # giữ labels list of tensors
)

# ---- Lấy batch đầu tiên ----
images, labels = next(iter(dataloader))

print("Images shape:", torch.stack(images).shape)  # [B, C, H, W]
print("Number of labels in batch:", len(labels))

print(labels)





