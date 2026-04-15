import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random
from src.config import *
from src.preprocessing import *
import numpy as np

class Brain_Tumor_Dataset(Dataset):
    def __init__(self, item_path, img_size=640):
        self.img_size = img_size
        self.item_list = []
        interfer_list = load_list(item_path)
        for image_name, label in interfer_list:
            image_name = image_name.replace("\\", "/")
            image = Image.open(image_name).convert('L')  # 1 channel
            image = image.resize((self.img_size, self.img_size))
            image = torch.from_numpy(np.array(image, dtype=np.float32)/255.).unsqueeze(0)  # [1,H,W]
            label = torch.tensor(label, dtype=torch.float32)  # shape [num_bbox,5] nếu có nhiều bbox
            self.item_list.append((image_name, image, label))

    def __getitem__(self, index):
        return (self.item_list[index][0], self.item_list[index][1]), self.item_list[index][2]

    def __len__(self):
        return len(self.item_list)
