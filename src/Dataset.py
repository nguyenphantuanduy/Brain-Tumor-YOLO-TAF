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
        self.item_list = load_list(item_path)
        self.img_size = img_size
    def __getitem__(self, index):
        image_name, label = self.item_list[index]
        image = Image.open(image_name).convert('L')  # 1 channel
        image = image.resize((self.img_size, self.img_size))
        image = torch.tensor(np.array(image)/255., dtype=torch.float32).unsqueeze(0)  # shape [1,H,W]
        label = torch.tensor(label, dtype=torch.float32)
        return image, label
    def __len__(self):
        return len(self.item_list)