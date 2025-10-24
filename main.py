import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.Dataset import Brain_Tumor_Dataset
from src.model.BrainTumorv1 import BrainTumorv1
from src.model.BrainTumorWrapper import BrainTumorWrapper
from src.utils import *
from src.model.MyBrainTumorWrapper import MyBrainTumorWrapper

def cls_loss_fn(pred, target):
    target = target.long()  # ép sang Long
    return nn.CrossEntropyLoss()(pred.unsqueeze(0), target.unsqueeze(0))

def main():
    # --- Dataset + DataLoader ---
    main_dataset = Brain_Tumor_Dataset("data/train_list.pkl")
    main_loader = DataLoader(dataset=main_dataset, batch_size=4,
                            shuffle=True, num_workers=0, pin_memory=False, collate_fn=yolo_collate_fn)
    
    # --- Model ---
    model = BrainTumorv1()
    
    # --- Losses ---
    reg_loss = nn.MSELoss()
    objness_loss = nn.BCEWithLogitsLoss()
    
    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # --- Wrapper ---
    wrapper = BrainTumorWrapper(model, optimizer, reg_loss, cls_loss_fn, objness_loss, device='cpu')
    
    # --- Train 1 epoch ---
    wrapper.fit(main_loader)

def test():
    model = BrainTumorv1()
    
    # --- Losses ---
    reg_loss = nn.MSELoss()
    objness_loss = nn.BCEWithLogitsLoss()
    
    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # --- Wrapper ---
    wrapper = BrainTumorWrapper(model, optimizer, reg_loss, cls_loss_fn, objness_loss, device='cpu')
    wrapper.img_predict("data/raw/Val/Glioma/images/gg (9).jpg")

def test02():
    # --- Dataset + DataLoader ---
    model = BrainTumorv1()
    main_dataset = Brain_Tumor_Dataset("data/train_list.pkl")
    main_loader = DataLoader(dataset=main_dataset, batch_size=4,
                            shuffle=True, num_workers=0, pin_memory=False, collate_fn=yolo_collate_fn)
    myWrapper = MyBrainTumorWrapper(model)
    myWrapper.fit(main_loader, epochs = 1, patience = 3, mode = "Sustain")
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # cần cho Windows
    test02()
