import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.Dataset import Brain_Tumor_Dataset
from src.model.BrainTumorv1 import BrainTumorv1
from src.model.BrainTumorWrapper import BrainTumorWrapper
from src.utils import *
from src.model.MyBrainTumorWrapper import MyBrainTumorWrapper
from src.preprocessing import *
from src.model.LossWrapper import *

# def cls_loss_fn(pred, target):
#     target = target.long()  # ép sang Long
#     return nn.CrossEntropyLoss()(pred.unsqueeze(0), target.unsqueeze(0))

# def main():
#     # --- Dataset + DataLoader ---
#     main_dataset = Brain_Tumor_Dataset("data/train_list.pkl")
#     main_loader = DataLoader(dataset=main_dataset, batch_size=4,
#                             shuffle=True, num_workers=0, pin_memory=False, collate_fn=yolo_collate_fn)
    
#     # --- Model ---
#     model = BrainTumorv1()
    
#     # --- Losses ---
#     reg_loss = nn.MSELoss()
#     objness_loss = nn.BCEWithLogitsLoss()
    
#     # --- Optimizer ---
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
#     # --- Wrapper ---
#     wrapper = BrainTumorWrapper(model, optimizer, reg_loss, cls_loss_fn, objness_loss, device='cpu')
    
#     # --- Train 1 epoch ---
#     wrapper.fit(main_loader)

# def test():
#     model = BrainTumorv1()
    
#     # --- Losses ---
#     reg_loss = nn.MSELoss()
#     objness_loss = nn.BCEWithLogitsLoss()
    
#     # --- Optimizer ---
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
#     # --- Wrapper ---
#     wrapper = BrainTumorWrapper(model, optimizer, reg_loss, cls_loss_fn, objness_loss, device='cpu')
#     wrapper.img_predict("data/raw/Val/Glioma/images/gg (9).jpg")

# def test02():
#     # --- Dataset + DataLoader ---
#     model = BrainTumorv1()
#     main_dataset = Brain_Tumor_Dataset("data/train_list.pkl")
#     main_loader = DataLoader(dataset=main_dataset, batch_size=4,
#                             shuffle=True, num_workers=0, pin_memory=False, collate_fn=yolo_collate_fn)
#     myWrapper = MyBrainTumorWrapper(model)
#     myWrapper.fit(main_loader, epochs = 1, patience = 3, mode = "Warm-up")

# def test03():
#     val_list = []
#     test_list = []
#     total_list = load_list("data/raw/val_list.pkl")
#     spilt_list = split_by_class(total_list)
#     for key, value in spilt_list.items():
#         mid = len(value) // 2
#         val_list = val_list + value[:mid]
#         test_list = test_list + value[mid:]
#     save_list(val_list, "data/val_list.pkl")
#     save_list(test_list, "data/test_list.pkl")

if __name__ == "__main__":
    # train_dataset = Brain_Tumor_Dataset("data/train_list.pkl")
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=8,
    #                 shuffle=True, num_workers=10, pin_memory=True, collate_fn=yolo_collate_fn)
    # val_dataset = Brain_Tumor_Dataset("data/val_list.pkl")
    # val_dataloader = DataLoader(dataset=val_dataset, batch_size=8,
    #                 shuffle=False, num_workers=10, pin_memory=True, collate_fn=yolo_collate_fn)
    # model = BrainTumorv1()
    # myWrapper = MyBrainTumorWrapper(model, CKPT_PATH="experiments/BrainTumorv1.pth.tar")
    # myWrapper.fit(train_dataloader, val_dataloader, 70, 5, "Sustain")

    # val_dataset = Brain_Tumor_Dataset("data/test_list.pkl")
    # val_dataloader = DataLoader(dataset=val_dataset, batch_size=8,
    #                 shuffle=False, num_workers=10, pin_memory=True, collate_fn=yolo_collate_fn)
    # model = BrainTumorv1()
    # myWrapper = MyBrainTumorWrapper(model, CKPT_PATH="experiments/BrainTumorv1.pth.tar")
    # myWrapper.compile("Sustain")
    # myWrapper.evaluate(val_dataloader, True)
    
    Llist = load_list("data/test_list.pkl")
    n = max([len(x) for _, x in Llist])
    print(n)