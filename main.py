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
from src.model.BrainTumorv2 import BrainTumorv2
from src.model.BrainTumorv3 import BrainTumorv3
from src.model.BrainTumorv4 import BrainTumorv4
from src.model.BrainTumorv5 import BrainTumorv5
from src.model.BrainTumorv6 import BrainTumorv6
from src.model.MyBrainTumorWrapperv2 import MyBrainTumorWrapperv2
from src.model.MyBrainTumorWrapperv3 import MyBrainTumorWrapperv3
from src.model.MyBrainTumorWrapperv4 import MyBrainTumorWrapperv4
from src.model.BrainTumorv7 import BrainTumorv7
from src.model.Yoloonly import run
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

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # SEED = 42
    # set_seed(SEED)

    # # --- Tạo generator cho DataLoader ---
    # g = torch.Generator()
    # g.manual_seed(SEED)

    # train_dataset = Brain_Tumor_Dataset("data/total_train_list_aug.pkl")
    # train_dataloader = DataLoader(
    #     dataset=train_dataset, batch_size=8,
    #     shuffle=True, num_workers=0, pin_memory=False,
    #     collate_fn=yolo_collate_fn,
    #     generator=g  # shuffle deterministic
    # )

    # val_dataset = Brain_Tumor_Dataset("data/total_val_list.pkl")
    # val_dataloader = DataLoader(
    #     dataset=val_dataset, batch_size=8,
    #     shuffle=False, num_workers=0, pin_memory=False,
    #     collate_fn=yolo_collate_fn
    # )

    # model = BrainTumorv2()
    # myWrapper = MyBrainTumorWrapperv4(
    #     model, CKPT_PATH="experiments/BrainTumorv2_legendary.pth.tar"
    # )
    # myWrapper.fit(train_dataloader, val_dataloader, 50, 10, "Sustain")



    # test_dataset = Brain_Tumor_Dataset("data/total_test_list.pkl")
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=8,
    #                 shuffle=False, num_workers=0, pin_memory=False, collate_fn=yolo_collate_fn)
    # # # import matplotlib.pyplot as plt
    # # # image_path = "data/raw/Val/Glioma/images/gg (9).jpg"
    # model = BrainTumorv2()
    # myWrapper = MyBrainTumorWrapperv4(model, CKPT_PATH="experiments/BrainTumorv2_legendary.pth.tar")
    # myWrapper.evaluate(test_dataloader, True)
    # img = myWrapper.img_predict(image_path)
    model = BrainTumorv2()
    myWrapper = MyBrainTumorWrapperv4(model, CKPT_PATH="experiments/BrainTumorv2_legendary.pth.tar")
    print(model)
    # import os
    # import csv

    # LOG_PATH = "training_log.csv"

    # # tạo file + header nếu chưa tồn tại
    # if not os.path.exists(LOG_PATH):
    #     with open(LOG_PATH, mode="w", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow([
    #             "epoch",
    #             "train_loss",
    #             "val_loss",
    #             "precision",
    #             "recall",
    #             "mAP_50",
    #             "mAP_50_95"
    #         ])


    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # plt.figure(figsize=(6,6))
    # plt.imshow(img_rgb)
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()
    # myWrapper.compile("Sustain")
    # myWrapper.evaluate(test_dataloader, True)

    # model = BrainTumorv4()
    # myWrapper = MyBrainTumorWrapperv4(model, CKPT_PATH="experiments/BrainTumorv4_legendary.pth.tar")
    # myWrapper.img_predict("data/raw/Val/Glioma/images/gg (9).jpg")

    # train_list02 = createListFromPath02("data/train")
    # val_list02 = createListFromPath02("data/valid")
    # test_list02 = createListFromPath02("data/test")

    # train_list01 = load_list("data/raw/train_list.pkl")
    # val_list01 = load_list("data/val_list.pkl")
    # test_list01 = load_list("data/test_list.pkl")

    # save_list(train_list02 + train_list01, "data/total_train_list.pkl")
    # save_list(val_list02 + val_list01, "data/total_val_list.pkl")
    # save_list(test_list02 + test_list01, "data/total_test_list.pkl")

    # train_list_aug = load_list("data/train_list.pkl")
    # save_list(train_list02 + train_list_aug, "data/total_train_list_aug.pkl")

    # save_list_txt("data/total_train_list_aug.pkl", "data/total_train_list_aug.txt")
    # save_list_txt("data/total_val_list.pkl", "data/total_val_list.txt")
    # save_list_txt("data/total_test_list.pkl", "data/total_test_list.txt")

    # import yaml

    # # Cấu hình YAML cho YOLOv8
    # yaml_config = {
    #     "train": "data/total_train_list_aug.txt",
    #     "val": "data/total_val_list.txt",
    #     "test": "data/total_test_list.txt",  # optional
    #     "nc": 4,
    #     "names": ["Glioma", "Meningioma", "NoTumor", "Pituitary"]
    # }

    # yaml_path = "data/brain_tumor.yaml"

    # with open(yaml_path, "w") as f:
    #     yaml.dump(yaml_config, f, sort_keys=False)

    # print(f"✅ Đã tạo file YAML: {yaml_path}")
    # run()
    # train_list = load_list("data/total_train_list.pkl")
    # val_list   = load_list("data/total_val_list.pkl")
    # test_list  = load_list("data/total_test_list.pkl")
    # train_list_aug = load_list("data/total_train_list_aug.pkl")

    # print("Dataset statistics")
    # print("-" * 30)
    # print(f"{'Train':15}: {len(train_list):6}")
    # print(f"{'Train (Aug)':15}: {len(train_list_aug):6}")
    # print(f"{'Validation':15}: {len(val_list):6}")
    # print(f"{'Test':15}: {len(test_list):6}")


