import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.config import *
from torchvision.ops import nms
from torchvision.ops import batched_nms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from PIL import Image
import numpy as np
from src.utils import visualize_mri_prediction
import os
import cv2


class BrainTumorWrapper:
    def __init__(self, model, optimizer=None, reg_loss = None, cls_loss = None, objness_loss = None, device=None, CKPT_PATH=None, num_classes = 4, strides = strides, img_size = img_size, scheduler = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.assign_map = {}
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.reg_loss = reg_loss
        self.cls_loss = cls_loss
        self.objness_loss = objness_loss
        self.history = {"train_loss": [], "val_loss": []}
        self.strides = strides
        self.img_size = img_size
        self.edge = [int(img_size/x) for x in self.strides]
        # Checkpoint (tuỳ chọn)
        self.CKPT_PATH = CKPT_PATH
        self.start_epoch = 0
        self.num_classes = num_classes

        #assign gt
        self.map_assign = {}
        s0 = strides[0]
        s1 = strides[1]
        s2 = strides[2]

        self.boundaries = torch.tensor([
        s0 + 1/3 * (s1 - s0),
        s0 + 2/3 * (s1 - s0),
        s1 + 1/3 * (s2 - s1),
        s1 + 2/3 * (s2 - s1)
        ], device=self.device)

        self.params = {0: [(0, s0, 0.3)],
                    1: [(0, s0, 0.3), (1, s1, 0.2)],
                    2: [(1, s1, 0.2)],
                    3: [(1, s1, 0.2), (2, s2, 0.1)],
                    4: [(2, s2, 0.1)]}


        # Nếu có checkpoint → load lại
        if CKPT_PATH and os.path.isfile(CKPT_PATH):
            self._load_checkpoint(CKPT_PATH)
            print(f"Resume training from checkpoint: {CKPT_PATH}")
        else:
            if CKPT_PATH:
                print(f"Checkpoint '{CKPT_PATH}' không tồn tại, sẽ train từ đầu.")
            else:
                print("Không có checkpoint, sẽ train từ đầu.")
    

    def _load_checkpoint(self, path):
        """Load checkpoint: model, optimizer, history"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1

        if "history" in checkpoint:
            self.history = checkpoint["history"]


    def compile(self, optimizer=None, reg_loss=None, cls_loss=None, objness_loss=None, scheduler = None):
        if optimizer:
            self.optimizer = optimizer
        if scheduler:
            self.scheduler = scheduler
        if reg_loss:
            # Nếu là class loss (có .to), đưa lên device
            if hasattr(reg_loss, "to"):
                self.reg_loss = reg_loss.to(self.device)
            else:
                self.reg_loss = reg_loss  # hàm loss thì giữ nguyên
        if cls_loss:
            if hasattr(cls_loss, "to"):
                self.cls_loss = cls_loss.to(self.device)
            else:
                self.cls_loss = cls_loss
        if objness_loss:
            if hasattr(objness_loss, "to"):
                self.objness_loss = objness_loss.to(self.device)
            else:
                self.objness_loss = objness_loss


    def cell_assign(self, x, y, w, h, stride, img_size=img_size, scale_ratio=0.3):
        device = x.device if isinstance(x, torch.Tensor) else 'cuda'

        # 1. Chuyển bbox sang pixels
        cx = x * img_size
        cy = y * img_size
        bw = w * img_size * scale_ratio
        bh = h * img_size * scale_ratio

        # 2. Xác định vùng bounding box nhỏ
        x0 = cx - bw / 2
        y0 = cy - bh / 2
        x1 = cx + bw / 2
        y1 = cy + bh / 2

        # 3. Chuyển sang indices feature map
        fmap_h = fmap_w = img_size // stride
        i0 = torch.clamp(torch.floor(y0 / stride).long(), 0, fmap_h - 1)
        j0 = torch.clamp(torch.floor(x0 / stride).long(), 0, fmap_w - 1)
        i1 = torch.clamp(torch.ceil(y1 / stride).long(), 0, fmap_h - 1)
        j1 = torch.clamp(torch.ceil(x1 / stride).long(), 0, fmap_w - 1)

        # 4. Tạo meshgrid tất cả cell indices trong bbox
        if i1 < i0 or j1 < j0:  # fallback nếu bbox quá nhỏ
            ci = torch.clamp((cy // stride).long(), 0, fmap_h - 1)
            cj = torch.clamp((cx // stride).long(), 0, fmap_w - 1)
            return torch.tensor([[ci, cj, 1.0]], device=device)

        ys = torch.arange(i0, i1 + 1, device=device)
        xs = torch.arange(j0, j1 + 1, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

        # 5. Tạo tensor GT bbox
        gt_tensor = self.yolo_to_xyxy(x, y, w, h, img_size).to(device).unsqueeze(0)  # [1,4]

        # 6. Tạo tensor cells dạng xyxy
        cell_cx = (grid_x + 0.5) * stride / img_size
        cell_cy = (grid_y + 0.5) * stride / img_size
        cell_w = stride / img_size
        cell_h = stride / img_size
        cell_tensor = self.yolo_to_xyxy(cell_cx.flatten(), cell_cy.flatten(), 
                                    torch.full_like(cell_cx.flatten(), cell_w),
                                    torch.full_like(cell_cy.flatten(), cell_h),
                                    img_size)  # [N,4]

        # 7. Tính IoU giữa GT và tất cả cell
        ious = box_iou(gt_tensor, cell_tensor)[0]  # [N]

        # 8. Ghép chỉ số i,j với iou
        assigned_cells = torch.stack([
            grid_y.flatten().to(dtype=torch.long),  # chuyển sang long
            grid_x.flatten().to(dtype=torch.long),  # chuyển sang long
            ious
        ], dim=1)  # [N,3]

        return assigned_cells


    # def cell_assign(self, x, y, w, h, stride, img_size=img_size, scale_ratio=0.3):
    #     # 1. Chuyển bbox sang pixels
    #     cx = x * img_size
    #     cy = y * img_size
    #     bw = w * img_size * scale_ratio
    #     bh = h * img_size * scale_ratio
    #     # 2. Xác định vùng bounding box nhỏ
    #     x0 = cx - bw / 2
    #     y0 = cy - bh / 2
    #     x1 = cx + bw / 2
    #     y1 = cy + bh / 2
    #     # 3. Chuyển vùng sang indices feature map
    #     fmap_w = fmap_h = img_size // stride
    #     i0 = max(int(y0 // stride), 0)
    #     j0 = max(int(x0 // stride), 0)
    #     i1 = min(int(y1 // stride), fmap_h - 1)
    #     j1 = min(int(x1 // stride), fmap_w - 1)
    #     # 4. Tạo tensor GT
    #     gt_tensor = self.yolo_to_xyxy(x, y, w, h, img_size).unsqueeze(0)
    #     # 5. Gom toàn bộ cell boxes trong vùng
    #     cells = []
    #     coords = []
    #     for i in range(i0, i1 + 1):
    #         for j in range(j0, j1 + 1):
    #             cell_cx = (j + 0.5) * stride / img_size
    #             cell_cy = (i + 0.5) * stride / img_size
    #             cell_w = stride / img_size
    #             cell_h = stride / img_size
    #             cell_xyxy = self.yolo_to_xyxy(cell_cx, cell_cy, cell_w, cell_h, img_size)
    #             cells.append(cell_xyxy)
    #             coords.append((i, j))
    #     if not cells:  # Không có cell nào được chọn
    #         ci = min(max(int(cy // stride), 0), fmap_h - 1)
    #         cj = min(max(int(cx // stride), 0), fmap_w - 1)
    #         return [(ci, cj, 1.0)]
    #     # 6. Chuyển sang tensor và tính IoU toàn bộ
    #     cell_tensor = torch.stack(cells)
    #     ious = box_iou(gt_tensor, cell_tensor)[0]  # [N]
    #     # 7. Ghép lại (i, j, iou)
    #     assigned_cells = [(i, j, iou.item()) for (i, j), iou in zip(coords, ious)]
    #     return assigned_cells

    def gt_assign(self, x, y, w, h):
        size = max(w * img_size, h * img_size)
        idx = torch.bucketize(size, self.boundaries)
        params = self.params[int(idx)]
        assign = [(param[0], self.cell_assign(x, y, w, h, param[1], img_size, param[2])) for param  in params]
        return assign

    # def gt_assign(self, x, y, w, h, img_size = img_size, strides = strides):
    #     size = max(w * img_size, h * img_size)
    #     s0 = strides[0]
    #     s1 = strides[1]
    #     s2 = strides[2]
    #     assign = []
    #     if size <= s0:
    #         assign.append((0, self.cell_assign(x, y, w, h, s0, img_size, 0.3)))
    #     elif size <= s1:
    #         part = (size - s0)/(s1 - s0)
    #         if part < 1/3:
    #             assign.append((0, self.cell_assign(x, y, w, h, s0, img_size, 0.3)))
    #         elif part < 2/3:
    #             assign.append((0, self.cell_assign(x, y, w, h, s0, img_size, 0.3)))
    #             assign.append((1, self.cell_assign(x, y, w, h, s1, img_size, 0.2)))
    #         else:
    #             assign.append((1, self.cell_assign(x, y, w, h, s1, img_size, 0.2)))
    #     elif size <= s2:
    #         part = (size - s1)/(s2 - s1)
    #         if part < 1/3:
    #             assign.append((1, self.cell_assign(x, y, w, h, s1, img_size, 0.2)))
    #         elif part < 2/3:
    #             assign.append((1, self.cell_assign(x, y, w, h, s1, img_size, 0.2)))
    #             assign.append((2, self.cell_assign(x, y, w, h, s2, img_size, 0.1)))
    #         else:
    #             assign.append((2, self.cell_assign(x, y, w, h, s2, img_size, 0.1)))
    #     else:
    #         assign.append((2, self.cell_assign(x, y, w, h, s2, img_size, 0.1)))
        
    #     return assign


    def head_to_yolo(self, output, stride, img_size):
        """
        Decode YOLO anchor-free head output of form [l, t, r, b, ...]
        into normalized YOLO-format [x_center, y_center, w, h, ...]
        """
        c, H, W = output.shape
        device = output.device

        # Giải nén l, t, r, b
        l = output[0]   # [H, W]
        t = output[1]
        r = output[2]
        b = output[3]
        rest = output[4:]

        # Tạo lưới pixel center (anchor point)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        grid_x = grid_x.float()
        grid_y = grid_y.float()

        # Decode ra toạ độ góc (pixel)
        x1 = (grid_x - l) * stride
        y1 = (grid_y - t) * stride
        x2 = (grid_x + r) * stride
        y2 = (grid_y + b) * stride

        # Tính toạ độ center và kích thước
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        # Chuẩn hoá về [0, 1]
        x_center /= img_size
        y_center /= img_size
        width /= img_size
        height /= img_size

        # Clamp để tránh lỗi ngoài biên
        x_center = x_center.clamp(0.0, 1.0)
        y_center = y_center.clamp(0.0, 1.0)
        width = width.clamp(0.0, 1.0)
        height = height.clamp(0.0, 1.0)

        # Gộp lại tensor chuẩn YOLO
        yolo_output = torch.zeros_like(output)
        yolo_output[0] = x_center
        yolo_output[1] = y_center
        yolo_output[2] = width
        yolo_output[3] = height
        yolo_output[4:] = rest

        return yolo_output



    def fit(self, train_loader, val_loader=None, epochs=10, patience=3):
        # pos_tensor = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        # neg_tensor = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        patience_counter = 0
        best_val_loss = float('inf')
        self.model.train()
        # torch.autograd.set_detect_anomaly(True)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for paths, imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                imgs = imgs.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                #outputs = torch.stack(outputs) if isinstance(outputs, list) else outputs
                #outputs = outputs.permute(1, 0, 2, 3, 4)
                for idx, target in zip(range(len(targets)), targets):
                    if target.numel() == 0:
                        continue
                    elif target.dim() == 1:
                        target = target.unsqueeze(0)  # thêm chiều đầu tiên → [1, N]
                    path = paths[idx]
                    target = target.to(self.device)
                    output = []
                    for (num_layer, layer) in zip(range(len(outputs)), outputs):
                        output.append(self.head_to_yolo(layer[idx], self.strides[num_layer], self.img_size))

                    sample_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                    output_tensor = torch.cat([fmap.flatten(1) for fmap in output], dim=1)
                    # if torch.isnan(output_tensor).any():
                    #     print("⚠️ output_tensor has NaN BEFORE loss!")
                    #     torch.save(output_tensor, "nan_output.pt")
                    #     break
                    mask = torch.zeros_like(output_tensor, device=self.device)
                    if path in self.assign_map: mask = self.assign_map[path]
                    else:
                        assigned_cells = []
                        best_cells = {}
                        for bbox in target:
                            cls, x, y, w, h = bbox                        
                            reg_tensor = torch.tensor([x, y, w, h], device=self.device)
                            assign_list = self.gt_assign(x, y, w, h)
                            for fmap_idx, cells in assign_list:
                                for i, j, iou in cells:
                                    assigned_cells.append((fmap_idx, i, j, iou, cls, reg_tensor))
                        iou_thresh = 0.3
                        #filtered = [c for c in assigned_cells if c[3] > iou_thresh]
                        for fmap_idx, i, j, iou, cls, reg_tensor in assigned_cells:
                            i = int(i)
                            j = int(j)
                            key = (fmap_idx, i, j)
                            if key not in best_cells or iou > best_cells[key][0]:
                                best_cells[key] = (iou, reg_tensor, 1.0, cls)  # 1.0 = objness
                        for (fmap_idx, i, j), (iou, reg_tensor, obj, cls) in best_cells.items():
                            # flatten index
                            start_idx = sum([e**2 for e in self.edge[:fmap_idx]])  # tổng số ô trước fmap_idx
                            idx = start_idx + i * self.edge[fmap_idx] + j  # int index
                            idx = int(idx)
                            one_hot_cls = torch.zeros(self.num_classes, device=self.device)
                            one_hot_cls[int(cls)] = 1.0
                            mask[:, idx] = torch.cat([reg_tensor, torch.tensor([obj], device=self.device), one_hot_cls])
                        self.assign_map[path] = mask

                    obj_channel_index = 4
                    has_gt = mask[obj_channel_index, :] > 0
                    no_gt  = ~has_gt
                    output_has_gt = output_tensor[:, has_gt]  # (C, num_gt)
                    mask_has_gt   = mask[:, has_gt]           # (C, num_gt)
                    output_no_gt  = output_tensor[:, no_gt]   # (C, num_no_gt)
                    mask_no_gt    = mask[:, no_gt]            # chỉ cần objness
                    # Phần có gt
                    reg_loss = self.reg_loss(output_has_gt[:4, :], mask_has_gt[:4, :])
                    obj_loss = self.objness_loss(output_has_gt[4:5, :], mask_has_gt[4:5, :])
                    cls_loss = self.cls_loss(output_has_gt[5:, :], mask_has_gt[5:, :])
                    loss_has_gt = reg_loss + obj_loss + cls_loss
                    # print("any nan in output_tensor:", torch.isnan(output_tensor).any())
                    # print(output_has_gt)
                    # print("kkkkkkkkk")
                    # Phần không có gt (objness = 0)
                    obj_loss_no_gt = self.objness_loss(output_no_gt[4:5, :], torch.zeros_like(output_no_gt[4:5, :]))
                    sample_loss += loss_has_gt + 1/3 * obj_loss_no_gt
                    # print(output_has_gt)
                    # print("hhhhhhhhhhhh")

                    # # Compute loss
                    # for fmap_idx, i, j, iou, cls, reg_tensor in best_cells.values():
                    #     pred = output[fmap_idx][:, i, j]
                    #     pred_reg = pred[:4]
                    #     pred_obj = pred[4:5]
                    #     pred_cls = pred[5:]
                    #     sample_loss = sample_loss + (
                    #             self.cls_loss(pred_cls, cls)
                    #             + self.reg_loss(pred_reg, reg_tensor)
                    #             + self.objness_loss(pred_obj, torch.ones_like(pred_obj))
                    #             )
                    # for fmap_idx, fmap in enumerate(output):
                    #     H, W = fmap.shape[1], fmap.shape[2]
                    #     for i in range(H):
                    #         for j in range(W):
                    #             if (fmap_idx, i, j) not in best_cells:
                    #                 pred_obj = fmap[4, i, j]  # objness channel
                    #                 sample_loss = sample_loss + 1/3 * self.objness_loss(pred_obj, torch.zeros_like(pred_obj))
                    # print(sample_loss)
                    batch_loss = batch_loss + sample_loss        
                #print("batch_loss.requires_grad =", batch_loss.requires_grad)
                batch_loss = batch_loss / len(targets)
                # print(batch_loss)
                batch_loss.backward()
                self.optimizer.step()
                epoch_loss += batch_loss.item()

            if self.scheduler: self.scheduler.step()
            avg_epoch_loss = epoch_loss / len(train_loader)
            self.history["train_loss"].append(avg_epoch_loss)

            print(f"Epoch {epoch+1}: train_loss={avg_epoch_loss:.4f}")

            # Evaluate per epoch
            if val_loader:
                eval_result = self.evaluate(val_loader, verbose=False)
                val_loss = eval_result["val_loss"]
                map_result = eval_result["mAP"]
                self.history["val_loss"].append(val_loss)
                print(f"   val_loss={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    print("=> Validation improved, saving model...")
                    if self.CKPT_PATH:
                        self.save_model(self.CKPT_PATH, epoch)
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience counter: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print("Early stopping triggered!")
                        break

    # def yolo_to_xyxy(self, x, y, w, h, img_size):
    #     if isinstance(img_size, int):
    #         W = H = img_size
    #     else:
    #         H, W = img_size
    #     # Convert
    #     x1 = (x - w / 2) * W
    #     y1 = (y - h / 2) * H
    #     x2 = (x + w / 2) * W
    #     y2 = (y + h / 2) * H
    #     # Stack lại
    #     converted = torch.tensor([x1, y1, x2, y2], device=self.device, dtype=torch.float32)
    #     # Clamp tránh ra ngoài ảnh
    #     converted[0] = converted[0].clamp(0, W - 1)
    #     converted[1] = converted[1].clamp(0, H - 1)
    #     converted[2] = converted[2].clamp(0, W - 1)
    #     converted[3] = converted[3].clamp(0, H - 1)
    #     return converted 
    def yolo_to_xyxy(self, x, y, w, h, img_size):
        """
        Chuyển từ YOLO format (x_center, y_center, w, h) sang XYXY (x1, y1, x2, y2)
        Hỗ trợ cả:
        - x, y, w, h là scalar
        - [N] tensor
        - [B, N] tensor
        """
        if isinstance(img_size, int):
            H = W = img_size
        else:
            H, W = img_size

        # Đảm bảo tất cả đều là tensor (tránh trường hợp float/int)
        device = x.device if isinstance(x, torch.Tensor) else self.device
        x = torch.as_tensor(x, device=device, dtype=torch.float32)
        y = torch.as_tensor(y, device=device, dtype=torch.float32)
        w = torch.as_tensor(w, device=device, dtype=torch.float32)
        h = torch.as_tensor(h, device=device, dtype=torch.float32)
        if x.dim() == 2 and x.shape[0] == 1:
            x = x.T
            y = y.T
            w = w.T
            h = h.T
        # Tính toán bằng broadcast (chạy được cả batch)
        x1 = (x - w / 2) * W
        y1 = (y - h / 2) * H
        x2 = (x + w / 2) * W
        y2 = (y + h / 2) * H

        # Stack theo chiều cuối cùng (tự động ra [4], [N,4], [B,N,4])
        converted = torch.stack([x1, y1, x2, y2], dim=-1)

        # Clamp để không vượt quá ảnh
        converted[..., 0] = converted[..., 0].clamp(0, W - 1)
        converted[..., 1] = converted[..., 1].clamp(0, H - 1)
        converted[..., 2] = converted[..., 2].clamp(0, W - 1)
        converted[..., 3] = converted[..., 3].clamp(0, H - 1)

        return converted


    def evaluate(self, val_loader, verbose=True):
        self.model.eval()
        total_loss = 0.0
        metric_map = MeanAveragePrecision().to(self.device)

        with torch.no_grad():
            for paths, imgs, targets in tqdm(val_loader, desc="Evaluating", disable=not verbose):
                batch_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                preds_for_metric = []
                gts_for_metric = []
                imgs = imgs.to(self.device)
                outputs = self.model(imgs)
                for idx, target in zip(range(len(targets)), targets):
                    if target.numel() == 0:  # kiểm tra tensor rỗng
                        continue
                    if target.dim() == 1:
                        target = target.unsqueeze(0)  # thêm chiều đầu tiên → [1, N]
                    
                    output = []
                    path = paths[idx]
                    target =target.to(self.device)
                    for (num_layer, layer) in zip(range(len(outputs)), outputs):
                        output.append(self.head_to_yolo(layer[idx], self.strides[num_layer], self.img_size))

                    output_tensor = torch.cat([fmap.flatten(1) for fmap in output], dim=1)
                    sample_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                    mask = torch.zeros_like(output_tensor, device=self.device)
                    if path in self.assign_map: mask = self.assign_map[path]
                    else:
                        assigned_cells = []
                        best_cells = {}
                        for bbox in target:
                            cls, x, y, w, h = bbox                        
                            reg_tensor = torch.tensor([x, y, w, h], device=self.device)
                            assign_list = self.gt_assign(x, y, w, h)
                            for fmap_idx, cells in assign_list:
                                for i, j, iou in cells:
                                    assigned_cells.append((fmap_idx, i, j, iou, cls, reg_tensor))
                        #print(assigned_cells)
                        iou_thresh = 0.3
                        # filtered = [c for c in assigned_cells if c[3] > iou_thresh]
                        for fmap_idx, i, j, iou, cls, reg_tensor in assigned_cells:
                            i = int(i)
                            j = int(j)
                            key = (fmap_idx, i, j)
                            if key not in best_cells or iou > best_cells[key][0]:
                                best_cells[key] = (iou,reg_tensor, 1.0, cls)  # 1.0 = objness
                        # print(best_cells)
                        for (fmap_idx, i, j), (iou, reg_tensor, obj, cls) in best_cells.items():
                            # flatten index
                            start_idx = sum([e**2 for e in self.edge[:fmap_idx]])  # tổng số ô trước fmap_idx
                            idx = start_idx + i * self.edge[fmap_idx] + j  # int index
                            idx = int(idx)
                            one_hot_cls = torch.zeros(self.num_classes, device=self.device)
                            one_hot_cls[int(cls)] = 1.0
                            mask[:, idx] = torch.cat([reg_tensor, torch.tensor([obj], device=self.device), one_hot_cls])
                        self.assign_map[path] = mask
                    obj_channel_index = 4
                    has_gt = mask[obj_channel_index, :] > 0
                    no_gt  = ~has_gt
                    output_has_gt = output_tensor[:, has_gt]  # (C, num_gt)
                    mask_has_gt   = mask[:, has_gt]           # (C, num_gt)
                    output_no_gt  = output_tensor[:, no_gt]   # (C, num_no_gt)
                    mask_no_gt    = mask[:, no_gt]            # chỉ cần objness
                    # Phần có gt
                    # if output_has_gt.numel() > 0:
                    reg_loss = self.reg_loss(output_has_gt[:4, :], mask_has_gt[:4, :])
                    obj_loss = self.objness_loss(output_has_gt[4:5, :], mask_has_gt[4:5, :])
                    cls_loss = self.cls_loss(output_has_gt[5:, :], mask_has_gt[5:, :])
                    loss_has_gt = reg_loss + obj_loss + cls_loss
                    # else:
                        # if not mask.any():
                        #     print(" toàn 0")
                        # print(f"path no gt: {path}")   
                        # print(target)
                        # # print("kkkkkkk")
                        # loss_has_gt = torch.tensor(0.0, device=output_has_gt.device)

                    # Phần không có gt (objness = 0)
                    obj_loss_no_gt = self.objness_loss(output_no_gt[4:5, :], torch.zeros_like(output_no_gt[4:5, :]))
                    sample_loss += loss_has_gt + 1/3 * obj_loss_no_gt
                    # print(f"Sample obj loss: {loss_has_gt.item()}")   
                    # print(f"Sample no obj loss: {obj_loss_no_gt.item()}")   



                    # assigned_cells_set = set()

                    # assigned_cells = []
                    # if path in self.assign_map: assigned_cells = self.assign_map[path]
                    # else:
                    #     for bbox in target:
                    #         cls, x, y, w, h = bbox                        
                    #         reg_tensor = torch.tensor([x, y, w, h], device=self.device)
                    #         assign_list = self.gt_assign(x, y, w, h)
                    #         for fmap_idx, cells in assign_list:
                    #             for i, j, iou in cells:
                    #                 assigned_cells.append((fmap_idx, i, j, iou, cls, reg_tensor))
                    #     self.assign_map[path] = assigned_cells

                    # gt_boxes = []
                    # gt_labels = []
                    # for bbox in target:
                    #     cls, x, y, w, h = bbox
                    #     gt_boxes.append(self.yolo_to_xyxy(x, y, w, h, self.img_size))
                    #     gt_labels.append(cls.long())

                    target = target.to(self.device)
                    if target.numel() == 0:
                        continue
                    else:
                        if target.dim() == 1:
                            target = target.unsqueeze(0)  # shape (1, 5)
                        gt_labels = target[:, 0].long()
                        gt_boxes = self.yolo_to_xyxy(target[:, 1], target[:, 2], target[:, 3], target[:, 4], self.img_size)

                    if gt_boxes.numel() == 0:
                        gt_boxes_tensor = torch.zeros((0, 4), device=self.device)
                        gt_labels_tensor = torch.zeros((0,), device=self.device, dtype=torch.long)
                    else:
                        gt_boxes_tensor = gt_boxes.to(self.device)
                        gt_labels_tensor = gt_labels.to(self.device)

                    gts_for_metric.append({"boxes":gt_boxes_tensor, "labels":gt_labels_tensor})

                    # pred_boxes_nms = []
                    # pred_score_nms = []
                    # pred_cls_nms = []

                    # iou_thresh = 0.3
                    # filtered = [c for c in assigned_cells if c[3] > iou_thresh]
                    # best_cells = {}
                    # for fmap_idx, i, j, iou, cls, reg_tensor in filtered:
                    #     i = int(i) if not isinstance(i, int) else i
                    #     j = int(j) if not isinstance(j, int) else j
                    #     key = (fmap_idx, i, j)
                    #     if key not in best_cells or iou > best_cells[key][3]:
                    #         best_cells[key] = (fmap_idx, i, j, iou, cls, reg_tensor)

                    # # Compute loss
                    # for fmap_idx, i, j, iou, cls, reg_tensor in best_cells.values():
                    #     pred = output[fmap_idx][:, i, j]
                    #     pred_reg = pred[:4]
                    #     pred_obj = pred[4:5]
                    #     pred_cls = pred[5:]
                    #     sample_loss += (
                    #             self.cls_loss(pred_cls, cls)
                    #             + self.reg_loss(pred_reg, reg_tensor)
                    #             + self.objness_loss(pred_obj, torch.ones_like(pred_obj))
                    #             )
                        
                    # assigned_cells_set = {(fmap_idx, i, j) for fmap_idx, i, j, iou, cls, reg_tensor in best_cells.values()}

                    # # Negative cells (obj = 0)
                    # for fmap_idx, fmap in enumerate(output):
                    #     H, W = fmap.shape[1], fmap.shape[2]
                    #     for i in range(H):
                    #         for j in range(W):
                    #             pred = output[fmap_idx][:, i, j]
                    #             x, y, w, h = pred[:4]
                    #             pred_obj = pred[4:5]
                    #             pred_cls = pred[5:]
                    #             pred_boxes_nms.append(self.yolo_to_xyxy(x, y, w, h, self.img_size))
                    #             score = (pred_obj * torch.max(pred_cls)).squeeze()  # đảm bảo 0D scalar
                    #             pred_score_nms.append(score)
                    #             pred_cls_nms.append(torch.argmax(pred_cls))
                    #             if (fmap_idx, i, j) not in assigned_cells_set:
                    #                 pred_obj = fmap[4, i, j]
                    #                 sample_loss += 1/3 * self.objness_loss(pred_obj, torch.zeros_like(pred_obj))

                    # pred_boxes_nms = torch.stack(pred_boxes_nms).to(self.device)
                    # pred_score_nms = torch.stack(pred_score_nms).to(self.device)
                    # pred_cls_nms = torch.stack(pred_cls_nms).to(self.device)

                    pred_boxes_nms = self.yolo_to_xyxy(
                                output_tensor[0, :],  # x
                                output_tensor[1, :],  # y
                                output_tensor[2, :],  # w
                                output_tensor[3, :],  # h
                                self.img_size
                        )  # Kết quả có shape (num_no_gt, 4)
                    pred_score_nms = output_tensor[4, :]
                    pred_cls_nms = output_tensor[5: , :].argmax(dim=0)
                    keep = batched_nms(pred_boxes_nms, pred_score_nms, pred_cls_nms, iou_threshold=0.5)
                    keep_boxes = pred_boxes_nms[keep]
                    keep_scores = pred_score_nms[keep]
                    keep_cls = pred_cls_nms[keep]

                    # Lọc theo threshold 0.6
                    mask = keep_scores > 0.6
                    keep_boxes = keep_boxes[mask]
                    keep_scores = keep_scores[mask]
                    keep_cls = keep_cls[mask]

                    # Giữ tối đa 5 bbox
                    if len(keep_scores) > 5:
                        topk_scores, topk_idx = torch.topk(keep_scores, 5)
                        keep_boxes = keep_boxes[topk_idx]
                        keep_scores = topk_scores
                        keep_cls = keep_cls[topk_idx]


                    keep_boxes_tensor = keep_boxes.to(self.device)
                    keep_scores_tensor = keep_scores.to(self.device)
                    keep_cls_tensor = keep_cls.to(self.device)

                    preds_for_metric.append({"boxes":keep_boxes_tensor, "scores":keep_scores_tensor, "labels":keep_cls_tensor})

                    batch_loss = batch_loss + sample_loss  
                    # print(f"Sample {i} loss: {sample_loss.item()}")   

                # Cập nhật mAP
                batch_loss = batch_loss/len(targets)
                # print(f"Batch {i} loss: {batch_loss.item()}")  

                total_loss += batch_loss
                metric_map.update(preds_for_metric, gts_for_metric)

        # Tính trung bình loss và mAP
        avg_loss = total_loss / len(val_loader)
        map_results = metric_map.compute()

        if verbose:
            print(f"Validation loss: {avg_loss:.4f}")
            print(f"mAP@0.5: {map_results['map_50']:.4f}, mAP@0.5:0.95: {map_results['map']:.4f}")

        return {"val_loss": avg_loss, "mAP": map_results}


    def predict(self, path):
        self.model.eval()
        
        with torch.no_grad():
            image = Image.open(path).convert('L')  # 1 channel
            image = image.resize((self.img_size, self.img_size))
            image = torch.tensor(np.array(image)/255., dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape [1,1,H,W]
            image = image.to(self.device)
            outputs = self.model(image)
            output = []
            for (num_layer, layer) in zip(range(len(outputs)), outputs):
                output.append(self.head_to_yolo(layer[0], self.strides[num_layer], self.img_size))
            pred_boxes_nms = []
            pred_score_nms = []
            pred_cls_nms = []
            for fmap_idx, fmap in enumerate(output):
                H, W = fmap.shape[1], fmap.shape[2]
                for i in range(H):
                    for j in range(W):
                        pred = output[fmap_idx][:, i, j]
                        x, y, w, h = pred[:4]
                        pred_obj = pred[4:5]
                        pred_cls = pred[5:]
                        pred_boxes_nms.append(self.yolo_to_xyxy(x, y, w, h, self.img_size))
                        score = (pred_obj * torch.max(pred_cls)).squeeze()  # đảm bảo 0D scalar
                        pred_score_nms.append(score)
                        pred_cls_nms.append(torch.argmax(pred_cls))
                                

            pred_boxes_nms = torch.stack(pred_boxes_nms).to(self.device)
            pred_score_nms = torch.stack(pred_score_nms).to(self.device)
            pred_cls_nms = torch.stack(pred_cls_nms).to(self.device)
            keep = batched_nms(pred_boxes_nms, pred_score_nms, pred_cls_nms, iou_threshold=0.5)
            keep_boxes = pred_boxes_nms[keep]
            keep_scores = pred_score_nms[keep]
            keep_cls = pred_cls_nms[keep]

            # Lọc theo threshold 0.6
            mask = keep_scores > 0.6
            keep_boxes = keep_boxes[mask]
            keep_scores = keep_scores[mask]
            keep_cls = keep_cls[mask]

            # Giữ tối đa 5 bbox
            if len(keep_scores) > 5:
                topk_scores, topk_idx = torch.topk(keep_scores, 5)
                keep_boxes = keep_boxes[topk_idx]
                keep_scores = topk_scores
                keep_cls = keep_cls[topk_idx]
        return keep_boxes, keep_scores, keep_cls
    
    def img_predict(self, path, class_names=class_names):
        keep_boxes, keep_scores, keep_cls = self.predict(path)
    #     # ======= DEBUG =======
    #     print("Number of boxes after NMS:", len(keep_boxes))
    #     print("Boxes:", keep_boxes)
    #     print("Scores:", keep_scores)
    #     print("Classes:", keep_cls)
    # # =====================
        img_result = visualize_mri_prediction(path, keep_boxes, keep_scores, keep_cls, class_names)
        cv2.imwrite("predicted.png", img_result)
        print("Prediction saved to predicted.png")


    def save_model(self, path, epoch=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "history": self.history
        }
        if self.optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch
    
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")


