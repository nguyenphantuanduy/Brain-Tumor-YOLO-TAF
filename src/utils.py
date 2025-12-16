import torch
import cv2

def yolo_collate_fn(batch):
    # batch là list của ((image_path, image_tensor), label_tensor)
    imgs = torch.stack([x[0][1] for x in batch], dim=0)  # x[0][1] = image_tensor
    paths = [x[0][0] for x in batch]  # x[0][0] = image_path
    labels = [x[1] for x in batch]    # label_tensor
    return paths, imgs, labels


def visualize_mri_prediction(image_path, boxes, scores, labels, class_names=None, model_input_size=640):
    """
    Vẽ bounding box và nhãn lên ảnh MRI gốc, tránh chữ bị đè chồng.
    boxes là output từ model (640x640), scale về kích thước gốc.
    """
    import cv2
    import numpy as np

    # Load ảnh grayscale và chuyển sang RGB
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    orig_h, orig_w = img.shape[:2]

    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()

    used_text_positions = []

    # Tỉ lệ scale từ 640x640 -> ảnh gốc
    scale_x = orig_w / model_input_size
    scale_y = orig_h / model_input_size

    for box, score, label in zip(boxes, scores, labels):
        # Scale bbox về ảnh gốc
        x1, y1, x2, y2 = box
        print(f"Box: [{x1},{y1},{x2},{y2}], Score: {score:.4f}, Label: {label}")
        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)

        # print(f"Box: [{x1},{y1},{x2},{y2}], Score: {score:.4f}, Label: {label}")

        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f"{class_names[label] if class_names else label}: {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        text_x = x1
        text_y = max(y1 - 5, text_h + 5)

        for used_x, used_y, used_h in used_text_positions:
            if abs(text_y - used_y) < text_h + 4 and abs(text_x - used_x) < text_w:
                text_y = used_y + used_h + 4

        used_text_positions.append((text_x, text_y, text_h))

        # Vẽ nền và text
        cv2.rectangle(img, (text_x, text_y - text_h - 2), (text_x + text_w, text_y + baseline), color, -1)
        cv2.putText(img, text, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img



def yolo_to_xyxy(x, y, w, h, img_size):
        if isinstance(img_size, int):
            H = W = img_size
        else:
            H, W = img_size

        # Đảm bảo tất cả đều là tensor (tránh trường hợp float/int)
        device = x.device if isinstance(x, torch.Tensor) else ("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.as_tensor(x, device=device, dtype=torch.float32)
        y = torch.as_tensor(y, device=device, dtype=torch.float32)
        w = torch.as_tensor(w, device=device, dtype=torch.float32)
        h = torch.as_tensor(h, device=device, dtype=torch.float32)
        # Tính toán bằng broadcast (chạy được cả batch)
        x1 = (x - w / 2) * W
        y1 = (y - h / 2) * H
        x2 = (x + w / 2) * W
        y2 = (y + h / 2) * H

        converted = torch.stack([x1, y1, x2, y2], dim=-1)
        converted = torch.clamp(converted, min=0, max=max(W-1,H-1))


        return converted

import torch

def shifted_sigmoid(x, shift=6.0):
    return 1 / (1 + torch.exp(-(x - shift)))


import torch
from torchvision.ops import box_iou

# class PrecisionRecall:
#     def __init__(self, iou_threshold=0.5):
#         self.iou_threshold = iou_threshold
#         self.tp = 0
#         self.fp = 0
#         self.fn = 0

#     @torch.no_grad()
#     def update(self, preds, targets):
#         """
#         preds, targets: list of dicts with keys 'boxes' [N,4], 'labels' [N]
#         """
#         for pred, target in zip(preds, targets):
#             boxes_pred = pred['boxes']
#             labels_pred = pred['labels']
#             boxes_gt = target['boxes']
#             labels_gt = target['labels']

#             device = boxes_gt.device  # lấy device của gt

#             if boxes_pred.numel() == 0:
#                 self.fn += boxes_gt.size(0)
#                 continue

#             matched_gt = set()
#             ious = box_iou(boxes_pred.to(device), boxes_gt)  # đảm bảo cùng device

#             for i, p_label in enumerate(labels_pred):
#                 p_label = p_label.to(device)
#                 mask = (labels_gt == p_label)

#                 iou_vals = ious[i][mask]
#                 gt_indices = torch.arange(len(labels_gt), device=device)[mask]

#                 if len(iou_vals) == 0:
#                     self.fp += 1
#                     continue

#                 max_iou, idx = iou_vals.max(0)
#                 gt_idx = gt_indices[idx]
#                 if max_iou >= self.iou_threshold and gt_idx.item() not in matched_gt:
#                     self.tp += 1
#                     matched_gt.add(gt_idx.item())
#                 else:
#                     self.fp += 1

#             self.fn += boxes_gt.size(0) - len(matched_gt)

#     def compute(self):
#         precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
#         recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
#         return {'precision': precision, 'recall': recall}

import torch
from torchvision.ops import box_iou

class PrecisionRecall:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.preds = []   # list of dicts
        self.gts = []     # list of dicts

    @torch.no_grad()
    def update(self, preds, targets):
        """
        preds:   list of dicts {boxes, scores, labels}
        targets: list of dicts {boxes, labels}
        """
        for pred, target in zip(preds, targets):
            self.preds.append({
                "boxes": pred["boxes"].detach(),
                "scores": pred["scores"].detach(),
                "labels": pred["labels"].detach()
            })
            self.gts.append({
                "boxes": target["boxes"].detach(),
                "labels": target["labels"].detach()
            })
    @torch.no_grad()
    def compute(self):
        if len(self.preds) == 0:
            return {"precision": 0.0, "recall": 0.0}

        device = self.preds[0]["boxes"].device

        all_boxes = []
        all_scores = []
        all_labels = []
        image_ids = []

        gt_boxes = []
        gt_labels = []

        img_id = 0
        for pred, gt in zip(self.preds, self.gts):
            n = pred["boxes"].size(0)

            if n > 0:
                all_boxes.append(pred["boxes"])
                all_scores.append(pred["scores"])
                all_labels.append(pred["labels"])
                image_ids.append(
                    torch.full((n,), img_id, device=device, dtype=torch.long)
                )

            gt_boxes.append(gt["boxes"])
            gt_labels.append(gt["labels"])
            img_id += 1

        if len(all_boxes) == 0:
            return {"precision": 0.0, "recall": 0.0}

        all_boxes = torch.cat(all_boxes)
        all_scores = torch.cat(all_scores)
        all_labels = torch.cat(all_labels)
        image_ids = torch.cat(image_ids)

        # 🔑 SORT GLOBAL THEO CONFIDENCE
        order = torch.argsort(all_scores, descending=True)
        all_boxes = all_boxes[order]
        all_scores = all_scores[order]
        all_labels = all_labels[order]
        image_ids = image_ids[order]

        matched = set()
        TP = 0
        FP = 0
        FN = 0

        precisions = []
        recalls = []

        total_gt = sum(gt.size(0) for gt in gt_boxes)

        for i in range(len(all_boxes)):
            img = image_ids[i].item()
            label = all_labels[i]

            gtb = gt_boxes[img]
            gtl = gt_labels[img]

            mask = (gtl == label)

            if mask.sum() == 0:
                FP += 1
            else:
                ious = box_iou(
                    all_boxes[i].unsqueeze(0),
                    gtb[mask]
                )[0]

                max_iou, idx = ious.max(0)
                gt_idx = torch.arange(len(gtl), device=device)[mask][idx].item()
                key = (img, gt_idx)

                if max_iou >= self.iou_threshold and key not in matched:
                    TP += 1
                    matched.add(key)
                else:
                    FP += 1

            precision = TP / (TP + FP + 1e-6)
            recall = TP / (total_gt + 1e-6)

            precisions.append(precision)
            recalls.append(recall)

        precisions = torch.tensor(precisions)
        recalls = torch.tensor(recalls)

        f1 = 2 * precisions * recalls / (precisions + recalls + 1e-6)
        best_idx = torch.argmax(f1)

        return {
            "precision": precisions[best_idx].item(),
            "recall": recalls[best_idx].item()
        }


