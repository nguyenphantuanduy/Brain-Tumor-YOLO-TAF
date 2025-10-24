import torch
import cv2

def yolo_collate_fn(batch):
    imgs = torch.stack([x[0] for x in batch], dim=0)  # tất cả ảnh phải cùng H,W
    labels = [x[1] for x in batch]  # giữ list, mỗi phần tử tensor [num_bbox, 5]
    return imgs, labels

def visualize_mri_prediction(image_path, boxes, scores, labels, class_names=None):
    """
    image_path: path tới ảnh MRI grayscale
    boxes: tensor [num_boxes, 4] (x1,y1,x2,y2)
    scores: tensor [num_boxes]
    labels: tensor [num_boxes]
    class_names: list tên class
    """
    # Load ảnh grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # [H,W], 1 channel
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)        # convert sang 3 channel để vẽ màu

    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)  # màu xanh
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{class_names[label] if class_names else label}: {score:.2f}"
        cv2.putText(img, text, (x1, max(y1-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img

def yolo_to_xyxy(x, y, w, h, img_size):
    if isinstance(img_size, int):
        W = H = img_size
    else:
        H, W = img_size

    # Convert
    x1 = (x - w / 2) * W
    y1 = (y - h / 2) * H
    x2 = (x + w / 2) * W
    y2 = (y + h / 2) * H

    # Stack lại
    converted = torch.stack([x1, y1, x2, y2])

    # Clamp out-of-place (không dùng inplace)
    converted = torch.stack([
        converted[0].clamp(0, W - 1),
        converted[1].clamp(0, H - 1),
        converted[2].clamp(0, W - 1),
        converted[3].clamp(0, H - 1)
    ])

    return converted
