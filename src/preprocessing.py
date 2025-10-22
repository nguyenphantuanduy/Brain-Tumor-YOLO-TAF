from pathlib import Path
from src.config import *

def load_yolo_txt(file_path):
    file_path = Path(file_path)
    bboxes = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            parts = line.split()
            # Chuyển thành float, class_id giữ nguyên số nguyên
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            bboxes.append([class_id] + coords)    
    return bboxes

def createListFromPath(path):
    """
    Tạo hashmap: key = path ảnh, value = list bbox từ file txt cùng tên
    """
    itemList = []
    for tumorcls in class_names:
        data_dir = Path(path) / tumorcls  # thư mục Train/Glioma ...
        print(f"Thư mục {tumorcls} tồn tại:", data_dir.exists())
        if not data_dir.exists():
            continue
        img_dir = data_dir / "images"
        label_dir = data_dir / "labels"
        img_list = list(img_dir.glob("*.jpg"))
        for img_path in img_list:
            # Lấy file txt cùng tên
            txt_path = label_dir / (img_path.stem + ".txt")
            if not txt_path.exists():
                print(f"Bỏ qua: {img_path.name} (không có label)")
                continue
            bboxes = load_yolo_txt(txt_path)
            itemList.append((str(img_path), bboxes))
    return itemList

import pickle

def save_list(train_list, save_path):
    """
    Lưu danh sách (img_path, bboxes) vào file .pkl
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(train_list, f)
    print(f"✅ Đã lưu list vào: {save_path}")

def load_list(save_path):
    """
    Đọc danh sách (img_path, bboxes) từ file .pkl
    """
    with open(save_path, "rb") as f:
        train_list = pickle.load(f)
    print(f"✅ Đã load list từ: {save_path}")
    return train_list

aug_plan = {
    "Glioma": 350,
    "Meningioma": 100,
    "NoTumor": 900,
    "Pituitary": 150
}

from pathlib import Path

def split_by_class(train_list):
    """
    Tách train_list thành 4 danh sách riêng biệt theo lớp.
    Mỗi phần tử trong train_list là (img_path, bboxes)
    """
    # Khởi tạo dict chứa list cho từng lớp
    split_dict = {cls: [] for cls in class_names}

    for img_path, bboxes in train_list:
        p = Path(img_path)
        for cls in class_names:
            # Nếu tên lớp xuất hiện trong đường dẫn ảnh
            if cls.lower() in str(p).lower():
                split_dict[cls].append((img_path, bboxes))
                break

    return split_dict

import cv2
import random
import pickle
from pathlib import Path
import albumentations as A
from .RicianNoise import RicianNoise

def img_augment(train_list, save_path):
    """
    Augment dữ liệu train_list và lưu vào save_path.
    - Mỗi ảnh có thể được augment n_per_image lần (default=1)
    - Bbox được cập nhật tương ứng (YOLO format)
    """
    save_path = Path(save_path)

    # Pipeline augmentation
    transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.CLAHE(p=1),
        ], p=0.5),

        A.OneOf([
            A.RandomGamma(p=1),
        ], p=0.3),

        RicianNoise(std=0.05, p=0.3),

        A.ElasticTransform(alpha=40, sigma=6, p=0.3),
        A.GridDistortion(distort_limit=0.2, border_mode=cv2.BORDER_REFLECT_101, p=0.3),

        # Bias Field Augmentation (vignette-like)
        A.RandomShadow(p=0.1),
        A.RandomFog(p=0.1),
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
    )

    print(f"Bắt đầu augment {len(train_list)} ảnh...")
    counter = 0
    split_list = split_by_class(train_list)
    for img_cls, num in aug_plan.items():
        cls_list = split_list[img_cls]
        img_dir = save_path / img_cls / "images"
        lbl_dir = save_path / img_cls / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num):
            img_path, bboxes = random.choice(cls_list)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Không đọc được ảnh: {img_path}")
                continue
            class_labels = [b[0] for b in bboxes]
            bbox_only = [b[1:] for b in bboxes]  # (x, y, w, h)
            transformed = transform(image=img, bboxes=bbox_only, class_labels=class_labels)
            aug_img = transformed['image']
            aug_bboxes = transformed['bboxes']
            aug_labels = transformed['class_labels']

            # Tạo tên file mới
            stem = Path(img_path).stem
            out_img = img_dir / f"{stem}_aug{i}.jpg"
            out_lbl = lbl_dir / f"{stem}_aug{i}.txt"

            # Lưu ảnh
            cv2.imwrite(str(out_img), aug_img)

            # Lưu bbox (YOLO format)
            with open(out_lbl, "w") as f:
                for cls, (x, y, w, h) in zip(aug_labels, aug_bboxes):
                    cls = int(cls)
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            counter += 1
    print(f"✅ Đã augment xong {counter} ảnh và lưu vào {save_path}")


    









