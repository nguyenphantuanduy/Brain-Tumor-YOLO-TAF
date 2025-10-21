# main.py
from src.preprocessing import*

if __name__ == "__main__":
    train_list = load_list("data/train_list.pkl")
    print(f"✅ Đã load train_list, tổng {len(train_list)} ảnh")

    # Kiểm tra 5 ảnh đầu
    print("\n--- 5 ảnh đầu ---")
    for i, (img_path, bboxes) in enumerate(train_list[:5]):
        print(f"\nẢnh {i+1}: {img_path}")
        print(f"Bboxes: {bboxes}")

    # Kiểm tra xem có ảnh bị mất file hay bbox trống không
    missing_imgs = [img for img, _ in train_list if not Path(img).exists()]
    empty_bboxes = [img for img, bbox in train_list if len(bbox) == 0]
    print(f"\nSố ảnh mất file: {len(missing_imgs)}")
    print(f"Số ảnh bbox rỗng: {len(empty_bboxes)}")




