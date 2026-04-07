# PHÁT HIỆN U NÃO SỬ DỤNG YOLOV8L VỚI HÀM MẤT MÁT NEUTRAL-NEGATIVE

Ứng dụng này sử dụng mô hình **YOLOv8l** kết hợp với **Neutral-Negative Loss** để **phát hiện u não trên ảnh y tế**. Người dùng có thể dễ dàng tải ảnh và nhận kết quả dự đoán trực quan thông qua giao diện đồ họa.

---

## Mục tiêu dự án

- Phát triển mô hình **nhận diện u não** hiệu quả dựa trên **YOLOv8l**.
- Sử dụng **Neutral-Negative Loss** để cải thiện độ chính xác của mô hình.
- Cung cấp **giao diện người dùng thân thiện** để dễ dàng thử nghiệm và sử dụng.

---

## Yêu cầu

- Python >= 3.8
- Thư viện được liệt kê trong `requirements.txt`.  
  Cài đặt tất cả các thư viện bằng lệnh:

```bash
pip install -r requirements.txt
```

## Hướng dẫn sử dụng

- Chạy ứng dụng:

```bash
python app.py
```

- Chờ giao diện người dùng xuất hiện.

- Chọn ảnh cần dự đoán:
  - Nhấn vào nút Chọn ảnh.

  - Chọn file ảnh từ máy tính (hỗ trợ định dạng .jpg, .png, …).

- Thực hiện dự đoán:
  - Nhấn vào nút Dự đoán.

  - Mô hình sẽ phát hiện u não trên ảnh và hiển thị khung bao quanh khối u trên giao diện.

- Xem kết quả: Kết quả dự đoán được hiển thị trực tiếp trên giao diện.

## Tác giả

- Nguyễn Phan Tuấn Duy

- Võ Thiện Đăng Khoa

## Dự án thực hiện cho mục đích học tập.
