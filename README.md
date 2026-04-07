# PHÁT HIỆN U NÃO SỬ DỤNG YOLOv8l VỚI HÀM MẤT MÁT NEUTRAL-NEGATIVE

Ứng dụng này sử dụng mô hình **YOLOv8l** kết hợp với **Neutral-Negative Loss** để **phát hiện u não trên ảnh y tế**. Người dùng có thể dễ dàng tải ảnh và nhận kết quả dự đoán trực quan thông qua giao diện đồ họa.

---

## 🎯 Mục tiêu dự án

- Phát triển mô hình **nhận diện u não** hiệu quả dựa trên **YOLOv8l**.
- Sử dụng **Neutral-Negative Loss** để cải thiện độ chính xác của mô hình.
- Cung cấp **giao diện người dùng thân thiện** để dễ dàng thử nghiệm và sử dụng.

---

## ⚙️ Yêu cầu hệ thống

- Python >= 3.8
- Các thư viện được liệt kê trong `requirements.txt`

Cài đặt tất cả thư viện bằng lệnh:

```bash
pip install -r requirements.txt
```
## 📥 Tải mô hình đã huấn luyện

Do file mô hình có dung lượng lớn nên không được lưu trực tiếp trong repository.
Sau khi cài đặt thư viện, hãy tải mô hình bằng lệnh:

```bash
python download_model.py
```

Lệnh này sẽ tải file mô hình:
BrainTumorv2_legendary.pth.tar
## 🚀 Hướng dẫn sử dụng

Sau khi tải mô hình thành công, chạy ứng dụng:
```bash
python app.py
```
Các bước sử dụng:
Chờ giao diện người dùng xuất hiện.
Nhấn Chọn ảnh để tải ảnh cần dự đoán.
Chọn file ảnh từ máy tính
(hỗ trợ .jpg, .png, ...).
Nhấn Dự đoán.
Kết quả sẽ hiển thị khung bao quanh vùng u não trên ảnh.
## 📁 Cấu trúc thư mục chính
Brain-Tumor-YOLO-TAF/
│
├── app.py
├── download_model.py
├── requirements.txt
│
├── experiments/
│   └── BrainTumorv2_legendary.pth.tar
│
├── models/
├── utils/
└── data/
## 👨‍💻 Tác giả
Nguyễn Phan Tuấn Duy
Võ Thiện Đăng Khoa
## 📌 Ghi chú

Dự án được thực hiện cho mục đích học tập và nghiên cứu.
