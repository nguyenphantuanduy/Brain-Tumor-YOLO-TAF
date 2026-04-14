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

- Do file mô hình có dung lượng lớn nên không được lưu trực tiếp trong repository.
- Sau khi cài đặt thư viện, hãy tải mô hình bằng lệnh:

```bash
python download_model.py
```

Lệnh này sẽ tải file mô hình: BrainTumorv2_legendary.pth.tar

## 🚀 Hướng dẫn sử dụng (README)

### 1. Chạy Backend (FastAPI)

Mở terminal tại thư mục project và chạy:

```bash
uvicorn ai_server:app --reload
```

Backend sẽ chạy tại:
http://localhost:8000

### 2. Chạy Frontend (React + Vite)

Mở terminal mới và thực hiện:

```bash
cd frontend
npm install
npm run dev
```

Frontend sẽ chạy tại:
http://localhost:5173

### 3. Sử dụng ứng dụng

- Mở trình duyệt và truy cập: http://localhost:5173
- Nhấn Chọn ảnh để upload ảnh MRI từ máy tính (.jpg, .png, ...)
- Nhấn Upload Image để hiển thị ảnh lên giao diện
- Nhấn Run Prediction để gửi ảnh đến AI model
- Kết quả sẽ hiển thị:
- Ảnh MRI gốc (input)
- Ảnh dự đoán có bounding box vùng u não

### ⚠️ Lưu ý

- Backend phải được chạy trước khi mở frontend
- Nếu gặp lỗi upload, cài thêm dependency:

```bash
pip install python-multipart
```

- Nếu lỗi CORS, đảm bảo backend có cấu hình:
  allow_origins=["http://localhost:5173"]

## 📁 Cấu trúc thư mục chính

```text
Brain-Tumor-YOLO-TAF/
│
├── app.py
├── download_model.py
├── requirements.txt
│
├── src/
│
├── BrainTumorv2_legendary.pth.tar
│   (sẽ được tạo sau khi chạy download_model.py)
│
└── README.md
```

## 👨‍💻 Tác giả

- Nguyễn Phan Tuấn Duy
- Võ Thiện Đăng Khoa

## 📌 Ghi chú

Dự án được thực hiện cho mục đích học tập và nghiên cứu.
