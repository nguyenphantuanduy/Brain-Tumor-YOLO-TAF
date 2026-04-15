# 🧠 Brain Tumor Detection using YOLOv8 + Neutral-Negative Loss

- This project is an AI-based web application for brain tumor detection from MRI images using YOLOv8l combined with Neutral-Negative Loss.
- It provides an interactive web interface for uploading MRI images and visualizing detection results.

# 🎯 Project Objectives

- Build an effective brain tumor detection model using YOLOv8l
- Improve robustness using Neutral-Negative Loss
- Provide a user-friendly web interface for real-time inference
- Deployable using both local environment (venv) and Docker

# ⚙️ System Requirements

- Python >= 3.8
- Node.js >= 22 (for frontend)
- pip / venv
- Docker (optional)

## 🚀 1. Run with Virtual Environment (Recommended for development)

### 📦 Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate # Windows

# source venv/bin/activate # Linux/Mac

pip install -r requirements.txt
```

### 📥 Download model

```bash
python download_model.py
```

▶ Run backend

```bash
uvicorn ai_server:app --reload
```

Backend runs at:
http://localhost:8000

### 🎨 Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:

http://localhost:5173
### ⚠️ Notes (Local setup)

If errors occur:

Install missing dependency:

```bash
pip install python-multipart
```

Fix CORS:

Ensure backend includes:

allow_origins=["http://localhost:5173"]

## 🐳 2. Run with Docker (Recommended for deployment)

📦 Build and start containers

```bash
docker compose up --build
```

### 🌐 Services

Service URL

- Frontend http://localhost:5173
- Backend API http://localhost:8000

### 🧠 Docker architecture

React Frontend → FastAPI Backend → YOLOv8 Model → Prediction Output

### ⚠️ Notes (Docker)

- First run may take longer due to model download
- Ensure ports 5173 and 8000 are not occupied
- If changes in Dockerfile → rebuild required:
  docker compose build

# 📁 Project Structure

```text
Brain-Tumor-YOLO-TAF/
│
├── backend/
│ ├── ai_server.py
│ ├── download_model.py
│ ├── requirements.txt
│ ├── Dockerfile
│ └── ...
│
├── frontend/
│ ├── src/
│ ├── package.json
│ └── ...
│
├── docker-compose.yml
└── README.md
```

# 👨‍💻 Authors

- Nguyễn Phan Tuấn Duy
- Võ Thiện Đăng Khoa

# 📌 Notes

This project is built for educational and research purposes, focusing on applying deep learning to medical image analysis.
