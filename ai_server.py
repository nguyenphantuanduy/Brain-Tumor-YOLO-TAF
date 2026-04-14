import os
import torch
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from src.model.BrainTumorv2 import BrainTumorv2
from src.model.MyBrainTumorWrapperv4 import MyBrainTumorWrapperv4

device = torch.device("cpu")

app = FastAPI(title="Brain Tumor AI Server")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Load model ONCE khi start server =====
print("Loading model...")

model = BrainTumorv2().to(device)

wrapper = MyBrainTumorWrapperv4(
    model,
    CKPT_PATH="BrainTumorv2_legendary.pth.tar",
    device=device
)

print("Model loaded ✔")

# ===== Helper: convert image =====
def preprocess_image(file_bytes):
    image = Image.open(file_bytes).convert("RGB")
    image = np.array(image)
    return image


# ===== API ROUTE =====
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # đọc bytes
        image_bytes = await file.read()

        from io import BytesIO
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = np.array(image)

        # predict
        result = wrapper.img_predict(image)

        if result is None:
            return JSONResponse(
                content={"success": False, "message": "Prediction failed"}
            )

        # nếu là numpy image
        if isinstance(result, np.ndarray):
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_img = Image.fromarray(result)
        else:
            result_img = result

        # encode base64
        import base64
        from io import BytesIO

        buffer = BytesIO()
        result_img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return {
            "success": True,
            "prediction_image": img_str
        }

    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)}
        )