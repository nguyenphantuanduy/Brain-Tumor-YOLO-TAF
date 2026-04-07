import os
import sys

# Ép PyTorch chạy CPU, tắt CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
device = torch.device("cpu")  # Dùng cho model và tensor

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from src.model.BrainTumorv2 import BrainTumorv2
from src.model.MyBrainTumorWrapperv4 import MyBrainTumorWrapperv4

# Hàm lấy đường dẫn đúng trong exe
def resource_path(relative_path):
    try:
        # Khi chạy exe PyInstaller
        base_path = sys._MEIPASS
    except Exception:
        # Khi chạy script gốc
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)



class BrainTumorApp:
    def resize_background(self, event=None):
        win_w = self.root.winfo_width()
        win_h = self.root.winfo_height()

        if win_w <= 1 or win_h <= 1:
            return

        img_w, img_h = self.bg_original.size

        scale = min(win_w / img_w, win_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        resized = self.bg_original.resize((new_w, new_h), Image.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(resized)

        self.bg_label.config(image=self.bg_photo)
        self.bg_label.place(
            x=(win_w - new_w) // 2,
            y=(win_h - new_h) // 2
        )
        
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Detection")
        self.root.geometry("1100x600")
        self.root.configure(bg="#ffffff")

        # Load background image
        self.bg_original = Image.open(resource_path("BrainTumorBG.jpg"))
        self.bg_photo = ImageTk.PhotoImage(self.bg_original)

        self.bg_label = tk.Label(root, bg="#ffffff", image=self.bg_photo)
        self.bg_label.place(x=0, y=0)
        self.root.bind("<Configure>", self.resize_background)

        # Frames
        self.left_frame = tk.Frame(root, bg="#ffffff", highlightbackground="black", highlightthickness=1)
        self.left_frame.place(relx=0.25, rely=0.5, anchor="center", width=400, height=400)
        self.right_frame = tk.Frame(root, bg="#ffffff", highlightbackground="black", highlightthickness=1)
        self.right_frame.place(relx=0.75, rely=0.5, anchor="center", width=400, height=400)

        # Labels for images
        self.image_label = tk.Label(self.left_frame, bg="#ffffff")
        self.image_label.pack(fill="both", expand=True)
        self.result_label = tk.Label(self.right_frame, bg="#ffffff")
        self.result_label.pack(fill="both", expand=True)

        # Buttons
        self.select_btn = tk.Button(root, text="🖼 Chọn ảnh", command=self.load_image,
                                    bg="white", fg="black", font=("Arial", 12, "bold"),
                                    relief="solid", borderwidth=1, width=15)
        self.select_btn.place(relx=0.4, rely=0.9, anchor="center")
        self.predict_btn = tk.Button(root, text="Dự đoán", command=self.predict,
                                     bg="white", fg="black", font=("Arial", 12, "bold"),
                                     relief="solid", borderwidth=1, width=15)
        self.predict_btn.place(relx=0.6, rely=0.9, anchor="center")

        # Status text
        self.status_label = tk.Label(root, text="", bg="#ffffff", fg="black",
                                     font=("Arial", 11, "italic"))
        self.status_label.place(relx=0.5, rely=0.06, anchor="center")
        
        # Footer disclaimer (y tế)
        self.footer_label = tk.Label(
            root,
            text="Kết quả dự đoán chỉ mang tính chất hỗ trợ, không thay thế chẩn đoán của bác sĩ.",
            bg="#ffffff",
            fg="#555555",
            font=("Arial", 9, "italic")
        )
        self.footer_label.place(relx=0.5, rely=0.97, anchor="center")


        # Load model
        print("Loading model, please wait...")
        self.status_label.config(text="Đang tải mô hình...")
        root.update()

        model = BrainTumorv2().to(device)  # Ép model về CPU
        self.wrapper = MyBrainTumorWrapperv4(model,
                                              CKPT_PATH=resource_path("BrainTumorv2_legendary.pth.tar"),
                                              device=device)  # Nếu wrapper có nhận device

        print("✅ Model loaded successfully.")
        self.status_label.config(text="Mô hình đã sẵn sàng!")
        self.image_path = None

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return
        self.image_path = file_path
        img = Image.open(file_path).resize((400, 400))
        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_img)
        self.result_label.config(image="")  # clear old result
        self.status_label.config(text="Ảnh đã được chọn.")

    def predict(self):
        if not self.image_path:
            self.status_label.config(text="Hãy chọn ảnh trước khi dự đoán.")
            return

        self.status_label.config(text="Đang dự đoán...")
        self.root.update()

        print(f"Running prediction on {self.image_path} ...")
        # self.wrapper.img_predict(self.image_path)
        # predicted_path = resource_path("predicted.png")
        # if os.path.exists(predicted_path):
        #     img = Image.open(predicted_path).resize((400, 400))
        #     self.tk_result = ImageTk.PhotoImage(img)
        #     self.result_label.config(image=self.tk_result)
        #     self.status_label.config(text="✅ Dự đoán hoàn tất!")
        #     print("✅ Prediction displayed.")
        # else:
        #     self.status_label.config(text="❌ Không tìm thấy kết quả dự đoán.")
        #     print("❌ No prediction result found.")
        import cv2
        from PIL import Image
        import numpy as np
        img = self.wrapper.img_predict(self.image_path)
        if img is None:
            self.status_label.config(text="❌ Dự đoán thất bại.")
            return
        # Nếu img là OpenCV image (numpy)
        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        if img is not None:
            img = img.resize((400, 400))
            self.tk_result = ImageTk.PhotoImage(img)
            self.result_label.config(image=self.tk_result)
            self.status_label.config(text="✅ Dự đoán hoàn tất!")
            print("✅ Prediction displayed.")
        else:
            self.status_label.config(text="❌ Không tìm thấy kết quả dự đoán.")
            print("❌ No prediction result found.")


if __name__ == "__main__":
    root = tk.Tk()
    app = BrainTumorApp(root)
    root.mainloop()
