import torch
from src.model.BrainTumorv2_ONNX import BrainTumorv2_ONNX


CKPT_PATH = "BrainTumorv2_legendary.pth.tar"
ONNX_PATH = "BrainTumorv2.onnx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint(model, path, device="cpu"):
    """Load checkpoint giống code bạn đang dùng"""
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"[OK] Loaded checkpoint from {path}")

    return model


def export_onnx(model, onnx_path):
    model.eval()

    # dummy input (MRI grayscale -> 1 channel)
    dummy_input = torch.randn(1, 1, 640, 640).to(DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={
            "images": {0: "batch"},
            "output": {0: "batch"}
        }
    )

    print(f"[OK] Exported ONNX model to {onnx_path}")


def main():
    # 1. build model
    model = BrainTumorv2_ONNX(num_classes=4).to(DEVICE)

    # 2. load checkpoint
    model = load_checkpoint(model, CKPT_PATH, DEVICE)

    # 3. export ONNX
    export_onnx(model, ONNX_PATH)


if __name__ == "__main__":
    main()