import gdown
import os

file_id = "1SDNVUu738fwqP2R9Gmp-nUjJvX73WDkp"

url = f"https://drive.google.com/uc?id={file_id}"

output = "BrainTumorv2_legendary.pth.tar"

# Nếu file đã tồn tại → skip
if os.path.exists(output):
    print("✅ Model already exists. Skip downloading.")
else:
    print("⬇️ Downloading model...")
    gdown.download(url, output, quiet=False)
    print("✅ Download completed.")