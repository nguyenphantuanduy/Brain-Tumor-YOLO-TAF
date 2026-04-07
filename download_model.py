import gdown
import os

file_id = "1SDNVUu738fwqP2R9Gmp-nUjJvX73WDkp"

url = f"https://drive.google.com/uc?id={file_id}"

output = "BrainTumorv2_legendary.pth.tar"

os.makedirs("experiments", exist_ok=True)

gdown.download(url, output, quiet=False)