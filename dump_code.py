import numpy as np
from PIL import Image
import re
import os

def parse_lbl(lbl_path):
    metadata = {}
    with open(lbl_path, 'r') as f:
        for line in f:
            # Simple key = value parsing
            match = re.match(r"^\s*(\w+)\s*=\s*(.*)", line)
            if match:
                key, value = match.groups()
                value = value.strip().strip('"')
                metadata[key.upper()] = value
    return metadata

def convert_img_lbl_to_jpg(lbl_path, output_path=None):
    meta = parse_lbl(lbl_path)

    # Get the corresponding .IMG file path
    img_path = os.path.splitext(lbl_path)[0] + ".IMG"
    if not os.path.exists(img_path):
        print("Corresponding .IMG file not found.")
        return

    # Extract key metadata
    samples = int(meta.get("LINE_SAMPLES", 0))      # Width
    lines = int(meta.get("LINES", 0))               # Height
    dtype_label = meta.get("SAMPLE_TYPE", "UNSIGNED_INTEGER")

    # Handle common data types
    dtype_map = {
        "UNSIGNED_INTEGER": np.uint8,
        "MSB_INTEGER": ">i2",  # Big endian 16-bit signed
        "LSB_INTEGER": "<i2",  # Little endian 16-bit signed
        "MSB_UNSIGNED_INTEGER": ">u2",
        "LSB_UNSIGNED_INTEGER": "<u2",
        "PC_REAL": "<f4",      # 32-bit float, little-endian
        "IEEE_REAL": ">f4",    # 32-bit float, big-endian
    }

    dtype = dtype_map.get(dtype_label, np.uint8)

    # Read raw image data
    with open(img_path, 'rb') as f:
        img = np.frombuffer(f.read(), dtype=dtype)
        img = img.reshape((lines, samples))

    # Normalize for 8-bit display
    img_min, img_max = img.min(), img.max()
    img_8bit = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

    pil_img = Image.fromarray(img_8bit).convert("RGB")

    if output_path is None:
        output_path = os.path.splitext(lbl_path)[0] + ".jpg"

    pil_img.save(output_path, "JPEG")
    print(f"✅ Converted to: {output_path}")

# Example usage:
convert_img_lbl_to_jpg("image.LBL")
