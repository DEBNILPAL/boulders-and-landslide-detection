# convert_masks_to_yolo.py

import os
import cv2
import numpy as np
from PIL import Image

IMAGE_DIR = 'images'
MASK_DIR = 'masks'
LABEL_DIR = 'labels'
CLASS_ID = 0  # e.g., 0 for landslide, change if needed

os.makedirs(LABEL_DIR, exist_ok=True)

image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]

for image_file in image_files:
    image_path = os.path.join(IMAGE_DIR, image_file)
    mask_path = os.path.join(MASK_DIR, image_file.replace('.jpg', '.png'))  # or .tif

    if not os.path.exists(mask_path):
        print(f"[⚠️] Mask not found for {image_file}")
        continue

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)  # grayscale
    h, w = image.shape[:2]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_path = os.path.join(LABEL_DIR, image_file.replace('.jpg', '.txt'))
    with open(label_path, 'w') as f:
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Normalize
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            norm_w = bw / w
            norm_h = bh / h
            f.write(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    print(f"[✅] Labels saved for {image_file}")
