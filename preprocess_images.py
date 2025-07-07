# --- File: src/1_preprocess_images.py ---
import cv2
import numpy as np
import os

def preprocess_image(image_path):
    """
    Load a lunar image, convert to grayscale, apply histogram equalization,
    and apply Gaussian blur to enhance edges and reduce noise.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)     # 5 , 5
    return img

def save_processed(image, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


