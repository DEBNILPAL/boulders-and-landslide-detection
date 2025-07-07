# --- File: src/2_boulder_detection.py ---

import cv2
from typing import List
import numpy as np

def detect_boulders(image):
    """
    Detect individual boulders in the image using adaptive threshold and filtering.
    Returns list of tuples (x, y, diameter) for each boulder.
    """
    # Step 1: Apply Gaussian Blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Step 2: Adaptive thresholding to isolate darker blobs
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Step 3: Morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Step 4: Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boulders = []


    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20 < area < 5000:  # filter by area
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2 + 1e-6)

            # filter by circularity (boulders tend to be round)
            if circularity > 0.5:
                mask = np.zeros_like(image)
                cnt = np.array(cnt)        # ✅ Optional fix to suppress IDE warnings
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_val = cv2.mean(image, mask=mask)[0]

                # filter very dark blobs (likely shadows)
                if mean_val > 50:
                    boulders.append((int(x), int(y), int(radius * 2)))

    return boulders
