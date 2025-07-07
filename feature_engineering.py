# --- File: src/7_feature_engineering.py ---
import numpy as np

def generate_pixel_features(image, slope_map, aspect_map):
    """
    Combine pixel intensity, slope, and aspect into a feature matrix for ML.
    """
    h, w = image.shape
    features = []
    for i in range(h):
        for j in range(w):
            features.append([
                image[i, j],
                slope_map[i, j],
                aspect_map[i, j]
            ])
    return np.array(features)