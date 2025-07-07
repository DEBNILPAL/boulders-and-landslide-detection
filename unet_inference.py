import os
from keras.models import load_model
import cv2
import numpy as np

def run_unet_segmentation(image_path):
    model_path = os.path.join(os.path.dirname(__file__), "weights", "unet_landslide.h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ U-Net model not found at {model_path}")

    model = load_model(model_path)

    # Read and resize image to model input
    image = cv2.imread(image_path)
    resized = cv2.resize(image, (256, 256))
    input_data = resized / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    # Predict
    mask = model.predict(input_data)[0, :, :, 0]
    mask = (mask > 0.5).astype(np.uint8) * 255

    # Resize mask back to original image size
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    return mask
