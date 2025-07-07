import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tifffile as tiff

# ============ CONFIG ============
IMAGE_DIR = "images/"
MASK_DIR = "masks/"
IMG_SIZE = 256  # Resize to 256x256
BATCH_SIZE = 8
EPOCHS = 20
MODEL_NAME = "unet_landslide.h5"
# ================================

def load_images(image_dir, mask_dir, img_size):
    images, masks = [], []
    for filename in tqdm(os.listdir(image_dir)):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):

            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, os.path.splitext(filename)[0] + ".tif")

            if not os.path.exists(mask_path):
                print(f"Warning: mask not found for {filename}, skipping.")
                continue

            # Load and resize input image
            img = cv2.imread(image_path)
            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0  # Normalize
            images.append(img)

            # Load and resize mask
            mask = tiff.imread(mask_path)
            if len(mask.shape) > 2:  # Convert RGB mask to single channel
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = cv2.resize(mask, (img_size, img_size))
            mask = (mask > 0).astype(np.float32)  # Binarize
            masks.append(mask[..., np.newaxis])  # Add channel dimension

    return np.array(images), np.array(masks)

# ============ U-NET MODEL ============
def build_unet(input_shape):
    inputs = layers.Input(input_shape)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return x

    def encoder_block(x, filters):
        f = conv_block(x, filters)
        p = layers.MaxPooling2D()(f)
        return f, p

    def decoder_block(x, skip, filters):
        us = layers.UpSampling2D()(x)
        concat = layers.Concatenate()([us, skip])
        return conv_block(concat, filters)

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bottleneck
    b1 = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    return models.Model(inputs, outputs)
# =====================================

# ============ MAIN ============
def main():
    print("📦 Loading data...")
    X, y = load_images(IMAGE_DIR, MASK_DIR, IMG_SIZE)
    print(f"Loaded {len(X)} image-mask pairs.")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🧠 Building model...")
    model = build_unet((IMG_SIZE, IMG_SIZE, 3))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("🚀 Training...")
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              batch_size=BATCH_SIZE,
              epochs=EPOCHS)

    print(f"💾 Saving model as {MODEL_NAME}")
    model.save(MODEL_NAME)

    print("✅ Done.")

if __name__ == "__main__":
    main()
