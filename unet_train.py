# --- File: src/ml_models/unet_train.py ---
import tensorflow as tf
from tensorflow.keras import layers, models

def get_unet(input_shape=(128, 128, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)

    # Decoder
    u1 = layers.UpSampling2D()(c3)
    concat1 = layers.Concatenate()([u1, c2])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat1)

    u2 = layers.UpSampling2D()(c4)
    concat2 = layers.Concatenate()([u2, c1])
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(concat2)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)

    model = models.Model(inputs, outputs)
    return model
