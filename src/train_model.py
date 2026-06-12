# Load X_train, X_test, y_train, y_test
# Build CNN-LSTM model
# Train model
# Test accuracy
# Save trained model
# ============================================================
# TRAIN CNN-LSTM MODEL FOR SIGN LINGO
# ============================================================

import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    TimeDistributed,
    Conv2D,
    MaxPooling2D,
    Flatten,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "numpy"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

FRAMES_PER_VIDEO = 30
IMG_SIZE = 224
CHANNELS = 3
NUM_CLASSES = 5

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sign_lingo_model.h5"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# BUILD CNN-LSTM MODEL
# ============================================================
#
# CNN extracts visual features from each frame.
# LSTM learns the movement/order of frames across time.
#
# Input shape:
# 30 frames, each frame 224x224 with 3 color channels
#
# (30, 224, 224, 3)

print("\nBuilding CNN-LSTM model...")

model = Sequential()

# CNN layer applied separately to every frame using TimeDistributed
model.add(
    TimeDistributed(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu"
        ),
        input_shape=(
            FRAMES_PER_VIDEO,
            IMG_SIZE,
            IMG_SIZE,
            CHANNELS
        )
    )
)

model.add(
    TimeDistributed(
        MaxPooling2D(pool_size=(2, 2))
    )
)

model.add(
    TimeDistributed(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu"
        )
    )
)

model.add(
    TimeDistributed(
        MaxPooling2D(pool_size=(2, 2))
    )
)

# Convert CNN feature maps into a 1D feature vector for each frame
model.add(
    TimeDistributed(
        Flatten()
    )
)

# LSTM learns motion pattern across 30 frames
model.add(
    LSTM(
        units=64,
        return_sequences=False
    )
)

# Dropout helps reduce overfitting
model.add(
    Dropout(0.5)
)

# Dense layer for learning final patterns
model.add(
    Dense(
        units=64,
        activation="relu"
    )
)

# Output layer
# 5 neurons because we have 5 sign classes
model.add(
    Dense(
        units=NUM_CLASSES,
        activation="softmax"
    )
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining model...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=2
)


# ============================================================
# EVALUATE MODEL
# ============================================================

print("\nEvaluating model...")

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(MODEL_PATH)

print("\nModel saved successfully!")
print("Saved at:", MODEL_PATH)