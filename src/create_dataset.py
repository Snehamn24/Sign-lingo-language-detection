# ============================================================
# CREATE DATASET FOR CNN-LSTM TRAINING
# ============================================================
#
# This script:
# 1. Reads extracted frames
# 2. Converts them into NumPy arrays
# 3. Assigns labels to each sign word
# 4. Splits data into training and testing sets
# 5. Saves everything as .npy files
#
# Output:
#
# dataset/processed/numpy/
# ├── X_train.npy
# ├── X_test.npy
# ├── y_train.npy
# └── y_test.npy
#
# ============================================================

import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# ============================================================
# PATHS
# ============================================================

# Get project root folder automatically

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Folder containing extracted frames

FRAMES_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "frames"
)

# Folder where NumPy arrays will be saved

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "numpy"
)

# Create output folder if it doesn't exist

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

# Image size used during frame extraction

IMG_SIZE = 224

# Every video contains 30 frames

FRAMES_PER_VIDEO = 30

# Classes (sign words)

CLASSES = [
    "book",
    "drink",
    "computer",
    "chair",
    "go"
]


# ============================================================
# DATA CONTAINERS
# ============================================================

# X = video frames
# y = labels

X = []
y = []


# ============================================================
# READ ALL CLASSES
# ============================================================

for label_index, class_name in enumerate(CLASSES):

    print(f"\nReading class: {class_name}")

    # Example:
    #
    # dataset/processed/frames/book

    class_folder = os.path.join(
        FRAMES_DIR,
        class_name
    )

    # Go through each video folder

    for video_folder in os.listdir(class_folder):

        video_path = os.path.join(
            class_folder,
            video_folder
        )

        # Skip if not a directory

        if not os.path.isdir(video_path):
            continue

        frames = []

        # ------------------------------------------------
        # Read 30 frames
        # ------------------------------------------------

        for i in range(FRAMES_PER_VIDEO):

            frame_path = os.path.join(
                video_path,
                f"frame_{i:03d}.jpg"
            )

            # Skip incomplete videos

            if not os.path.exists(frame_path):
                break

            # Read image

            image = cv2.imread(frame_path)

            # Resize again for safety

            image = cv2.resize(
                image,
                (IMG_SIZE, IMG_SIZE)
            )

            # Normalize pixels
            #
            # 0-255  →  0-1

            image = image / 255.0

            frames.append(image)

        # ------------------------------------------------
        # Add only complete videos
        # ------------------------------------------------

        if len(frames) == FRAMES_PER_VIDEO:

            X.append(frames)

            # Store label
            #
            # book = 0
            # drink = 1
            # computer = 2
            # chair = 3
            # go = 4

            y.append(label_index)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

print("\nConverting to NumPy arrays...")

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(y)

# Convert labels to one-hot encoding
#
# Example:
#
# 0 -> [1,0,0,0,0]
# 1 -> [0,1,0,0,0]
# 2 -> [0,0,1,0,0]

y = to_categorical(
    y,
    num_classes=len(CLASSES)
)

print("\nDataset Statistics")

print("X shape:", X.shape)
print("y shape:", y.shape)

# Example:
#
# X shape = (57, 30, 224, 224, 3)
#
# Meaning:
#
# 57 videos
# 30 frames/video
# 224x224 image
# RGB channels


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

print("\nCreating train-test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# SAVE DATASET
# ============================================================

print("\nSaving NumPy files...")

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_train.npy"
    ),
    X_train
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_test.npy"
    ),
    X_test
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_train.npy"
    ),
    y_train
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_test.npy"
    ),
    y_test
)

print("\nDataset creation completed successfully!")

print("\nSaved files:")

print("X_train.npy")
print("X_test.npy")
print("y_train.npy")
print("y_test.npy")