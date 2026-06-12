# Import required libraries

import json          # To read WLASL_v0.3.json
import os            # To work with folders and file paths
import cv2           # OpenCV for reading videos and saving frames


# ----------------------------------------------------
# PROJECT PATHS
# ----------------------------------------------------

# Gets the project root folder automatically
# Example:
# D:\Sign-lingo-language-detection

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Path of WLASL JSON file
JSON_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "WLASL_v0.3.json"
)

# Path containing all videos
VIDEOS_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "videos"
)

# Folder where extracted frames will be saved
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "frames"
)


# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------

# We start with only 5 words
# Later we can increase to 20, 50, 100 words

SELECTED_WORDS = [
    "book",
    "drink",
    "computer",
    "chair",
    "go"
]

# Use maximum 20 videos per word
MAX_VIDEOS_PER_WORD = 20

# Extract 30 frames from each video
FRAMES_PER_VIDEO = 30

# Resize every image to 224x224
# Standard size used in CNN models
IMG_SIZE = 224


# ----------------------------------------------------
# FUNCTION TO EXTRACT FRAMES
# ----------------------------------------------------

def extract_frames(video_path, output_folder):

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Total frames available in video
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # Video could not be read
    if total_frames <= 0:
        print("Could not read:", video_path)
        return

    # Calculate interval
    # Example:
    #
    # Total frames = 300
    # Need only 30
    #
    # step = 300 / 30 = 10
    #
    # We take:
    # 0,10,20,30,40...
    #
    step = max(
        total_frames // FRAMES_PER_VIDEO,
        1
    )

    saved = 0
    frame_no = 0

    while saved < FRAMES_PER_VIDEO:

        # Jump directly to desired frame
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_no
        )

        success, frame = cap.read()

        if not success:
            break

        # Resize image
        frame = cv2.resize(
            frame,
            (IMG_SIZE, IMG_SIZE)
        )

        # Create filename
        frame_name = f"frame_{saved:03d}.jpg"

        # Save image
        cv2.imwrite(
            os.path.join(
                output_folder,
                frame_name
            ),
            frame
        )

        saved += 1
        frame_no += step

    cap.release()


# ----------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------

def main():

    print("Loading WLASL JSON...")

    # Open JSON file
    with open(JSON_PATH, "r") as file:
        data = json.load(file)

    print("Total words in dataset:", len(data))

    # Loop through all words
    for item in data:

        # Example:
        # book
        # drink
        # computer
        gloss = item["gloss"]

        # Skip unwanted words
        if gloss not in SELECTED_WORDS:
            continue

        print("\n" + "=" * 50)
        print("Processing word:", gloss)

        count = 0

        # Each word contains many videos
        for instance in item["instances"]:

            # Stop after 20 videos
            if count >= MAX_VIDEOS_PER_WORD:
                break

            # Example video ID:
            # 12345

            video_id = instance["video_id"]

            # Convert to:
            # 12345.mp4

            video_path = os.path.join(
                VIDEOS_DIR,
                video_id + ".mp4"
            )

            # Skip missing video
            if not os.path.exists(video_path):
                continue

            # Output structure:
            #
            # frames/
            #   book/
            #      12345/
            #          frame_001.jpg
            #          frame_002.jpg

            output_folder = os.path.join(
                OUTPUT_DIR,
                gloss,
                video_id
            )

            print(
                f"Extracting {video_id}.mp4"
            )

            extract_frames(
                video_path,
                output_folder
            )

            count += 1

        print(
            f"Completed {count} videos for {gloss}"
        )

    print("\nDataset preparation completed!")


# ----------------------------------------------------
# PROGRAM ENTRY POINT
# ----------------------------------------------------

if __name__ == "__main__":
    main()