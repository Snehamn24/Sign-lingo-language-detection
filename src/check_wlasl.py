import json
import os

JSON_PATH = "dataset/WLASL_v0.3.json"
VIDEOS_PATH = "dataset/videos"

with open(JSON_PATH, "r") as file:
    data = json.load(file)

print("Total words in JSON:", len(data))
print("Videos folder exists:", os.path.exists(VIDEOS_PATH))

print("\nFirst 10 words:")
for item in data[:10]:
    print(item["gloss"])