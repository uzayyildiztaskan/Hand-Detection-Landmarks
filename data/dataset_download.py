import json
import os
import requests
from tqdm import tqdm

def download_filtered_dataset(config):
    FILTERED_ANNOTATIONS_PATH = config.FILTERED_ANNOTATIONS_PATH
    SAVE_DIR = config.IMAGES_PATH
    os.makedirs(SAVE_DIR, exist_ok=True)

    with open(FILTERED_ANNOTATIONS_PATH, 'r') as f:
        filtered_annotations = json.load(f)

    for key, value in tqdm(filtered_annotations.items(), desc="Downloading images"):
        img_url = value['coco_url']
        img_filename = os.path.basename(img_url)
        save_path = os.path.join(SAVE_DIR, img_filename)

        if os.path.exists(save_path):
            continue

        try:
            response = requests.get(img_url, timeout=10)

            if response.status_code == 200:
                with open(save_path, 'wb') as img_file:
                    img_file.write(response.content)
            else:
                print(f"[WARN] Could not fetch {img_url} (Status code: {response.status_code})")

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to download {img_url}:\n{e}")
