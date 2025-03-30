import json
from tqdm import tqdm
import random

def filter_annotations(config, non_valid_image_split=0.3):

    with open(config.ANNOTATIONS_PATH, 'r') as file:
        data = json.load(file)

    filtered_anottations_json = {}
    no_hand_images_json ={}
    counter = 0
    num_no_hand_images = 0
    limit_no_hand_images = (int)(len(data['annotations']) * non_valid_image_split)

    for annotation in tqdm(data['annotations'], desc="Processing annotations"):
            

        image_id = annotation['image_id']

        if image_id == 391895:
            print(f"DEBUG: Image -> 391895\nDEBUG: Left Hand Valid: {annotation['lefthand_valid']}\nRight Hand Valid: {annotation['righthand_valid']}")

        image_info = next((img for img in data['images'] if img['id'] == image_id), None)
        
        if image_info:
            key = f"{counter:06d}"

            value = {
                "righthand_box": annotation['righthand_box'],
                "lefthand_box": annotation['lefthand_box'],
                "righthand_kpts": annotation['righthand_kpts'],
                "lefthand_kpts": annotation['lefthand_kpts'],
                "right_hand_valid": annotation['righthand_valid'],
                "left_hand_valid": annotation['lefthand_valid'],
                "file_name": image_info['file_name'],
                "height": image_info['height'],
                "width": image_info['width'],
                "coco_url": image_info['coco_url'],
                "image_id": image_id
            }

            
            if annotation['lefthand_valid'] or annotation['righthand_valid']:    
            
                filtered_anottations_json[key] = value
            
                counter += 1

            else:
                no_hand_images_json[key] = value

    num_hand_images = counter
    shuffled_no_hand_keys = list(no_hand_images_json.keys())
    random.shuffle(shuffled_no_hand_keys)

    for key in shuffled_no_hand_keys:
        if num_no_hand_images >= limit_no_hand_images:
            break
        counter += 1
        num_no_hand_images += 1
        filtered_anottations_json[counter] = no_hand_images_json[key]


    print(f"Total hand images: {num_hand_images}")
    print(f"Total no-hand images: {num_no_hand_images}")
    print(f"Total images: {counter}")

    print(f"Hand images: {num_hand_images/counter*100}%")
    print(f"No-hand images: {num_no_hand_images/counter*100}%")


    with open(config.FILTERED_ANNOTATIONS_PATH, 'w') as f:
        json.dump(filtered_anottations_json, f, indent=4)