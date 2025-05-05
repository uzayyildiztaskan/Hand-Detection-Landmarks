import json
from tqdm import tqdm
import random

def filter_annotations(config, non_valid_image_split=0.3):

    with open(config.ANNOTATIONS_PATH, 'r') as file:
        data = json.load(file)

    filtered_annotations_json = {}
    no_hand_images_json ={}
    person_id = 0
    num_hand_images = 0
    num_no_hand_images = 0

    for annotation in tqdm(data['annotations'], desc="Processing annotations"):            

        image_id = annotation['image_id']

        image_info = next((img for img in data['images'] if img['id'] == image_id), None)

        image_id = f"{annotation['image_id']:012d}"
        
        if image_info:
            key = f"{person_id:06d}"

            value_person = {
                "righthand_box": annotation['righthand_box'],
                "lefthand_box": annotation['lefthand_box'],
                "righthand_kpts": annotation['righthand_kpts'],
                "lefthand_kpts": annotation['lefthand_kpts'],
                "right_hand_valid": annotation['righthand_valid'],
                "left_hand_valid": annotation['lefthand_valid']                
            }

            value_image = {
                "file_name": image_info['file_name'],
                "height": image_info['height'],
                "width": image_info['width'],
                "coco_url": image_info['coco_url'],
                "image_id": image_id,
                "has_valid_hands": False,
                "hand_annotations": {}
            }

            
            if annotation['lefthand_valid'] or annotation['righthand_valid']:

                if image_id in no_hand_images_json.keys():

                    filtered_annotations_json[image_id] = no_hand_images_json.pop(image_id)

                    filtered_annotations_json[image_id]['has_valid_hands'] = True

                    num_no_hand_images -= 1

                elif not image_id in filtered_annotations_json.keys():

                    value_image['has_valid_hands'] = True 

                    filtered_annotations_json[image_id] = value_image

                    num_hand_images += 1

                filtered_annotations_json[image_id]['hand_annotations'][key] = value_person

            else:

                if not image_id in no_hand_images_json.keys():  

                    no_hand_images_json[image_id] = value_image

                    num_no_hand_images += 1

                no_hand_images_json[image_id]['hand_annotations'][key] = value_person
            
            
            person_id += 1

    shuffled_no_hand_keys = list(no_hand_images_json.keys())
    random.shuffle(shuffled_no_hand_keys)

    count = 0
    limit_no_hand_images = (int)((non_valid_image_split / (1 - non_valid_image_split)) * num_hand_images)
    total_images = num_hand_images + limit_no_hand_images

    for key in shuffled_no_hand_keys:
        if count >= limit_no_hand_images:
            break
        count += 1
        filtered_annotations_json[key] = no_hand_images_json[key]


    print(f"Total hand images: {num_hand_images}")
    print(f"Total no-hand images: {limit_no_hand_images}")
    print(f"Total images: {total_images}")

    print(f"Hand images: {num_hand_images/total_images*100}%")
    print(f"No-hand images: {limit_no_hand_images/total_images*100}%")


    with open(config.FILTERED_ANNOTATIONS_PATH, 'w') as f:
        json.dump(filtered_annotations_json, f, indent=4)