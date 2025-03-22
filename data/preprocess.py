import json
import numpy as np
from tqdm import tqdm

## This is a preprocessing script to convert 3D keypoint annotations to 2D annotations.
## This is done by normalizing the x,y coordinates of the keypoints by dividing them with the depth of the keypoints.
## Note: The dataset doesn't provide 2D x,y keypoints, instead only the 3D keypoints are available

def convert_to_2d(hp):
    
    with open(hp.KEYPOINT_ANNOTATION_3D_PATH) as f:
        xyz = np.array(json.load(f))

    with open(hp.INTRINSIC_CAMERA_MATRIX_PATH) as f:
        K = np.array(json.load(f))

    keypoints_2d = []

    for i in tqdm(range(len(xyz))):
        joints = xyz[i]
        k_matrix = K[i]
        
        joints_2d = []
        for joint in joints:
            X, Y, Z = joint
            u = (k_matrix[0, 0] * X + k_matrix[0, 2] * Z) / Z
            v = (k_matrix[1, 1] * Y + k_matrix[1, 2] * Z) / Z
            joints_2d.append([u, v])
        
        keypoints_2d.append(joints_2d)

    keypoints_2d = np.array(keypoints_2d)

    np.save(hp.KEYPOINT_ANNOTATION_2D_PATH, keypoints_2d)
    print("Saved 2D keypoints as training_kp2d.np")
