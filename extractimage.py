from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
import numpy as np
import os
from PIL import Image
def extract_img_path():
    # Set the path to the NuScenes dataset
    data_root = 'data_set'

    # Set the version of the NuScenes dataset
    nusc_version = 'v1.0-mini'

    # Set the scene and sample index for the image you want to extract
    scene_idx = 0
    sample_idx = 0

    # Load the NuScenes dataset
    nusc = NuScenes(version=nusc_version, dataroot=data_root)

    # Get the scene and sample data for the given indices
    scene = nusc.scene[scene_idx]
    sample = nusc.get('sample', scene['first_sample_token'])

    # Get the camera data for the given sample
    cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])

    # Get the image filename and path for the given camera data
    image_filename = os.path.join(data_root, cam_data['filename'])
    image_path = nusc.get_sample_data_path(sample['data']['CAM_FRONT'])

    # Load the image using the PIL library
    img = Image.open(image_path)

    # Print the size of the image
    print(image_path)
    print("Image size: ", img.size)
    return image_path