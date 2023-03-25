import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import os.path as osp

# Set the path to the NuScenes dataset and version
data_root = 'data_set'
nusc_version = 'v1.0-mini'

# Set the scene and sample indices
scene_idx = 0
sample_idx = 0

# Set the point to be projected in the ego vehicle coordinate system
point = np.array([10, 0, 0])
print(point)

# Load the NuScenes dataset
nusc = NuScenes(version=nusc_version, dataroot=data_root)

# Get the sample data for the given sample index
sample_data = nusc.get('sample_data', nusc.sample[sample_idx]['data']['CAM_FRONT'])

# Load the camera intrinsic matrix K and the extrinsic matrix T
cam_calib = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
K = np.array(cam_calib['camera_intrinsic'])
#T = np.array(nusc.get('ego_pose', sample_data['ego_pose_token'])['translation'])
T = np.array(nusc.get('calibrated_sensor',sample_data['calibrated_sensor_token'])['translation']) #(x,y,z)
q = np.array(nusc.get('calibrated_sensor',sample_data['calibrated_sensor_token'])['rotation']) #（w,x,y,z)
#q /= np.linalg.norm(q)
#R = np.array([[1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
              #[2*q[1]*q[2] + 2*q[0]*q[3], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]],
              #[2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], 1 - 2*q[1]**2 - 2*q[2]**2]])

#R = np.array([[2*q[0]**2 + 2*q[1]**2 - 1, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
              #[2*q[1]*q[2] + 2*q[0]*q[3], 2*q[0]**2 - 2*q[2]**2 - 1, 2*q[2]*q[3] - 2*q[0]*q[1]],
              #[2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], 2*q[0]**2 - 2*q[3]**2 - 1]])

R = Quaternion(q).rotation_matrix

M = np.matmul(K, np.hstack((R, T.reshape(-1, 1))))
print("This is K")
print(K)
print("This is T")
print(T)
print("This is R")
print(R)

# Project 3D point [x, y, z] onto 2D image plane
p = np.array([1, 1, 0, 1])
p_img = np.matmul(M, p)
p_img = p_img / p_img[2]  # normalize by depth
u, v = p_img[:2]  # pixel coordinates of projected point
print(u,v)
# Load the image and draw a circle at the projected pixel coordinates
image_path = nusc.get_sample_data_path(sample_data['token'])
image = cv2.imread(image_path)
cv2.circle(image, (int(u), int(v)), radius=3, color=(0, 255, 0), thickness=-1)

# Show the image with the projected point
cv2.imshow('Image with projected point', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Transform the point from the ego vehicle coordinate system to the camera coordinate system
x = np.hstack((point, 1))
print(x)
#point_camera = T.cross(x)#[:3]
point_camera = np.cross(T,x)
print(point_camera)

# Project the point onto the camera image plane
point_image = view_points(point_camera.reshape(1, 3), K)[0]

# Convert the homogeneous coordinates to pixel coordinates
pixel_coords = (point_image / point_image[2])[:2].astype(int)

# Load the image and draw a circle at the projected pixel coordinates
image_path = nusc.get_sample_data_path(sample_data['token'])
image = cv2.imread(image_path)
cv2.circle(image, (pixel_coords[0], pixel_coords[1]), radius=3, color=(0, 255, 0), thickness=-1)

# Show the image with the projected point
cv2.imshow('Image with projected point', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
