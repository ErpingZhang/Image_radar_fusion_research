import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

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
# T = np.array(nusc.get('ego_pose', sample_data['ego_pose_token'])['translation'])
T = np.array(nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])['translation'])  # (x,y,z)
q = np.array(nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])['rotation'])  # （w,x,y,z)
q /= np.linalg.norm(q)
R = np.array([[1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
[2*q[1]*q[2] + 2*q[0]*q[3], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]],
[2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], 1 - 2*q[1]**2 - 2*q[2]**2]])

#R = np.array([[2*q[0]**2 + 2*q[1]**2 - 1, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
#[2*q[1]*q[2] + 2*q[0]*q[3], 2*q[0]**2 - 2*q[2]**2 - 1, 2*q[2]*q[3] - 2*q[0]*q[1]],
#[2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], 2*q[0]**2 - 2*q[3]**2 - 1]])

#R = Quaternion(q).rotation_matrix

M = np.matmul(K, np.hstack((R, T.reshape(-1, 1))))