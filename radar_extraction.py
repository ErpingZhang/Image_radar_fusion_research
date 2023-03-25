from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
import numpy as np

# Set the data root directory where the nuScenes dataset is located
data_root = 'data_set'

# Create a NuScenes object for the dataset version and configuration you want to use
nusc = NuScenes(version='v1.0-mini', dataroot=data_root)

# Get the first radar pointcloud
sample_token = nusc.sample[0]['token']
sensor_token = nusc.get('sample', sample_token)['data']['RADAR_FRONT']
pc_path = nusc.get('sample_data', sensor_token)['filename']
radar_pc = RadarPointCloud.from_file(pc_path)

# Get one radar point
point_index = 0
point = radar_pc.points[point_index]

# Get the sensor pose (extrinsic matrix)
calib_data = nusc.get('calibrated_sensor', sensor_token)
sensor_pose = np.array(calib_data['sensor_t'])
sensor_extrinsic = np.zeros((4, 4))
sensor_extrinsic[:3, :3] = sensor_pose[:3, :3]
sensor_extrinsic[:3, 3] = sensor_pose[:3, 3]
sensor_extrinsic[3, 3] = 1.0

# Get the global pose of the radar sensor (ego vehicle)
ego_pose = np.array(nusc.get('ego_pose', nusc.get('sample_data', sensor_token)['ego_pose_token'])['translation'])
ego_transformation = np.eye(4)
ego_transformation[:3, 3] = ego_pose

# Transform the radar point to the ego vehicle frame
point = np.append(point, 1.0)  # add homogeneous coordinate
point = sensor_extrinsic @ point
point = np.linalg.inv(ego_transformation) @ point

# Print the transformed point in world coordinates
print(f'Point in world coordinates: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})')
