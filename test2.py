import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import os.path as osp
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from PIL import Image
import matplotlib.pyplot as plt
import hdbscan
from sklearn.cluster import DBSCAN
from yolov5_master.yolov5_master.detect_new import detect # return xyxy, confidence, class

# Set the path to the NuScenes dataset and version

data_root = 'data_set'
nusc_version = 'v1.0-mini'

# Set the scene and sample indices
scene_idx = 1
sample_idx = 1

# Set the point to be projected in the ego vehicle coordinate system
point = np.array([10, 0, 0])
print(point)

# Load the NuScenes dataset
nusc = NuScenes(version=nusc_version, dataroot=data_root)
my_scene = nusc.scene[scene_idx]
my_sample = nusc.sample[sample_idx]

sample_record = nusc.get('sample', my_sample['token'])

camera_token = sample_record['data']['CAM_FRONT']
cam = nusc.get('sample_data', camera_token)

pointsensor_token = sample_record['data']['RADAR_FRONT']
pointsensor = nusc.get('sample_data', pointsensor_token)

pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
pc = RadarPointCloud.from_file(pcl_path)
im = Image.open(osp.join(nusc.dataroot, cam['filename']))

cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
pc.translate(np.array(cs_record['translation']))

# Second step: transform from ego to the global frame.
poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
pc.translate(np.array(poserecord['translation']))

# Third step: transform from global into the ego vehicle frame for the timestamp of the image.
poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
pc.translate(-np.array(poserecord['translation']))
pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

# Fourth step: transform from ego into the camera.
cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
pc.translate(-np.array(cs_record['translation']))
pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)


# Fifth step: actually take a "picture" of the point cloud.
# Grab the depths (camera frame z axis points away from the camera).
depths = pc.points[2, :]
coloring = depths

#-------------Ectract plane coordinate as input of DBSCAN--------------
coordinate = np.transpose(pc.points[[0,2],:])
# Create and fit the HDBSCAN model
#hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=5)
#hdbscan_model.fit(coordinate)
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(coordinate)

# Get the cluster labels (-1 indicates an outlier)
#labels = hdbscan_model.labels_
labels = dbscan.labels_

# Plot the data points colored by their cluster label
plt.scatter(coordinate[:, 0], coordinate[:, 1], c=labels)
plt.show()



view = np.array(cs_record['camera_intrinsic'])
points = pc.points[:3, :]
assert view.shape[0] <= 4
assert view.shape[1] <= 4
assert points.shape[0] == 3

viewpad = np.eye(4)
viewpad[:view.shape[0], :view.shape[1]] = view

nbr_points = points.shape[1]

# Do operation in homogenous coordinates.
points = np.concatenate((points, np.ones((1, nbr_points))))
points = np.dot(viewpad, points)
points = points[:3, :]

if 1:
    points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

# Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
# Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
# casing for non-keyframes which are slightly out of sync.
mask = np.ones(depths.shape[0], dtype=bool)
min_dist = 1
mask = np.logical_and(mask, depths > min_dist)
mask = np.logical_and(mask, points[0, :] > 1)
mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
mask = np.logical_and(mask, points[1, :] > 1)
mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
points = points[:, mask] # this is the image x,y coordinate
coloring = coloring[mask]
cluster = np.empty(points.shape[1])

#------------------------------------------------------------------------------------------
det = detect()
xyxy = np.array(det)
xyxy = np.delete(xyxy,[4,5],1)

print(xyxy)
counter = 1
for objects in range(xyxy.shape[0]):
    for radar_point in range(points.shape[1]):
        print(points[0,radar_point])
        if int(points[0,radar_point])>=xyxy[objects][0] and int(points[0,radar_point]) <= xyxy[objects][2] and int(points[1,radar_point]) >= xyxy[objects][1] and int(points[1,radar_point]) <= xyxy[objects][3]:
            cluster[radar_point]=int(objects)
clustered_points=np.array(points)
clustered_points = np.append(clustered_points,[cluster],0)
#print(cluster)

image_path = nusc.get_sample_data_path(nusc.get('sample_data', my_sample['data']['CAM_FRONT'])['token'])
image = cv2.imread(image_path)
for i in range(points.shape[1]):
    cv2.circle(image, (int(points[0,i]), int(points[1,i])), radius=3, color=(0, 25*cluster[i], 0), thickness=-1)
cv2.imshow('Image with projected point', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#fig, ax = plt.subplots(1, 1, figsize=(9, 16))
#fig.canvas.set_window_title(my_sample)
#ax.imshow(im)
#ax.scatter(points[0, :], points[1, :], c=coloring, s=5)
#ax.axis('off')
#print(points)