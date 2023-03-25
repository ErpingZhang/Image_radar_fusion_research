import torch
import cv2
from PIL import Image
from extractimage import extract_img_path
from corp import img_crop
# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# Load the image
img_path = extract_img_path()

# Crop the image and save it as "vehicle.jpg"
img_crop(img_path, 1600)

# Read vehicle.jpg into img
img = cv2.imread("vehicle.jpg")

# Convert image to RGB format
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create a PyTorch tensor from the image
img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

# Run inference on the image
results = model(img_tensor)

# Get the detected vehicle class indices
vehicle_class_indices = [2, 5, 7]

# Loop through each detected object and draw a bounding box if it's a vehicle
for detection in results.data[0]:
    if int(detection[5]) in vehicle_class_indices:
        box = detection[0:4].int()
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

# Show the image with bounding boxes drawn around the detected vehicles
img = Image.fromarray(img)
img.show()
