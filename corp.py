from PIL import Image

def img_crop(img_path, crop_size):
    # Open the image using PIL
    img = Image.open(img_path)

    # Get the width and height of the image
    width, height = img.size

    # Define the size of the cropped image
    #crop_size = 128

    # Calculate the center point of the image
    center_x = width // 2
    center_y = height // 2

    # Calculate the coordinates of the crop
    x1 = center_x - crop_size // 2
    y1 = center_y - crop_size // 2
    x2 = center_x + crop_size // 2
    y2 = center_y + crop_size // 2
    box = (x1, y1, x2, y2)

    # Crop the image
    crop = img.crop(box)

    # Save the crop with a unique filename
    crop.save('vehicle.jpg')

