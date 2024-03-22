import cv2
import numpy as np

def rotate_and_flip_images(image_path):
    # Read the original image
    original_image = cv2.imread(image_path)

    # Create three modified images: rotated 90 degrees, rotated 180 degrees, and flipped vertically
    rotated_90 = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)
    rotated_180 = cv2.rotate(original_image, cv2.ROTATE_180)
    flipped_vertical = cv2.flip(original_image, 0)  # 0 means flipping around the x-axis

    return rotated_90, rotated_180, flipped_vertical

# Function to generate patches from an image
def split_image(image_path):
    # Load the original image
    original_image = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if original_image is None:
        print("Error: Could not load the image.")
        return None

    # Get the dimensions of the original image
    height, width, _ = original_image.shape

    # Check if the dimensions are suitable for splitting
    if height != 460 or width != 700:
        print("Error: Input image dimensions must be 700x460.")
        return None

    # Define the size of each sub-image
    x_size = 224
    y_size = 224

    # Split the original image into 6 sub-images
    sub_images = []
    for i in range(0, 2):
        for j in range(0, 3):
            start_row = i * x_size
            end_row = start_row + x_size
            start_col = j * y_size
            end_col = start_col + y_size

            sub_image = original_image[start_row:end_row, start_col:end_col]
            sub_images.append(sub_image)

    return sub_images