import cv2
import os

# Path to the directory containing the images
directory_path = '.'

# Create a new directory for binarized images
binarized_directory = os.path.join(directory_path, 'binarized')
os.makedirs(binarized_directory, exist_ok=True)

# Function to binarize an image using OpenCV
def binarize_image(image_path, threshold=150):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    _, binarized_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binarized_image

# Binarize each image in the directory and save the result in the new directory
for filename in os.listdir(directory_path):
    if filename.endswith('.png'):
        image_path = os.path.join(directory_path, filename)
        binarized_image = binarize_image(image_path)
        cv2.imwrite(os.path.join(binarized_directory, filename), binarized_image)

print("Binarization complete!")
