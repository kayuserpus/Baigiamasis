import cv2
from skimage.feature import hog
import numpy as np

def preprocess_image(image_path):
    """Load and preprocess an image."""
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (64, 64))  # Standard size for consistency
    return resized_image

def extract_hog_features(image):
    """Extract HOG features from an image."""
    hog_features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, block_norm='L2-Hys')
    return hog_features
