# src/svr_model.py

from sklearn.svm import SVR
from Src.data_preprocessing import preprocess_image, extract_hog_features
from Src.model_saving import save_model
import os
import csv
import numpy as np

def load_data_and_labels():
    """Load image paths and labels from the training data."""
    image_paths = []
    labels = []
    
    for folder in os.listdir('data/raw/Final_Training_Images'):
        csv_path = f'data/raw/Final_Training_Images/{folder}/GT-{folder}.csv'
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                image_file = os.path.join(f'data/raw/Final_Training_Images/{folder}', row['Filename'])
                image_paths.append(image_file)
                labels.append(int(row['ClassId']))
    
    return image_paths, labels

def train_svr_model(image_paths, labels, model_save_path='data/models/svr_model.pkl'):
    """Train an SVR model on HOG features extracted from images."""
    
    hog_features_list = []
    for image_path in image_paths:
        image = preprocess_image(image_path)
        hog_features = extract_hog_features(image)
        hog_features_list.append(hog_features)
    
    hog_features = np.array(hog_features_list)
    
    svr_model = SVR(kernel='linear')
    svr_model.fit(hog_features, labels)
    
    save_model(svr_model, model_save_path)
    print(f"SVR model saved to {model_save_path}")
    
    return svr_model
