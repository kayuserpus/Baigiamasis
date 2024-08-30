from sklearn.svm import SVR
from Src.data_preprocessing import preprocess_image, extract_hog_features
from Src.model_saving import save_model
import os
import numpy as np

def train_svr_model(image_paths, labels, model_save_path='data/models/svr_model.pkl'):
    """Train an SVR model on HOG features extracted from images."""
    
    # Extract HOG features for each image
    hog_features_list = []
    for image_path in image_paths:
        image = preprocess_image(image_path)
        hog_features = extract_hog_features(image)
        hog_features_list.append(hog_features)
    
    hog_features = np.array(hog_features_list)
    
    # Train the SVR model
    svr_model = SVR(kernel='linear')
    svr_model.fit(hog_features, labels)
    
    # Save the trained model
    save_model(svr_model, model_save_path)
    print(f"SVR model saved to {model_save_path}")

    return svr_model
