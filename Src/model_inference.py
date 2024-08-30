from Src.model_saving import load_model, load_cnn_model
from Src.data_preprocessing import preprocess_image, extract_hog_features
import numpy as np
import cv2

def predict_with_svr(image_path, model_path='data/models/svr_model.pkl'):
    """Predict the class of an image using a trained SVR model."""
    model = load_model(model_path)
    
    image = preprocess_image(image_path)
    hog_features = extract_hog_features(image)
    
    prediction = model.predict([hog_features])
    return prediction

def predict_with_cnn(image_path, model_path='data/models/cnn_model.h5'):
    """Predict the class of an image using a trained CNN model."""
    model = load_cnn_model(model_path)
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))
    image = image.reshape(1, 64, 64, 1)  # Reshape for CNN input

    prediction = model.predict(image)
    return np.argmax(prediction, axis=1)
