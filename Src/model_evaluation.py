# src/model_evaluation.py

from Src.model_saving import load_model, load_cnn_model
from Src.model_inference import predict_with_svr, predict_with_cnn
from app.models import GroundTruth, ImageData
from sklearn.metrics import accuracy_score, precision_score, recall_score
from app import db

def evaluate_svr_model(model_path='data/models/svr_model.pkl'):
    """Evaluate the SVR model's performance against ground truth."""
    model = load_model(model_path)
    
    # Retrieve test data
    test_data = ImageData.query.filter_by(folder_name='test').all()
    
    y_true = []
    y_pred = []
    
    for data in test_data:
        image_path = f'data/raw/Final_Test_Images/{data.filename}'
        predicted_class = predict_with_svr(image_path, model_path)
        
        # Compare with ground truth
        ground_truth = GroundTruth.query.filter_by(filename=data.filename).first()
        if ground_truth:
            y_true.append(ground_truth.ground_truth_class_id)
            y_pred.append(int(predicted_class[0]))
    
        # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    
    print(f'SVR Model Accuracy: {accuracy * 100:.2f}%')
    print(f'SVR Model Precision: {precision * 100:.2f}%')
    print(f'SVR Model Recall: {recall * 100:.2f}%')
    
    return accuracy, precision, recall

def evaluate_cnn_model(model_path='data/models/cnn_model.h5'):
    """Evaluate the CNN model's performance against ground truth."""
    model = load_cnn_model(model_path)
    
    # Retrieve test data
    test_data = ImageData.query.filter_by(folder_name='test').all()
    
    y_true = []
    y_pred = []
    
    for data in test_data:
        image_path = f'data/raw/Final_Test_Images/{data.filename}'
        predicted_class = predict_with_cnn(image_path, model_path)
        
        # Compare with ground truth
        ground_truth = GroundTruth.query.filter_by(filename=data.filename).first()
        if ground_truth:
            y_true.append(ground_truth.ground_truth_class_id)
            y_pred.append(int(predicted_class[0]))
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    
    print(f'CNN Model Accuracy: {accuracy * 100:.2f}%')
    print(f'CNN Model Precision: {precision * 100:.2f}%')
    print(f'CNN Model Recall: {recall * 100:.2f}%')
    
    return accuracy, precision, recall

