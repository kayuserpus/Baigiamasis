# cnn_train_and_test.py

from Src.cnn_model import load_data_and_labels, train_cnn_model
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    images, labels, num_classes = load_data_and_labels()
    
    train_data, val_data, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    train_cnn_model(train_data, train_labels, val_data, val_labels)
