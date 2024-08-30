# svr_train_and_test.py

from Src.svr_model import load_data_and_labels, train_svr_model

if __name__ == "__main__":
    image_paths, labels = load_data_and_labels()
    
    train_svr_model(image_paths, labels)
