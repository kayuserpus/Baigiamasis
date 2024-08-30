# src/cnn_model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import cv2
import csv

def create_cnn_model(input_shape, num_classes):
    """Create and compile a CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_data_and_labels():
    """Load and preprocess image data and labels."""
    images = []
    labels = []
    
    for folder in os.listdir('data/raw/Final_Training_Images'):
        csv_path = f'data/raw/Final_Training_Images/{folder}/GT-{folder}.csv'
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                image_file = os.path.join(f'data/raw/Final_Training_Images/{folder}', row['Filename'])
                image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (64, 64))
                images.append(image)
                labels.append(int(row['ClassId']))
    
    images = np.array(images).reshape(-1, 64, 64, 1)  # Reshape for CNN input
    labels = np.array(labels)
    
    num_classes = len(np.unique(labels))
    labels = to_categorical(labels, num_classes)
    
    return images, labels, num_classes

def train_cnn_model(train_data, train_labels, val_data, val_labels, model_save_path='data/models/cnn_model.h5', epochs=10, batch_size=32):
    """Train and save the CNN model."""
    input_shape = train_data.shape[1:]
    num_classes = train_labels.shape[1]

    cnn_model = create_cnn_model(input_shape, num_classes)
    cnn_model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=epochs, batch_size=batch_size)
    
    cnn_model.save(model_save_path)
    print(f"CNN model saved to {model_save_path}")
    
    return cnn_model
