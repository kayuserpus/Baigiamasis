# src/model_saving.py

import joblib
import tensorflow as tf

def save_model(model, filename):
    """Save a machine learning model using joblib."""
    joblib.dump(model, filename)

def load_model(filename):
    """Load a machine learning model using joblib."""
    return joblib.load(filename)

def save_cnn_model(model, filename):
    """Save a CNN model using TensorFlow/Keras."""
    model.save(filename)

def load_cnn_model(filename):
    """Load a CNN model using TensorFlow/Keras."""
    return tf.keras.models.load_model(filename)
