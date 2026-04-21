import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
import os

def predict_single_image(img_path, model_path='../models/brain_tumor_model.h5'):
    """Predicts presence of brain tumor from an input image path using a trained model."""
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'. Please run train.py first to generate the model.")
        return
    
    if not os.path.exists(img_path):
        print(f"Error: Image not found at '{img_path}'.")
        return

    # Load the trained model
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Class labels mapping 
    # Must match the categorical encoding done by ImageDataGenerator during training
    # Alphabetical order: glioma, meningioma, notumor, pituitary
    class_labels = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}
    
    # Preprocess image
    img_width, img_height = 224, 224
    try:
        img = image.load_img(img_path, target_size=(img_width, img_height))
    except Exception as e:
        print(f"Error loading image: {e}")
        return
        
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array /= 255.0 # Rescale to [0,1]
    
    # Predict
    print("Running inference...")
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    
    predicted_label = class_labels.get(predicted_class_idx, "Unknown")
    
    has_tumor = predicted_label != 'notumor'
    
    print("\n" + "="*40)
    print("       PREDICTION RESULTS       ")
    print("="*40)
    print(f"Input Image     : {img_path}")
    print(f"Predicted Class : {predicted_label.capitalize()}")
    print(f"Tumor Detected  : {'YES' if has_tumor else 'NO'}")
    print(f"Confidence      : {confidence:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        print("Example: python predict.py ../data/Testing/glioma/image.jpg")
        sys.exit(1)
        
    image_path_arg = sys.argv[1]
    predict_single_image(image_path_arg)
