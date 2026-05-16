import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model


# Load Trained Model
MODEL_PATH = "models\VGG16.keras"
model = load_model(MODEL_PATH)


# Class Labels
class_names = ['Begin', 'Malignant', 'Normal']


# Image Size
IMG_WIDTH = 224
IMG_HEIGHT = 224


# Preprocess Image Function
def preprocess_image(image):

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    # Convert to float32
    image = image.astype(np.float32)

    # Normalize
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image



# Prediction Function
def predict_image(image):

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    confidence = float(np.max(prediction)) * 100

    probabilities = {
        class_names[i]: float(prediction[0][i]) * 100
        for i in range(len(class_names))
    }

    return predicted_class, confidence, probabilities