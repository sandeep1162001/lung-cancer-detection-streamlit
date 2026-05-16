# Lung Cancer Detection Using Deep Learning

A Deep Learning-based web application for detecting lung cancer from CT scan images using TensorFlow/Keras, Transfer Learning, and Streamlit.

This project classifies lung CT scan images into:

- Begin
- Malignant
- Normal

The application uses a fine-tuned VGG16 Transfer Learning model integrated with a Streamlit web application for real-time prediction and visualization.

---

# Project Highlights

- Deep Learning-based medical image classification
- Transfer Learning using VGG16
- Fine-tuning pretrained models
- Image augmentation techniques
- Streamlit-based interactive web application
- Confidence score prediction
- Probability visualization
- Modular deployment-ready architecture

---

# Models Implemented

## 1. Custom CNN
A custom Convolutional Neural Network built using:

- Conv2D
- MaxPooling
- BatchNormalization
- Dropout
- Dense layers

---

## 2. InceptionV3
Transfer Learning model using pretrained ImageNet weights.

Features:
- Frozen pretrained layers
- Fine-tuning
- Improved feature extraction
- Better generalization on medical images

---
## 3. VGG16
Transfer Learning using the VGG16 architecture.

Features:
- Deep convolutional architecture
- Fine-tuning of deeper layers
- Enhanced image representation learning

---

# Model Architecture

## VGG16 Transfer Learning

This project uses the VGG16 pretrained convolutional neural network with ImageNet weights.

### Features:
- Pretrained ImageNet weights
- Frozen convolutional base layers
- Fine-tuning of deeper layers
- Improved feature extraction
- Better generalization on medical imaging datasets

### Custom Layers Added:
- Flatten Layer
- Dense Layer (1024 neurons)
- Dropout Layer
- Softmax Output Layer

---

# Dataset

Dataset Used:

## IQ-OTHNCCD Lung Cancer Dataset

The dataset contains CT scan images categorized into:

- Benign cases
- Malignant cases
- Normal cases

---

# Tech Stack

## Languages
- Python

## Deep Learning Frameworks
- TensorFlow
- Keras

## Libraries
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Plotly
- Scikit-learn

## Frontend
- Streamlit

---

# Project Structure

```text
Lungs cancer/
│
├── app/
│   ├── app.py
│   ├── utils.py
│
├── models/
│   ├── VGG16.keras
│   └── InceptionV3.keras
│
├── requirements.txt
├── README.md
└── .gitignore