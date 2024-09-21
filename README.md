---
title: Plant Disease Detection
emoji: 🚀
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---
# 🌿 Plant Disease Detection

This project implements a machine learning model for detecting diseases in plant leaves using the PlantVillage dataset. The model was trained multiple times to achieve high accuracy and provides a user-friendly interface via **Gradio**, which has been deployed on **Hugging Face Spaces**.

## 🚀 Overview

Plants are susceptible to numerous diseases that can cause significant losses in agriculture. This project utilizes deep learning techniques to classify and detect diseases in plant leaves, helping farmers and agriculturists to take timely actions. The model is capable of recognizing 15 different classes of plant diseases.

### Key Features:
- **15 Classes of Plant Diseases**: Trained using the PlantVillage dataset.
- **Accuracy**: Achieved a test accuracy of **98.70%**.
- **UI with Gradio**: An interactive user interface to upload plant leaf images and get real-time disease predictions.
- **Deployment**: The model is deployed on **Hugging Face Spaces**.

## 📊 Dataset

The project uses the **PlantVillage Dataset**, which contains approximately **20,000 images** of plant leaves classified into **15 different categories** of diseases, including healthy plants.

### Dataset Details:
- **Classes (15)**:
  - `Pepper__bell___Bacterial_spot`
  - `Pepper__bell___healthy`
  - `Potato___Early_blight`
  - `Potato___Late_blight`
  - `Potato___healthy`
  - `Tomato_Bacterial_spot`
  - `Tomato_Early_blight`
  - `Tomato_Late_blight`
  - `Tomato_Leaf_Mold`
  - `Tomato_Septoria_leaf_spot`
  - `Tomato_Spider_mites_Two_spotted_spider_mite`
  - `Tomato__Target_Spot`
  - `Tomato__Tomato_YellowLeaf__Curl_Virus`
  - `Tomato__Tomato_mosaic_virus`
  - `Tomato_healthy`

Each class contains leaf images labeled with the corresponding disease or health condition. The images were preprocessed and augmented before being fed into the model.

## 🧠 Model

The model used for this project is a **Convolutional Neural Network (CNN)** trained with the PlantVillage dataset. The model was saved multiple times during the training process using `model.save()` to preserve checkpoints and ensure optimal performance.

- **Framework**: TensorFlow/Keras
- **Image Size**: 256x256 pixels
- **Training Accuracy**: ~98.50%
- **Testing Accuracy**: **98.70%**

### Training Process:

- **Data Augmentation**: Applied techniques like rotation, flipping, and zooming to increase dataset variability.
- **Optimizer**: Adam optimizer was used for training.
- **Loss Function**: Categorical Cross-Entropy was used due to the multi-class nature of the classification problem.

The model has been trained and validated using multiple epochs and achieved high accuracy.

## 🎛️ User Interface

We have used **Gradio** to create a simple and intuitive user interface that allows users to upload plant leaf images and get predictions about the disease or health condition.

### How to Use:
1. Upload an image of a plant leaf.
2. The model will analyze the image and display the predicted class (disease or healthy).
3. The interface will show the result along with a confidence score.

You can interact with the model live on **Hugging Face Spaces**: [Plant Disease Detection on Hugging Face](#).

## 🛠️ Deployment

The trained model and the Gradio UI have been deployed on **Hugging Face Spaces**. The deployment allows users to directly interact with the model and predict diseases in plant leaves.

### Run Locally

If you'd like to run this project locally:
https://huggingface.co/spaces/username/Plant-Disease-Detection


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
