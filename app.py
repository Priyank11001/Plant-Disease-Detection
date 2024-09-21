import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_model('./Model')
class_names = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']

IMAGE_SIZE = (256, 256)

def predict(img):
    # Convert the PIL image to a format the model can process
    img_array = img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0)  

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]  

    return predicted_class

# Define Gradio interface
title = "Plant Disease Detection"
description = "Upload a plant leaf image to detect the disease using the PlantVillage dataset."


interface = gr.Interface(fn=predict, 
                         inputs=gr.Image(type="pil"),  
                         outputs="text",                      
                         title=title,
                         description=description)

# Launch the app
if __name__ == "__main__":
    interface.launch()
