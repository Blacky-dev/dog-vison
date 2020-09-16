import streamlit as st
import keras
from keras import layers
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import tempfile
import os
def teachable_machine_classification():
    # Load the model
    weights_file=r'C:\Users\Black\OneDrive\Desktop\Dog_ai_webapp\20200911-121337-10000-images-mobilenet-v2-Adam_optimizer.h5'
    model = tf.keras.models.load_model(weights_file,
                                   custom_objects={'KerasLayer':hub.KerasLayer})
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 512, 512, 3), dtype=np.float32)
    image = Image.open(r'C:\Users\Black\OneDrive\Desktop\German_Shepherd_-_DSC_0346_(10096362833)')
    #image sizing
    size = (512, 512)
    image = ImageOps.fit(image, size)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255)

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    prediction=np.argmax(prediction)
    result=(np.max(prediction))
teachable_machine_classification()