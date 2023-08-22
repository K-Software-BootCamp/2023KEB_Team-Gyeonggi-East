import os
import copy  

import tensorflow as tf 
from tensorflow import keras

import cv2

from PIL import Image

import numpy as np



def ready():
    global new_model
    new_model= tf.keras.models.load_model('C:\\Users\\Lenovo\\Desktop\\pythoncodecode\\model.h5')
    return 'ready'

def classification(getimage):
    getimage = np.array(getimage)
    images = np.stack((getimage,)*3, axis=-1)
    images = cv2.resize(images.astype("float32"),(600, 600))
    images = images.astype("float32") / 255
    images = np.expand_dims(images,axis=0)
    
    predictions = np.array(new_model.predict([images]))

    predictions = predictions.astype("float64")


    
    return predictions[0]*100

