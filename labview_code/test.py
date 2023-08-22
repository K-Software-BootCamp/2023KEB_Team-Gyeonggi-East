import numpy as np
import os
import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
import copy  
import argparse
import cv2
from PIL import Image

def test(getarray):
    getarray = np.array(getarray)
    getarray = np.stack((getarray,)*3, axis=-1)
    return str(getarray.shape)

