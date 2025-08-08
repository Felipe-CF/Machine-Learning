import os, keras
import tensorflow as tf
import pandas as pd
from pathlib import Path
from keras import layers
import numpy as np, random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.preprocessing import image
from keras import regularizers, optimizers
from keras.callbacks import ModelCheckpoint
from PIL import Image, UnidentifiedImageError
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dense, Dropout, Activation


def test():
    pass

if __name__ == '__main__':
    dataframe = pd.read_csv(
        'C:\Users\FelipeCF\Desktop\Codigos\Machine-Learning\DataCrohnIPI_2021_03\DataCrohnIPI\CrohnIPI_description.csv',
        sep=',', 
        encoding='iso-8859-1', dtype=str
        )
    
    # dataframe = pd.DataFrame.copy(dataframe)




