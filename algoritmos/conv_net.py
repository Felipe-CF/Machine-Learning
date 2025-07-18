import tensorflow as tf
import os
from pathlib import Path
from activations import *
import numpy as np, random
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator

# objeto ConvNet iniciado
conv_net = Sequential()

# 1° camada conv
conv_net.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# 1° camada pool
conv_net.add(MaxPooling2D(pool_size=(2, 2)))

# 2° camada conv
conv_net.add(Conv2D(32, (3, 3), activation='relu'))

# 2° camada pool
conv_net.add(MaxPooling2D(pool_size=(2, 2)))

# flattening
conv_net.add(Flatten())

#full connection
conv_net.add(Dense(units=128, activation='relu'))

conv_net.add(Dense(units=1, activation='sigmoid'))

#compilando a rede
conv_net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#objeto com regras para o pré-processamento de imagens
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'algoritmos\\datasets\\cnn\\train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
    )

validation_set = validation_datagen.flow_from_directory(
    'algoritmos\\datasets\\cnn\\validation',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
    )

conv_net.fit(
    training_set, 
    steps_per_epoch=8000,
    epochs=5,
    validation_data=validation_set,
    validation_steps=2000
    )



