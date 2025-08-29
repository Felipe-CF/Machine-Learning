import keras
from util import *
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
from keras.losses import CategoricalCrossentropy
from keras.optimizers import SGD
from keras.layers import Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dense, Dropout, Activation



def create_load_net(file_dir=None):
    conv_net = None
    
    if not file_dir:
        conv_net = Sequential()

        conv_net.add(Conv2D(32, (3, 3), input_shape=(320, 320, 3)))

        conv_net.add(layers.LeakyReLU(alpha=0.01))

        conv_net.add(MaxPooling2D(pool_size=(2, 2)))

        conv_net.add(Conv2D(32, (3, 3)))
        
        conv_net.add(layers.LeakyReLU(alpha=0.01))

        conv_net.add(MaxPooling2D(pool_size=(2, 2)))

        conv_net.add(Conv2D(32, (3, 3)))
        
        conv_net.add(layers.LeakyReLU(alpha=0.01))

        conv_net.add(MaxPooling2D(pool_size=(2, 2)))

        conv_net.add(Conv2D(32, (3, 3)))
        
        conv_net.add(layers.LeakyReLU(alpha=0.01))

        conv_net.add(MaxPooling2D(pool_size=(2, 2)))

        conv_net.add(Conv2D(32, (3, 3)))
        
        conv_net.add(layers.LeakyReLU(alpha=0.01))

        conv_net.add(MaxPooling2D(pool_size=(2, 2)))

        conv_net.add(Conv2D(32, (3, 3)))
        
        conv_net.add(layers.LeakyReLU(alpha=0.01))

        conv_net.add(MaxPooling2D(pool_size=(2, 2)))

        conv_net.add(Conv2D(32, (3, 3)))
        
        conv_net.add(layers.LeakyReLU(alpha=0.01))

        conv_net.add(MaxPooling2D(pool_size=(2, 2)))

        conv_net.add(Conv2D(32, (3, 3)))
        
        conv_net.add(layers.LeakyReLU(alpha=0.01))

        conv_net.add(MaxPooling2D(pool_size=(2, 2)))

        conv_net.add(Conv2D(32, (3, 3)))
        
        conv_net.add(layers.LeakyReLU(alpha=0.01))

        conv_net.add(MaxPooling2D(pool_size=(2, 2)))

        conv_net.add(Conv2D(32, (3, 3)))
        
        conv_net.add(layers.LeakyReLU(alpha=0.01))

        conv_net.add(MaxPooling2D(pool_size=(2, 2)))

        conv_net.add(Conv2D(32, (3, 3)))
        
        conv_net.add(layers.LeakyReLU(alpha=0.01))

        conv_net.add(MaxPooling2D(pool_size=(2, 2)))

        conv_net.add(Conv2D(32, (3, 3)))
        
        conv_net.add(layers.LeakyReLU(alpha=0.01))

        conv_net.add(MaxPooling2D(pool_size=(2, 2)))

        conv_net.add(Conv2D(32, (3, 3)))
        
        conv_net.add(layers.LeakyReLU(alpha=0.01))

        conv_net.add(MaxPooling2D(pool_size=(2, 2)))

        conv_net.add(Conv2D(32, (3, 3)))
        
        conv_net.add(layers.LeakyReLU(alpha=0.01))

        conv_net.add(MaxPooling2D(pool_size=(2, 2)))

        conv_net.add(Conv2D(32, (3, 3)))
        
        conv_net.add(layers.LeakyReLU(alpha=0.01))

        conv_net.add(MaxPooling2D(pool_size=(2, 2)))

        conv_net.add(Flatten())

        conv_net.add(Dense(units=128, activation=layers.LeakyReLU(alpha=0.01)))

        conv_net.add(Dense(units=7, activation='softmax'))

        conv_net.compile(
            optimizer=SGD(momentum=0.99), 
            loss=CategoricalCrossentropy(), 
            metrics=['accuracy']
            )
        
    else:
        checkpoint_dir = os.path.join(file_dir, 'model_checkpoints')

        best_model_path = os.path.join(checkpoint_dir, 'conv_net_accuracy_0.69.keras')

        conv_net = keras.saving.load_model(best_model_path, compile=True, safe_mode=True, custom_objects=None)
    
    return conv_net



if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))

    conv_net = create_load_net(file_dir)

    training_set, validation_set = create_sets()

    checkpoint_dir = os.path.join(file_dir, 'model_checkpoints')

    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'conv_net_accuracy_{val_accuracy:.2f}.keras'),
        mode='max', # adequado para val_accuracy
        save_best_only=True, # salvar quando a métrica melhora
        save_weights_only=False, # somente os pesos
        monitor='val_accuracy', # métrica balizadora do armazenamento (precisão da validação)
        verbose=1 # logs de salvamento
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.1,
        patience=10,
        mode='auto',
        start_from_epoch=30,
        restore_best_weights=True,
    )

    conv_net.fit(
        training_set, 
        steps_per_epoch=87, 
        epochs=100,
        validation_data=validation_set,
        validation_steps=20, 
        callbacks=[model_checkpoint, early_stop]
    )





