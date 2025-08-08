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


def invalid_files(dir_path, valid_extensions={'.jpg', '.png', '.jpeg'}):
        list_dir = []

        list_dir.append(dir_path + '\\cats')

        list_dir.append(dir_path + '\\dogs')

        for dir in list_dir:
            files = os.listdir(dir)

            for file_name in files:
                file_path = os.path.join(dir, file_name)

                extension = os.path.splitext(file_name)

                extension = extension[1].lower()

                if extension not in valid_extensions:
                    os.remove(file_path)
                    print(f'[REMOVED] corrupted image: {file_path}')
            
            for file_name in files:
                try:
                    file_path = os.path.join(dir, file_name)

                    with Image.open(file_path) as img:
                        img.verify()

                except (UnidentifiedImageError, IOError,OSError, SyntaxError):
                    print(f'[REMOVED] corrupted image: {file_path}')

                    os.remove(file_path)
                        

def create_conv_net():
    conv_net = Sequential()

    conv_net.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))

    conv_net.add(layers.LeakyReLU(alpha=0.01))

    conv_net.add(MaxPooling2D(pool_size=(2, 2)))

    conv_net.add(Conv2D(32, (3, 3)))
    
    conv_net.add(layers.LeakyReLU(alpha=0.01))

    conv_net.add(MaxPooling2D(pool_size=(2, 2)))

    conv_net.add(Flatten())

    conv_net.add(Dense(units=128, activation=layers.LeakyReLU(alpha=0.01)))

    conv_net.add(Dense(units=7, activation='softmax'))

    conv_net.compile(
        # optimizer='adam', 
        # optimizer=optimizer, 
        optimizer="SGD", 
        loss='binary_crossentropy', 
        metrics=['accuracy']
        )

    return conv_net


def create_sets():

    data_gen = ImageDataGenerator( #objeto com regras para o pré-processamento de imagens
        rescale=1./255, 
        # augmentation
        shear_range=0.2, # distorção de inclinação
        zoom_range=0.2, # zoom in e out aleatorio
        horizontal_flip=True, # aleatorio
        validation_split=0.2, # separação do subset de validação
    )

    dataframe = pd.read_csv(
        'C:\\Users\\FelipeCF\\Desktop\\Codigos\\Machine-Learning\\DataCrohnIPI_2021_03\\DataCrohnIPI\\CrohnIPI_description.csv',
        sep=',',
        encoding='iso-8859-1'
        )

    training_set = data_gen.flow_from_dataframe(
        directory='C:\\Users\FelipeCF\\Desktop\\Codigos\\Machine-Learning\\DataCrohnIPI_2021_03\\DataCrohnIPI\\imgs',
        dataframe=dataframe,
        x_col='Frame name',
        y_col='Label',
        subset='training',
        batch_size=32,
        shuffle=True,
        class_mode='categorical',
        target_size=(64, 64)
    )

    validation_set = data_gen.flow_from_dataframe(
        directory='C:\\Users\FelipeCF\\Desktop\\Codigos\\Machine-Learning\\DataCrohnIPI_2021_03\\DataCrohnIPI\\imgs',
        dataframe=dataframe,
        x_col='Frame name',
        y_col='Label',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        subset='validation'
        )
    
    return training_set, validation_set


def prediction(conv_net, image_name):
    test_image = image.load_img(
        os.path.join('datasets\\cnn\\test', 
                     image_name), target_size=(64, 64))
    
    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis=0)

    result = conv_net.predict(test_image)

    training_set.class_indices
    
    pass


if __name__ == '__main__':
    conv_net = create_conv_net()

    file_dir = os.path.dirname(os.path.abspath(__file__))
    
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

    # best_model_path = os.path.join(checkpoint_dir, 'conv_net_accuracy_0.84.keras')

    # new_net = keras.saving.load_model(best_model_path, compile=True, safe_mode=True, custom_objects=None)

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





