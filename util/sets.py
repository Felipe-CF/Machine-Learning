import os, json
import pandas as pd
import matplotlib as plt
from util.preprocessing import *
from PIL import Image, UnidentifiedImageError
from keras_preprocessing.image import ImageDataGenerator


def create_sets(processing=None):
    data_gen = ImageDataGenerator( #objeto com regras para o pré-processamento de imagens
        rescale=1./255, 
        # augmentation
        shear_range=0.2, # distorção de inclinação
        zoom_range=0.2, # zoom in e out aleatorio
        horizontal_flip=True, # aleatorio
        vertical_flip=True, # aleatorio
        brightness_range=[0.2, 0.8],
        samplewise_std_normalization=True,
        validation_split=0.2, # separação do subset de Validation
    )

    file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if processing is None:
        dataframe_preprocessing()

    dataset_dir = file_dir + '\\db\\DataCrohnIPI_2021_03\\DataCrohnIPI\\'
    
    dataframe_path = os.path.join(dataset_dir, 'CrohnIPI_description_screening_processed.csv')

    dataframe = pd.read_csv(dataframe_path, sep=',', encoding='iso-8859-1')

    training_set = data_gen.flow_from_dataframe(
        directory= dataset_dir + '\\imgs',
        dataframe=dataframe,
        y_col=['0', '1'],
        x_col='2',
        subset='training',
        batch_size=16,
        shuffle=True,
        class_mode='raw',
        target_size=(320, 320)
    )

    validation_set = data_gen.flow_from_dataframe(
        directory= dataset_dir + '\\imgs',
        dataframe=dataframe,
        y_col=['0', '1'],
        x_col='2',
        target_size=(320, 320),
        batch_size=16,
        class_mode='raw',
        shuffle=True,
        subset='validation'
        )
    
    return training_set, validation_set