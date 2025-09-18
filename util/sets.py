import os, json, random
import pandas as pd
import matplotlib as plt
from util.preprocessing import *
from keras_preprocessing.image import ImageDataGenerator


def create_sets(kfolds):
    training_df, validation_fold =  None, None

    fold_test_n = 1

    for i, fold in enumerate(kfolds):

        if fold['test'] is False:
            fold['test'] = True

            validation_fold = kfolds.pop(i)

            fold_test_n = validation_fold['fold_n']

            break
    
    training_df = pd.concat([fold['fold'] for fold in kfolds])

    validation_df = validation_fold['fold']

    kfolds.append(validation_fold)

    #objeto com regras para o pré-processamento de imagens
    data_gen = ImageDataGenerator( 
        rescale=1./255, 
        # augmentation
        shear_range=0.2, # distorção de inclinação
        zoom_range=0.2, # zoom in e out aleatorio
        horizontal_flip=True, # aleatorio
        vertical_flip=True, # aleatorio
        rotation_range=90,
        brightness_range=[0.2, 0.8],
        samplewise_std_normalization=True,
    )

    file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    dataset_dir = file_dir + '\\db\\DataCrohnIPI_2021_03\\DataCrohnIPI\\'

    training_set = data_gen.flow_from_dataframe(
        directory= dataset_dir + '\\imgs',
        dataframe=training_df,
        y_col=[0, 1],
        x_col=2,
        batch_size=16,
        shuffle=True,
        class_mode='raw',
        target_size=(320, 320)
    )

    validation_set = data_gen.flow_from_dataframe(
        directory= dataset_dir + '\\imgs',
        dataframe=validation_df,
        y_col=[0, 1],
        x_col=2,
        target_size=(320, 320),
        batch_size=16,
        class_mode='raw',
        shuffle=True,
        )
    
    return training_set, validation_set, fold_test_n



