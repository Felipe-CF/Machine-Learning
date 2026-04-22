import os, json, random
import pandas as pd
from util.preprocessing import *
from keras_preprocessing.image import ImageDataGenerator


DATASET_DIR_IMG = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '\\db\\DataCrohnIPI_2021_03\\DataCrohnIPI\\'


def create_sets(path_folds, fold_test):
    training_df, validation_fold =  None, None

    with open(path_folds, 'r', encoding='utf-8') as file:
        kfolds = json.loads(file.read())

    kfolds[f'fold{fold_test}']['validation'] = True

    new_fold_validation = kfolds[f'fold{fold_test}']

    with open(path_folds, 'w', encoding='utf-8') as file:
        json.dump(kfolds, file, ensure_ascii=False, indent=4)

    validation_fold = pd.read_json(kfolds[f'fold{fold_test}']['fold'])

    check = f'fold{fold_test}'

    training_df = pd.concat([pd.read_json(fold['fold']) for key, fold in kfolds.items() if key != check])

    kfolds = None

    fold_test += 1

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

    training_set = data_gen.flow_from_dataframe(
        directory= DATASET_DIR_IMG + '\\imgs',
        dataframe=training_df,
        y_col=[0, 1],
        x_col=2,
        batch_size=16,
        shuffle=True,
        class_mode='raw',
        target_size=(320, 320)
    )

    validation_set = data_gen.flow_from_dataframe(
        directory= DATASET_DIR_IMG + '\\imgs',
        dataframe=validation_fold,
        y_col=[0, 1],
        x_col=2,
        target_size=(320, 320),
        batch_size=16,
        class_mode='raw',
        shuffle=True,
        )

    return training_set, validation_set, fold_test



