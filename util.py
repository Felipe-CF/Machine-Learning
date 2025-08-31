import os
import pandas as pd
import keras
from PIL import Image, UnidentifiedImageError
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold


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


def create_sets():
    data_gen = ImageDataGenerator( #objeto com regras para o pré-processamento de imagens
        rescale=1./255, 
        # augmentation
        shear_range=0.2, # distorção de inclinação
        zoom_range=0.2, # zoom in e out aleatorio
        horizontal_flip=True, # aleatorio
        validation_split=0.2, # separação do subset de validação
    )

    file_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_dir = file_dir + '\\DataCrohnIPI_2021_03\\DataCrohnIPI\\'

    dataframe_path = os.path.join(dataset_dir, 'CrohnIPI_description_processed.csv')

    dataframe = pd.read_csv(dataframe_path, sep=',', encoding='iso-8859-1')

    training_set = data_gen.flow_from_dataframe(
        directory= dataset_dir + '\\imgs',
        dataframe=dataframe,
        y_col=['0', '1', '2', '3', '4', '5', '6'],
        x_col='7',
        subset='training',
        batch_size=16,
        shuffle=True,
        class_mode='raw',
        target_size=(320, 320)
    )

    validation_set = data_gen.flow_from_dataframe(
        directory= dataset_dir + '\\imgs',
        dataframe=dataframe,
        y_col=['0', '1', '2', '3', '4', '5', '6'],
        x_col='7',
        target_size=(320, 320),
        batch_size=16,
        class_mode='raw',
        shuffle=True,
        subset='validation'
        )
    
    return training_set, validation_set


def kfolds_subsets():
    pass


def dataframe_preprocessing(file_dir):
    file_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_dir = file_dir + '\\DataCrohnIPI_2021_03\\DataCrohnIPI'

    dataframe_path = os.path.join(dataset_dir, 'CrohnIPI_description.csv')

    dataframe = pd.read_csv(dataframe_path, sep=',', encoding='iso-8859-1')

    dataframe = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1])], remainder='passthrough').fit_transform(dataframe)

    dataframe = pd.DataFrame(dataframe)


    '''
    N ==> column=2
    U>10 ==> column=6
    U3-10 ==> column=5 
    E ==> column=1 
    AU ==> column=0 
    O ==> column=3 
    S ==> column=4 

            0    1    2    3    4    5    6          7  8
    0    0.0  0.0  1.0  0.0  0.0  0.0  0.0  00001.jpg  2
    20   0.0  0.0  0.0  0.0  0.0  0.0  1.0  00021.jpg  4
    36   0.0  0.0  0.0  0.0  0.0  1.0  0.0  00037.jpg  3
    37   0.0  1.0  0.0  0.0  0.0  0.0  0.0  00038.jpg  2
    40   1.0  0.0  0.0  0.0  0.0  0.0  0.0  00041.jpg  4
    95   0.0  0.0  0.0  1.0  0.0  0.0  0.0  00098.jpg  5
    158  0.0  0.0  0.0  0.0  1.0  0.0  0.0  00161.jpg  2

    '''

    dataframe.to_csv('DataCrohnIPI_2021_03\\DataCrohnIPI\\CrohnIPI_description_processed.csv')


def early_stopping():

    return keras.callbacks.EarlyStopping(
        monitor='val_auc',
        min_delta=0.01,
        patience=10,
        mode='max',
        start_from_epoch=30,
        restore_best_weights=True,
    )


def model_checkpoint(checkpoint_dir):

    return ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'crohnv_net_val_auc_{val_auc:.4f}.keras'),
        mode='max', # detecta automaticamente
        save_best_only=True, # salvar quando a métrica melhora
        save_weights_only=False, # somente os pesos
        monitor='val_auc', # métrica balizadora do armazenamento (precisão da validação)
        verbose=1 # logs de salvamento
    )