import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
from keras_preprocessing.image import ImageDataGenerator


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
        target_size=(320, 320)
    )

    validation_set = data_gen.flow_from_dataframe(
        directory='C:\\Users\FelipeCF\\Desktop\\Codigos\\Machine-Learning\\DataCrohnIPI_2021_03\\DataCrohnIPI\\imgs',
        dataframe=dataframe,
        x_col='Frame name',
        y_col='Label',
        target_size=(320, 320),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        subset='validation'
        )
    
    return training_set, validation_set
