import os, json
import pandas as pd
import matplotlib as plt
from PIL import Image, UnidentifiedImageError
from keras_preprocessing.image import ImageDataGenerator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def dataframe_preprocessing():
    file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset_dir = file_dir + '\\db\\DataCrohnIPI_2021_03\\DataCrohnIPI'

    dataframe_path = os.path.join(dataset_dir, 'CrohnIPI_description.csv')

    dataframe = pd.read_csv(dataframe_path, sep=',', encoding='iso-8859-1')

    dataframe['Label'].replace(
        {
            "U>10" : 'P',
            "U3-10" : 'P',
            "E" : 'P', 
            "AU" : 'P', 
            "O" : 'P',
            "S" : 'P' 
        }, inplace=True
    )

    dataframe = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1])], remainder='passthrough').fit_transform(dataframe)

    dataframe = pd.DataFrame(dataframe)

    dataframe.to_csv('db\\DataCrohnIPI_2021_03\\DataCrohnIPI\\CrohnIPI_description_screening_processed.csv')