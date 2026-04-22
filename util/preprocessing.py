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

    df = pd.read_csv(dataframe_path, sep=',', encoding='iso-8859-1')

    df['Label'].replace(
        {
            "U>10" : 'P',
            "U3-10" : 'P',
            "E" : 'P',
            "AU" : 'P',
            "O" : 'P',
            "S" : 'P'
        }, inplace=True
    )


    df = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1])], remainder='passthrough').fit_transform(df)

    df = pd.DataFrame(df)

    df[0] = df[0].astype(float)

    df[1] = df[1].astype(float)

    df = pd.DataFrame(df)

    folds = []

    json_folds = {}

    for i in range(5):
        fold_df = pd.DataFrame(df[df[3] == i+1])

        kfold = {
            'validation': False,
            'fold': fold_df.to_json(orient='records'),
        }

        json_folds[f'fold{i+1}'] = kfold

        folds.append(kfold)

    path_folds = 'db\\DataCrohnIPI_2021_03'

    with open('db\\DataCrohnIPI_2021_03\\dados.json', 'w', encoding='utf-8') as file:
        json.dump(json_folds, file, ensure_ascii=False, indent=4)
