import os, json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


file_dir = os.path.dirname(os.path.abspath(__file__))

dataset_dir = file_dir + '\\DataCrohnIPI_2021_03\\DataCrohnIPI'

dataframe_path = os.path.join(dataset_dir, 'CrohnIPI_description.csv')

dataframe = pd.read_csv(dataframe_path, sep=',', encoding='iso-8859-1')

print(dataframe)

print(f"{dataframe['Label'].unique()}")

# print(f"{(dataframe['Label'] == 'N').sum()}")
# print(f"{(dataframe['Label'] == 'E').sum()}")
# print(f"{(dataframe['Label'] == 'S').sum()}")
# print(f"{(dataframe['Label'] == 'O').sum()}")
# print(f"{(dataframe['Label'] == 'AU').sum()}")
# print(f"{(dataframe['Label'] == 'U>10').sum()}")
# print(f"{(dataframe['Label'] == 'U3-10').sum()}")

dataframe = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1])], remainder='passthrough').fit_transform(dataframe)

dataframe = pd.DataFrame(dataframe)

print(dataframe)

# dataframe.to_csv('DataCrohnIPI_2021_03\\DataCrohnIPI\\CrohnIPI_description_screening_processed.csv')



