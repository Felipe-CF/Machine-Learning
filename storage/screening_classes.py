from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import os
import pandas as pd



file_dir = os.path.dirname(os.path.abspath(__file__))

dataset_dir = file_dir + '\\DataCrohnIPI_2021_03\\DataCrohnIPI'

dataframe_path = os.path.join(dataset_dir, 'CrohnIPI_description.csv')

dataframe = pd.read_csv(dataframe_path, sep=',', encoding='iso-8859-1')

print(f"O {(dataframe['Label'] == 'O').sum()}")
print(f"S {(dataframe['Label'] == 'S').sum()}")
print(f"E {(dataframe['Label'] == 'E').sum()}")
print(f"AU {(dataframe['Label'] == 'AU').sum()}")
print(f"U>10 {(dataframe['Label'] == 'U>10').sum()}")
print(f"U3-10 {(dataframe['Label'] == 'U3-10').sum()}")


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

print(f"P {(dataframe['Label'] == 'P').sum()}")
print(f"N {(dataframe['Label'] == 'N').sum()}")


# dataframe = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1])], remainder='passthrough').fit_transform(dataframe)

# dataframe = pd.DataFrame(dataframe)

# dataframe.to_csv('DataCrohnIPI_2021_03\\DataCrohnIPI\\CrohnIPI_description_processed.csv')
