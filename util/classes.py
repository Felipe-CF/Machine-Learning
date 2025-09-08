from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import os
import pandas as pd



file_dir = os.path.dirname(os.path.abspath(__file__))

dataset_dir = file_dir + '\\DataCrohnIPI_2021_03\\DataCrohnIPI'

dataframe_path = os.path.join(dataset_dir, 'CrohnIPI_description.csv')

dataframe = pd.read_csv(dataframe_path, sep=',', encoding='iso-8859-1')

print(f"U>10 {(dataframe['Label'] == 'U>10').sum()}")
print(f"U3-10 {(dataframe['Label'] == 'U3-10').sum()}")
print(f"E {(dataframe['Label'] == 'E').sum()}")
print(f"AU {(dataframe['Label'] == 'AU').sum()}")
print(f"O {(dataframe['Label'] == 'O').sum()}")
print(f"S {(dataframe['Label'] == 'S').sum()}")
print(f"N {(dataframe['Label'] == 'N').sum()}")


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
