import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image, UnidentifiedImageError
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold


if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_dir = file_dir + '\\DataCrohnIPI_2021_03\\DataCrohnIPI'

    # dataframe_path = os.path.join(dataset_dir, 'CrohnIPI_description_screening_processed.csv')
    dataframe_path = os.path.join(dataset_dir, 'CrohnIPI_description.csv')

    dataframe = pd.read_csv(dataframe_path, sep=',', encoding='iso-8859-1')

    print(f"U>10 {(dataframe['Label'] == 'U>10').sum()}")
    print(f"U3-10 {(dataframe['Label'] == 'U3-10').sum()}")
    print(f"E {(dataframe['Label'] == 'E').sum()}")
    print(f"AU {(dataframe['Label'] == 'AU').sum()}")
    print(f"O {(dataframe['Label'] == 'O').sum()}")
    print(f"S {(dataframe['Label'] == 'S').sum()}")
    print(f"N {(dataframe['Label'] == 'N').sum()}")

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

    # dataframe.to_csv('DataCrohnIPI_2021_03\\DataCrohnIPI\\CrohnIPI_description_processed.csv')

    print(f"1 {dataframe[(dataframe['Fold'] == 1) & (dataframe['Label'] == 'N')].shape[0]}")
    print(f"1 {dataframe[(dataframe['Fold'] == 1) & (dataframe['Label'] == 'P')].shape[0]}")
    print(f"2 {dataframe[(dataframe['Fold'] == 2) & (dataframe['Label'] == 'N')].shape[0]}")
    print(f"2 {dataframe[(dataframe['Fold'] == 2) & (dataframe['Label'] == 'P')].shape[0]}")
    print(f"3 {dataframe[(dataframe['Fold'] == 3) & (dataframe['Label'] == 'N')].shape[0]}")
    print(f"3 {dataframe[(dataframe['Fold'] == 3) & (dataframe['Label'] == 'P')].shape[0]}")
    print(f"4 {dataframe[(dataframe['Fold'] == 4) & (dataframe['Label'] == 'N')].shape[0]}")
    print(f"4 {dataframe[(dataframe['Fold'] == 4) & (dataframe['Label'] == 'P')].shape[0]}")
    print(f"5 {dataframe[(dataframe['Fold'] == 5) & (dataframe['Label'] == 'N')].shape[0]}")
    print(f"5 {dataframe[(dataframe['Fold'] == 5) & (dataframe['Label'] == 'P')].shape[0]}")



    print(f"1 {(dataframe['Fold'] == 1 and dataframe['Label'] == 'P').sum()}")
    # print(f"2 {(dataframe['Fold'] == 2).sum()}")
    # print(f"3 {(dataframe['Fold'] == 3).sum()}")
    # print(f"4 {(dataframe['Fold'] == 4).sum()}")
    # print(f"5 {(dataframe['Fold'] == 5).sum()}")

    # kfold_1 = dataframe.loc[dataframe['3'] == 1]

    # kfold_2 = dataframe.loc[dataframe['3'] == 2]

    # kfold_3 = dataframe.loc[dataframe['3'] == 3]

    # kfold_4 = dataframe.loc[dataframe['3'] == 4]

    # kfold_5 = dataframe.loc[dataframe['3'] == 5]
    




# dataframe.to_csv('DataCrohnIPI_2021_03\\DataCrohnIPI\\CrohnIPI_description_processed.csv')



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



        # print('5')
        # print(f"P {(kfold_5['1'] == 1).sum()}")
        # print(f"N {(kfold_5['0'] == 1).sum()}")


'''

        # print(dataframe)

        # dataframe = dataframe.sample(frac=1).reset_index(drop=True)

        # dataframe = dataframe.iloc[0:5]

        # print(dataframe)

        # dataframe = pd.DataFrame(dataframe)
