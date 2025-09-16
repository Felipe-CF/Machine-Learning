import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
from PIL import Image, UnidentifiedImageError
from keras_preprocessing.image import ImageDataGenerator


def create_sets(processing=None):
    data_gen = ImageDataGenerator( #objeto com regras para o pré-processamento de imagens
        rescale=1./255, 
        # augmentation
        shear_range=0.2, # distorção de inclinação
        zoom_range=0.2, # zoom in e out aleatorio
        horizontal_flip=True, # aleatorio
        vertical_flip=True, # aleatorio
        brightness_range=[0.2, 0.8],
        samplewise_std_normalization=True,
        validation_split=0.2, # separação do subset de Validation
    )

    file_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_dir = file_dir + '\\db\\DataCrohnIPI_2021_03\\DataCrohnIPI\\'

    dataframe_path = os.path.join(dataset_dir, 'CrohnIPI_description_screening_processed.csv')

    dataframe = pd.read_csv(dataframe_path, sep=',', encoding='iso-8859-1')

    training_set = data_gen.flow_from_dataframe(
        directory= dataset_dir + '\\imgs',
        dataframe=dataframe,
        y_col=['0', '1'],
        x_col='2',
        subset='training',
        batch_size=16,
        shuffle=True,
        class_mode='raw',
        target_size=(320, 320)
    )

    validation_set = data_gen.flow_from_dataframe(
        directory= dataset_dir + '\\imgs',
        dataframe=dataframe,
        y_col=['0', '1'],
        x_col='2',
        target_size=(320, 320),
        batch_size=16,
        class_mode='raw',
        shuffle=True,
        subset='validation'
        )
    
    return training_set, validation_set


if __name__ == '__main__':
        file_dir = os.path.dirname(os.path.abspath(__file__))

        dataset_dir = file_dir + '\\DataCrohnIPI_2021_03\\DataCrohnIPI'

        dataframe_path = os.path.join(dataset_dir, 'CrohnIPI_description_screening_processed.csv')

        dataframe = pd.read_csv(dataframe_path, sep=',', encoding='iso-8859-1')

        kfold_1 = dataframe.loc[dataframe['3'] == 1]

        print('1')
        print(f"P {(kfold_1['1'] == 1).sum()}")
        print(f"N {(kfold_1['0'] == 1).sum()}")

        kfold_2 = dataframe.loc[dataframe['3'] == 2]

        print('2')
        print(f"P {(kfold_2['1'] == 1).sum()}")
        print(f"N {(kfold_2['0'] == 1).sum()}")


        kfold_3 = dataframe.loc[dataframe['3'] == 3]

        print('3')
        print(f"P {(kfold_3['1'] == 1).sum()}")
        print(f"N {(kfold_3['0'] == 1).sum()}")


        kfold_4 = dataframe.loc[dataframe['3'] == 4]

        print('4')
        print(f"P {(kfold_4['1'] == 1).sum()}")
        print(f"N {(kfold_4['0'] == 1).sum()}")

        kfold_5 = dataframe.loc[dataframe['3'] == 5]

        print('5')
        print(f"P {(kfold_5['1'] == 1).sum()}")
        print(f"N {(kfold_5['0'] == 1).sum()}")



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

# print(f"U>10 {(dataframe['Label'] == 'U>10').sum()}")
# print(f"U3-10 {(dataframe['Label'] == 'U3-10').sum()}")
# print(f"E {(dataframe['Label'] == 'E').sum()}")
# print(f"AU {(dataframe['Label'] == 'AU').sum()}")
# print(f"O {(dataframe['Label'] == 'O').sum()}")
# print(f"S {(dataframe['Label'] == 'S').sum()}")
# print(f"N {(dataframe['Label'] == 'N').sum()}")


'''

        # print(dataframe)

        # dataframe = dataframe.sample(frac=1).reset_index(drop=True)

        # dataframe = dataframe.iloc[0:5]

        # print(dataframe)

        # dataframe = pd.DataFrame(dataframe)
