import os, keras, json
import pandas as pd
import matplotlib as plt
from PIL import Image, UnidentifiedImageError
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
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
        validation_split=0.2, # separação do subset de Validation
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
        start_from_epoch=70,
        restore_best_weights=True,
    )


def model_checkpoint(checkpoint_dir):

    return ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'crohn_net_{val_auc:.4f}.keras'),
        mode='max', # detecta automaticamente
        save_best_only=True, # salvar quando a métrica melhora
        save_weights_only=False, # somente os pesos
        monitor='val_auc', # métrica balizadora do armazenamento (Accuracy da Validation)
        verbose=1 # logs de salvamento
    )


def save_history(file_dir, history):
    file_dir = os.path.dirname(os.path.abspath(__file__))

    history_path = os.path.join(file_dir, 'resnet_fit_history')

    auc = max(history.history['auc'])

    val_auc = max(history.history['val_auc'])

    history_path = history_path + f'\\fit_history_auc_{auc:.4f}_val_auc_{val_auc:.4f}.json'

    with open(history_path, 'w') as file:
        file.write(json.dumps(history.history))

    print('Last history of training saved sucessfull!')


def generate_grafics(history):
    auc = history['auc']

    val_auc = history['val_auc']

    loss = history['binary_crossentropy']

    val_loss = history['val_binary_crossentropy']

    binary_crossentropy = history['binary_crossentropy']

    val_binary_crossentropy = history['val_binary_crossentropy']

    epochs = history['epoch']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Training History - CrohnNet', fontsize=16)
    
    # Gráfico de AUC (o equivalente a acurácia no seu log)
    axes[0].plot(epochs, auc, label='Traning AUC', color='blue')
    axes[0].plot(epochs, val_auc, label='Validation AUC', color='red', linestyle='--')
    axes[0].set_title('Accuracy - AUC vs. Epochs')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('AUC')
    axes[0].legend()
    axes[0].grid(True)
    
    # Gráfico de Binary Cross-Entropy
    axes[1].plot(epochs, binary_crossentropy, label='Traning BCE', color='blue')
    axes[1].plot(epochs, val_binary_crossentropy, label='Validation BCE', color='red', linestyle='--')
    axes[1].set_title('Binary Cross-Entropy vs. Epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Binary Cross-Entropy')
    axes[1].legend()
    axes[1].grid(True)

    # Gráfico de Loss
    axes[2].plot(epochs, loss, label='Traning Loss', color='blue')
    axes[2].plot(epochs, val_loss, label='Validation Loss', color='red', linestyle='--')
    axes[2].set_title('Loss Fuction vs. Epochs')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def learning_rate_plateau():

    return ReduceLROnPlateau(
        monitor='val_auc',
        mode='max',
        factor=0.1, 
        patience=10,
        min_delta=0.01,
        cooldown=10,
        verbose=1
    )