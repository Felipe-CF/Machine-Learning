import numpy as np
import keras, os
from keras_preprocessing import image


def load_model(file_dir):

    best_model_path = os.path.join(file_dir, 'screening_checkpoints\\kfold4\\screening_net_0.9372.keras')

    conv_net = keras.saving.load_model(best_model_path, compile=True, safe_mode=True, custom_objects=None)

    return conv_net


def processing_image(img):
    img_array  = image.img_to_array(img)

    img_array /=  255.0 #rescale

    img_array -= np.mean(img_array, keepdims=True) # subtracted by the mean

    img_array /= np.std(img_array, keepdims=True) + 1e-6 # divided by std dev

    img_array= np.expand_dims(img_array, axis=0) # img ready to be predicted

    return img_array


def create_folders(pathologic_folder, health_folder, avaliation_folder):

    if not os.path.exists(pathologic_folder):
        os.makedirs(pathologic_folder)

    if not os.path.exists(health_folder):
        os.makedirs(health_folder)

    if not os.path.exists(avaliation_folder):
        os.makedirs(avaliation_folder)

    return pathologic_folder, health_folder, avaliation_folder