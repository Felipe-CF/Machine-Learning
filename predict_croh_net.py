import numpy as np
import tensorflow as tf
import keras, os, json, time
from datetime import datetime
from keras_preprocessing import image


def load_model(file_dir):

    best_model_path = os.path.join(file_dir, 'screening_checkpoints\\kfold4\\screening_net_0.9372.keras')

    conv_net = keras.saving.load_model(best_model_path, compile=True, safe_mode=True, custom_objects=None)
    
    return conv_net


def sample_std_dev(np_array):

    best_model_path = os.path.join(file_dir, 'screening_checkpoints\\kfold4\\screening_net_0.9372.keras')

    conv_net = keras.saving.load_model(best_model_path, compile=True, safe_mode=True, custom_objects=None)
    
    return conv_net


if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))

    croh_net = load_model(file_dir)

    predict_results = []

    imgs_path = os.path.join(file_dir, 'db\\DataCrohnIPI_2021_03\\DataCrohnIPI\\imgs')
    
    times = []

    img_files = []

    for root, dirs, files in os.walk(imgs_path):

        for file in files[:1000]:
            img_files.append(os.path.join(root, file))

    for i in range(0, len(img_files)):
        start = time.perf_counter()

        img = image.load_img(
            path=img_files[i],
            target_size=(320, 320)
        )

        img_array  = image.img_to_array(img)

        img_array /=  255.0 #rescale

        img_array -= np.mean(img_array, keepdims=True) # subtracted by the mean

        img_array /= np.std(img_array, keepdims=True) + 1e-6 # divided by std dev
        
        img_array= np.expand_dims(img_array, axis=0) # img ready to be predicted

        prediction = np.where(croh_net.predict(img_array, verbose=0) > 0.5, 1, 0).tolist() # save the prediction into a python list

        end = time.perf_counter() - start

        times.append(end)

    mean_time_per_image = np.mean(times)

    print(f"Tempo médio por imagem: {mean_time_per_image:.2f} segundos")

    mean_time_per_1k = (mean_time_per_image * 1000) / 60

    print(f"Tempo médio a cada 1000 imagens: {mean_time_per_1k:.2f} minutos")


