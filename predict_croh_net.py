import os, time, shutil
import numpy as np
from util.predict_settings import *


def predict_crohn_net(img_files, pathologic_path, health_path, avaliation_path):

    for i in range(0, len(img_files)):
        start = time.perf_counter()

        img = image.load_img(
            path=img_files[i],
            target_size=(320, 320)
        )

        img_array = processing_image(img)

        prediction = croh_net.predict(img_array, verbose=0)

        avaliation_prediction = prediction.flatten().tolist()

        if abs(avaliation_prediction[0] - avaliation_prediction[1]) < 0.1:
            shutil.copy(img_files[i], avaliation_path)

        else:
            prediction = np.where(prediction > 0.5, 1, 0).flatten().tolist()

            if prediction[0] == 1:
                shutil.copy(img_files[i], pathologic_path)

            else:
                shutil.copy(img_files[i], health_path)

        end = time.perf_counter() - start

        times.append(end)

    mean_time_per_image = np.mean(times)

    print(f"Tempo médio por imagem: {mean_time_per_image:.2f} segundos")

    mean_time_per_1k = (mean_time_per_image * 1000) / 60

    print(f"Tempo médio a cada 1000 imagens: {mean_time_per_1k:.2f} minutos")


if __name__ == '__main__':

    try:
        file_dir = os.path.dirname(os.path.abspath(__file__))

        croh_net = load_model(file_dir)

        predict_results = []

        imgs_path = os.path.join(file_dir, 'db\\DataCrohnIPI_2021_03\\DataCrohnIPI\\imgs')

        times = []

        img_files = []

        for root, dirs, files in os.walk(imgs_path):

            for file in files[:100]:
                img_files.append(os.path.join(root, file))

        pathologic_folder = os.path.join(imgs_path, 'patologico')

        health_folder = os.path.join(imgs_path, 'saudavel')

        avaliation_folder = os.path.join(imgs_path, 'avaliacao')

        pathologic_path, health_path, avaliation_path = create_folders(pathologic_folder, health_folder, avaliation_folder)

        predict_crohn_net(
            img_files,
            pathologic_path,
            health_path,
            avaliation_path
            )

    except FileNotFoundError:
        print(f"Error: The source file was not found")

    except Exception as e:
        print(f"An error occurred: {e}")








