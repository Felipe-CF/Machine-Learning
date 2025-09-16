import os, json
import pandas as pd
import matplotlib as plt
from PIL import Image, UnidentifiedImageError
from keras_preprocessing.image import ImageDataGenerator
from util.preprocessing import dataframe_preprocessing


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

