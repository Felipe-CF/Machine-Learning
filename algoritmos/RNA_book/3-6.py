import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv


def open_dataset():
    # load_dotenv()

    # dataset_path = os.getenv('apendice1')

    dataset = pd.read_csv("C:\\Users\\FelipeCF\\Desktop\\Codigos\\Machine-Learning\\algoritmos\\RNA_book\\datasets\\apendice1.csv", sep=',', encoding='iso-8859-1')

    dataset = pd.DataFrame.copy(dataset)

    entradas = dataset.iloc[:, 1:4]

    saidas = dataset.iloc[:, 4:]
    
    return entradas, saidas



if __name__ == '__main__':
    x, y = open_dataset()

    