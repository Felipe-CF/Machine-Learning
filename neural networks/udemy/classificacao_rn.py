import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

load_dotenv()

output_dir = os.getenv('OUTPUT_DIR')

dados = pd.read_csv(os.path.join(output_dir, 'heart_tratado.csv'), sep=';', encoding='iso-8859-1')

dados_tratados = pd.DataFrame.copy(dados)

dados_tratados['Sex'].replace({
    'M': 0,
    'F': 1
}, inplace=True)

dados_tratados['ChestPainType'].replace({
    'TA': 0,
    'ATA': 1,
    'NAP': 2,
    'ASY': 3
}, inplace=True)

dados_tratados['RestingECG'].replace({
    'Normal': 0,
    'ST': 1,
    'LVH': 2
}, inplace=True)

dados_tratados['ExerciseAngina'].replace({
    'N': 0,
    'Y': 1
}, inplace=True)

dados_tratados['ST_Slope'].replace({
    'Up': 0,
    'Flat': 1,
    'Down': 2
}, inplace=True)

previsores = dados_tratados.iloc[:, 0:11].values

alvo = dados_tratados.iloc[:, 11].values

previsores_esc = StandardScaler().fit_transform(previsores)

previsores_label = dados_tratados.iloc[:, 0:11].values

previsores_label[:, 1] = LabelEncoder().fit_transform(previsores_label[:, 1])
previsores_label[:, 2] = LabelEncoder().fit_transform(previsores_label[:, 2])
previsores_label[:, 6] = LabelEncoder().fit_transform(previsores_label[:, 6])
previsores_label[:, 8] = LabelEncoder().fit_transform(previsores_label[:, 8])
previsores_label[:, 10] = LabelEncoder().fit_transform(previsores_label[:, 10])

previsores_encoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 2, 6, 8, 10])],
                                     remainder='passthrough').fit_transform(previsores_label)

previsores_encoder_esc = StandardScaler().fit_transform(previsores_encoder)
