import os
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

output_dir = os.getenv('OUTPUT_DIR')

dados = pd.read_csv(os.path.join(output_dir, 'heart_tratado.csv'),
                    sep=';', encoding='iso-8859-1')

dados_tratados = pd.DataFrame.copy(dados)

dados_tratados['Sex'].replace(
    {
        "M":0,
        "F": 1
    }, inplace=True
)

dados_tratados['ChestPainType'].replace(
    {'TA': 0, 
     'ATA': 1, 
     'NAP': 2, 
     'ASY': 3},inplace=True)

dados_tratados['RestingECG'].replace(
    {'Normal': 0, 
        'ST': 1, 
        'LVH': 2},inplace=True)

dados_tratados['ExerciseAngina'].replace(
    {'N': 0, 
     'Y': 1},inplace=True)

dados_tratados['ST_Slope'].replace(
    {'Up': 0, 
     'Flat': 1, 
     'Down': 2},inplace=True)

previsores = dados_tratados.iloc[:, 0:11].values

alvo = dados_tratados.iloc[:,11].values


