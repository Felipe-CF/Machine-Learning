import os, warnings
import numpy as np
import pandas as pd
from dataset_mall import *
from dotenv import load_dotenv
from kmodes.kprototypes import KPrototypes

warnings.filterwarnings('ignore')

kproto = KPrototypes(n_clusters=4)

dados_kproto = dados_frame[['genero', 'idade', 'rendimento_anual', 'pontuacao']]

kproto1 = kproto.fit_predict(dados_kproto, categorical=[0])

agrupamento = pd.DataFrame(kproto1, columns=['grupo'])

dados_concat = pd.concat([dados_frame, agrupamento], axis=1)

print(dados_concat['grupo'].unique())
print(dados_concat.head())


