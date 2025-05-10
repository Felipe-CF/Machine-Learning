import os, warnings
import numpy as np
import pandas as pd
from dataset_bankmarketing import *
from dotenv import load_dotenv
from kmodes.kprototypes import KPrototypes

warnings.filterwarnings('ignore')

'''
Agrupamento de dados categóricos
'''


dados_kproto = dados_frame[[
    'age', 'job', 'marital', 'education', 'default', 'housing',
    ]]

# kproto1 = kproto.fit_predict(dados_kproto, categorical=[0])

# agrupamento = pd.DataFrame(kproto1, columns=['grupo'])

# dados_concat = pd.concat([dados_frame, agrupamento], axis=1)

# print(dados_concat['grupo'].unique())
# print(dados_concat.head())


