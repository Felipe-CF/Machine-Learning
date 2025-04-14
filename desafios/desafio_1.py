from dados_base.data_d1 import *
import pandas as pd

# c2 = dados_frame.iloc[:, 2].values
# print(c2.mean())
print(dados_frame.head())

previsores = dados_frame.iloc[:, 2:32]

# print(dados_frame.isna().sum())