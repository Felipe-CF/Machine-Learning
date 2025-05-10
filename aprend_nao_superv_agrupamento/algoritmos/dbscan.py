import os, warnings
import numpy as np
import pandas as pd
import seaborn as sns
from dataset_mall import *
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


warnings.filterwarnings('ignore')

scaler = StandardScaler()

dados_frame_esc = scaler.fit_transform(dados_frame)

# samples crescem junto com a quantidade de ruídos
dbscan = DBSCAN(eps=0.72, min_samples=4) 

dbscan.fit(dados_frame_esc)

classificacao = dbscan.labels_

agrupamento = pd.DataFrame(classificacao, columns=['grupo'])

dados_concat = pd.concat([dados_frame, agrupamento], axis=1)

print(dados_concat['grupo'].unique())

print((dados_concat.grupo == -1).sum())
