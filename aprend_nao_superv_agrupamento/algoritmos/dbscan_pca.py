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

pca = PCA(n_components=2)

dados_frame_pca = pca.fit_transform(dados_frame_esc)

# samples crescem junto com a quantidade de ruídos
dbscan = DBSCAN(eps=0.35, min_samples=2) 

dbscan.fit(dados_frame_pca)

classificacao = dbscan.labels_

graf = px.scatter(x=dados_frame_pca[:, 0], y=dados_frame_pca[:, 1], color=classificacao)
graf.update_layout(width=800, height=500, title_text='Agrupamento DBSCAN')
graf.show()

agrupamento = pd.DataFrame(classificacao, columns=['grupo'])

dados_concat = pd.concat([dados_frame, agrupamento], axis=1)

print(dados_concat['grupo'].unique())


print(dados_concat.head())
