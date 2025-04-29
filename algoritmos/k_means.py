import os
import warnings
import numpy as np
import pandas as pd
from dataset_mall import *
from dotenv import load_dotenv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore')

# com todos os atributos

scaler = StandardScaler()

dados_frame_esc = scaler.fit_transform(dados_frame)

wcss = []

for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, init='k-means++',  random_state=5, max_iter=300)

    kmeans.fit(dados_frame_esc)

    wcss.append(kmeans.inertia_)

# Gráfico
plt.figure(figsize=(10, 5))
sns.lineplot(x=range(1, 15), y=wcss, marker='o', color='red')
plt.title('Elbow method')
plt.xlabel('n° clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=6, init='k-means++',  random_state=5, max_iter=300)

kmeans2 = kmeans.fit(dados_frame_esc)

centroides = scaler.inverse_transform(kmeans2.cluster_centers_)

dados_dataframe = scaler.inverse_transform(dados_frame_esc)

classificacao = kmeans2.labels_

agrupamento = pd.DataFrame(classificacao, columns=['grupo'])

dados_concat = pd.concat([dados_frame, agrupamento], axis=1)

print(dados_concat.head())
print(dados_frame.iloc[125, :])









