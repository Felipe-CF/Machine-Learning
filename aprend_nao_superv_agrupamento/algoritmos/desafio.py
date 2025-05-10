import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from dataset_wine import *
import plotly.express as px
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

wcss = []

for cluster_i in list(range(1, 15)):
    kmeans = KMeans(n_clusters=cluster_i, init='k-means++', random_state=5, max_iter=400, verbose=1, tol=.0008)
    kmeans.fit(dados_frame_esc)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
sns.lineplot(x=range(1, 15), y=wcss, marker='o', color='red')
plt.title('Elbow method')
plt.xlabel('n° clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=5, max_iter=400, tol=.0008)

kmeans2 = kmeans.fit(dados_frame_esc)

centroides = scaler.inverse_transform(kmeans2.cluster_centers_)

dados_df = scaler.inverse_transform(dados_frame_esc)

classificacao = kmeans2.labels_

graf = px.scatter(x=dados_df[:, 0], y=dados_df[:, 1], color=classificacao)
graf2 = px.scatter(x=centroides[:, 0], y=centroides[:, 1], size=[15,15,15]) # size = qnt de clusters
graf3 = go.Figure(data=graf.data+graf2.data)
graf3.update_layout(width=800, height=500, title_text='Agrupamento K-Means')
graf3.update_xaxes(title='rendimento')
graf3.update_yaxes(title='pontuacao')
graf3.show()

agrupamento = pd.DataFrame(classificacao, columns=['grupo'])

dados_concat = pd.concat([dados_frame, agrupamento], axis=1)

print(dados_concat.head())

