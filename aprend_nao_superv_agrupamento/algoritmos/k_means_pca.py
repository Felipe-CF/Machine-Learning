import os, warnings
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
from sklearn.decomposition import PCA


scaler = StandardScaler()

dados_frame_esc = scaler.fit_transform(dados_frame)

pca = PCA(n_components=3)

dados_frame_pca = pca.fit_transform(dados_frame_esc)

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',  random_state=5, max_iter=300)

    kmeans.fit(dados_frame_pca)

    wcss.append(kmeans.inertia_)

# Gráfico
plt.figure(figsize=(10, 5))
sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
plt.title('Elbow method')
plt.xlabel('n° clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=4, init='k-means++',  random_state=5, max_iter=300)

kmeans2 = kmeans.fit(dados_frame_pca)

centroides = kmeans2.cluster_centers_

classificacao = kmeans2.labels_

agrupamento = pd.DataFrame(classificacao, columns=['grupo'])

dados_concat = pd.concat([dados_frame, agrupamento], axis=1)


# visualizar o agrupamento
graf = px.scatter(x=dados_frame_pca[:, 0], y=dados_frame_pca[:, 1], color=classificacao)
graf2 = px.scatter(x=centroides[:, 0], y=centroides[:, 1], size=[15,15,15,15]) # size = qnt de clusters
graf3 = go.Figure(data=graf.data+graf2.data)
graf3.update_layout(width=800, height=500, title_text='Agrupamento K-Means com PCA')
graf3.update_xaxes(title='componente 1')
graf3.update_yaxes(title='componente 2')
graf3.show()

# lista de grupos
agrupamento = pd.DataFrame(classificacao, columns=['grupo'])

# concatenar com o dataframe dos clientes, gerando a identificação por grupo
dados_concat = pd.concat([dados, agrupamento], axis=1)

print(dados_concat.head())