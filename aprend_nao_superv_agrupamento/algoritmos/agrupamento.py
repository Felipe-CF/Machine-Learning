import os, warnings
import numpy as np
import pandas as pd
import seaborn as sns
from dataset_mall import *
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering


warnings.filterwarnings('ignore')

scaler = StandardScaler()

dados_frame_esc = scaler.fit_transform(dados_frame)

pca = PCA(n_components=3)

dados_frame_pca = pca.fit_transform(dados_frame_esc)

# criar dendograma
dendrograma = dendrogram(linkage(dados_frame_pca, method='complete'))

hier = AgglomerativeClustering(n_clusters=7, linkage='complete')

classificacao = hier.fit_predict(dados_frame_esc)

agrupamento = pd.DataFrame(classificacao, columns=['grupo'])

dados_concat = pd.concat([dados_frame, agrupamento], axis=1)

print(dados_concat.head())