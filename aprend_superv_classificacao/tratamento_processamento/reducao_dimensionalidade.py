import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA

dados = pd.read_csv('C:\\Users\\FelipeCF\\Desktop\\Codigos\\Machine-Learning\\archive\\heart_tratado.csv',
                    sep=';', encoding='iso-8859-1')

dados_tratados = pd.DataFrame.copy(dados)

previsores = dados.iloc[:, 0:11].values

previsores2 = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 2, 6, 8, 10])], 
                                remainder='passthrough').fit_transform(previsores)

previsores3_df = pd.DataFrame(previsores2)

previsores3_escalonados = StandardScaler().fit_transform(previsores2)

