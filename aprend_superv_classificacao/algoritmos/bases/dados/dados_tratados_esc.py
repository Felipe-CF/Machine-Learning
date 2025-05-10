from .dados_tratados import *
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


previsores_escalonados = StandardScaler().fit_transform(previsores)

# Aplica Label Encoding — ou seja, converte valores categóricos em números inteiros.
previsores_label = dados.iloc[:, 0:11].values

previsores_label[:, 1] = LabelEncoder().fit_transform(previsores[:, 1])
previsores_label[:, 2] = LabelEncoder().fit_transform(previsores[:, 2])
previsores_label[:, 6] = LabelEncoder().fit_transform(previsores[:, 6])
previsores_label[:, 8] = LabelEncoder().fit_transform(previsores[:, 8])
previsores_label[:, 10] = LabelEncoder().fit_transform(previsores[:, 10])


#   === OneHotEncoder: Criação de variáveis Dummy   ===
previsores_dummy = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 2, 6, 8, 10])],
                                     remainder='passthrough').fit_transform(previsores_label)

previsores_dummy_escalonados = StandardScaler().fit_transform(previsores_dummy)


