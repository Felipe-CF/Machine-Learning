import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

'''

RESUMO 
alvo    ==> 	variável que se pretende atingir (tem ou não doença cardíaca)

previsores      ==> 	conjunto de variáveis previsoras com as variáveis categóricas
                    transformadas em numéricas manualmente, sem escalonar

previsores_escalonados    ==> 	conjunto de variáveis previsoras com as variáveis categóricas
                    transformadas em numéricas e escalonadas

previsores2    ==> 	conjunto de variáveis previsoras com as variáveis categóricas
                    transformadas em numéricas pelo LabelEncoder

previsores3    ==> 	conjunto de variáveis previsoras com as variáveis categóricas
                    transformadas em numéricas pelo LabelEncoder e OneHotEncoder, sem escalonar

previsores3_escalonados    ==> 	conjunto de variáveis previsoras com as variáveis categóricas
                    transformadas em numéricas pelo LabelEncoder e OneHotEncoder, escalonadas

'''

dados = pd.read_csv('C:\\Users\\FelipeCF\\Desktop\\Codigos\\Machine-Learning\\archive\\heart_tratado.csv',
                    sep=';', encoding='iso-8859-1')

dados_tratados = pd.DataFrame.copy(dados)

# ===   Transformando as variáveis categóricas nominais em categóricas ordinais ===

dados_tratados['Sex'].replace(
    {
        "M":0,
        "F": 1
    }, inplace=True
)

dados_tratados['ChestPainType'].replace(
    {'TA': 0, 
     'ATA': 1, 
     'NAP': 2, 
     'ASY': 3},inplace=True)

dados_tratados['RestingECG'].replace(
    {'Normal': 0, 
        'ST': 1, 
        'LVH': 2},inplace=True)

dados_tratados['ExerciseAngina'].replace(
    {'N': 0, 
     'Y': 1},inplace=True)

dados_tratados['ST_Slope'].replace(
    {'Up': 0, 
     'Flat': 1, 
     'Down': 2},inplace=True)

# ===   LEGENDA ===

'''
    Age = int

    Sex = (M=0, F=1)

    Chest Pain Type (tipo de dor no peito) = [TA=0(tangina típica), ATA=1(angina atípica), NAP=2 (dor não anginosa), ASY=3 (assintomático)]

    RestinBP (pressão sanguínea em repouso em mmHg) = float

    Cholesterol (sérico, mg/dl) = float

    Fasting BS (açucar no sangue em jejum mg/dl) = [0 : < 120, 1 >= 120]

    Resting ECG (eletrocardiograma em repouso) = [normal=0, ST(anormalidade da onda)=1, ST-T (hipertrofia ventricular esquerda)]

    Max HR (frequencia cardiaca maxima) = float

    Exercise Angina = [Não=0, Sim=1]

    Old Peak (depressão de ST induzida por exercicio em relação ao repouso)

    ST_Slope (inclinação do segmenet ST) = [Up=0, Flat=1, Down=2]

    Heart Disease = [Não=0, Sim=1]
'''

# ===   Atributos previsores e alvo ===

previsores = dados_tratados.iloc[:, 0:11].values # linhas, colunas

alvo = dados_tratados.iloc[:, 11].values # pegar coluna alvo da análise


#   === Análise das escalas dos atributos (escalonamento)   ===

'''
Padronização (utiliza a média e o desvio padrão como referência) <==

Normalização(utiliza os valores max e min como referência)

==> média proxima de 0 e desvio padrão proximo de 1
'''

previsores_escalonados = StandardScaler().fit_transform(previsores)

previsoresdf = pd.DataFrame(previsores_escalonados)


#   === LabelEncoder: Codificação de variáveis categóricas   ===
previsores2 = dados.iloc[:, 0:11].values

# Aplica Label Encoding — ou seja, converte valores categóricos em números inteiros.
previsores2[:, 1] = LabelEncoder().fit_transform(previsores[:, 1])
previsores2[:, 2] = LabelEncoder().fit_transform(previsores[:, 2])
previsores2[:, 6] = LabelEncoder().fit_transform(previsores[:, 6])
previsores2[:, 8] = LabelEncoder().fit_transform(previsores[:, 8])
previsores2[:, 10] = LabelEncoder().fit_transform(previsores[:, 10])

# print(previsores2.shape)

#   === OneHotEncoder: Criação de variáveis Dummy   ===
'''
transformers    ==> 	Lista de tuplas no formato ('nome', transformador, colunas_do_labelEncoder)

remainder	==>     Define o que fazer com as colunas não transformadas:
🔹 'drop' (padrão) → descarta
🔹 'passthrough' → mantém

sparse_threshold    ==>	Se algum transformador retorna matriz esparsa, define se o resultado final também será esparso (padrão: 0.3)

n_jobs	==> Número de threads para paralelizar (ex: -1 usa todos os núcleos)

verbose	==> Se True, mostra mais informações durante a transformação

verbose_feature_names_out   ==>	Se True, mostra os nomes completos das features geradas

'''

# Aplica OneHot Encoding nas colunas que já foram LabelEncoded. Ou seja, transforma essas colunas numéricas em vetores binários (dummies).
previsores2 = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 2, 6, 8, 10])], 
                                remainder='passthrough').fit_transform(previsores2)

previsores3_df = pd.DataFrame(previsores3)

print(previsores3_df.head())

# print(dados.head())

#   === Escalonamento com as variáveis Dummy   ===

previsores3_escalonados = StandardScaler().fit_transform(previsores3)
 
previsores3_df = pd.DataFrame(previsores3_escalonados)

print(previsores3_df.describe())
