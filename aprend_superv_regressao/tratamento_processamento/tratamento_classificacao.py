import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler

dados = pd.read_csv('C:\\Users\\FelipeCF\\Desktop\\Codigos\\Machine-Learning\\archive\\dataset-heart.csv',
                    sep=',', encoding='iso-8859-1')

# dados.head(10) # primeiras linhas
# dados.tail(10) # ultimas linhas

# dados.shape quantidade de registros

dados['Age'].value_counts()

dados['Age'].value_counts().sort_index() # ordena resultados pelo parametro acessado


# ===   GRÁFICO com seaborn ===
'''
    sns.histplot(dados, x='RestingBP', bins=30, color='orange', kde=True, stat='count')

    sns.histplot(dados, x='MaxHR', bins=30, color='orange', kde=True, stat='count')
'''
    # CONTAGEM
'''
    sns.countplot(dados, x='ExerciseAngina')
'''

# ===   GRÁFICO com plotly ===
    # PIZZA
'''
    px.pie(dados, names='ExerciseAngina')

    px.pie(dados, names='ST_Slope')

    px.pie(dados, names='HeartDisease')
'''
    # HISTOGRAMA
'''
    histograma = px.histogram(dados, x='Age', nbins=60) # nbins define agrupamento das infos

    histograma.update_layout(width=800, height=500, title_text="Distribuição das idades") # dimensões de exibição

    histograma.show()
'''
    # BOXPLOT
'''
    px.box(dados2, y='Age')

'''


# Análise dos tipos de atributos
'''
    dados.dtypes
    # object = strings
    # float64 = reais
    # compelx = complexos
'''


# Valores Missing
'''
    dados.isnull().sum()

    # excluir dados missing
    dados2 = dados.dropna()

    # substituir os dados missing pela média ou por qualquer outro valor
    dados2.Age.fillna(dados2.Age.mean(), inplace=True) # alterar e manter a alteração (True)

'''

# Tratando valores incoerentes

'''
    # pressão com valor 0
    dados2 = dados.loc[dados.RestingBP != 0] # localizar todos os valores diferentes de 0

    dados2.describe()

    # colesterol com valor 0
    dados2.Cholesterol.replace(0, np.nan, inplace=True) #excluindo registros com valor 0

    #substituir pela media
    dados2.Cholesterol.replace(np.nan, dados2.Cholesterol.mean(),inplace=True)

    # ou
    dados2.Cholesterol.fillna(dados2.Cholesterol.mean(), inplace=True)
'''


# Salvando (exportando) o dataframe tratado
'''
    dados2.to_csv('heart_tratado.csv', sep=';', encoding='utf-8', index=False)
'''


