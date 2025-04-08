import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler

dados = pd.read_csv('C:\\Users\\FelipeCF\\Desktop\\Codigos\\Machine-Learning\\archive\\dataset-heart.csv',
                    sep=',', encoding='iso-8859-1')

    dados.head(10) primeiras linhas
    dados.tail(10) ultimas linhas

    dados.shape quantidade de registros

## Acessando previsores
    dados['Age'].value_counts()

    dados['Age'].value_counts().sort_index() # ordena resultados pelo parametro acessado

    dados.Age.value_counts()

# ===   GRÁFICO com seaborn ===

    # HISTOGRAMA
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





