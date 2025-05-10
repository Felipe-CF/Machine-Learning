import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dados = pd.read_csv('C:\\Users\\FelipeCF\\Desktop\\Codigos\\Machine-Learning\\archive\\heart_tratado.csv',
                    sep=';', encoding='iso-8859-1')

dados_tratados = pd.DataFrame.copy(dados)

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

print("RESULTADOS GERADOS COM OS PREVISORES NÃO ESCALONADOS")

previsores = dados_tratados.iloc[:, 0:11].values

alvo = dados_tratados.iloc[:,11].values

# Separo os dados dos previsores para treino e teste, seguindo a respectiva proporção: 0.7 e 0.3
# O mesmo serve para os resultados de ambos
x_train, x_test, y_train, y_test = train_test_split(previsores, alvo, random_state=0, test_size=0.3) 

naive = GaussianNB()

naive.fit(x_train, y_train)

# print("   ===  Previsões de teste ===  ") 
# previsoes_teste = naive.predict(x_test)

# print(f"x_train: {x_train.shape}")

# print(f"{previsoes_teste}\n")

# print(f"Y_TEST \n {y_test} \n")

# acuracia = accuracy_score(y_test, previsoes_teste) # acuracia da previsão

# print(f"Acurácia: {(acuracia*100):.2f}%  \n")

# matriz_confusao = confusion_matrix(y_test, previsoes_teste) # Matriz de confusão (onde a diagonal principal representa os acertos [0][0] e [1][1])

# print(f"Matriz de confusao \n {matriz_confusao} \n")

# report_classificacao = classification_report(y_test, previsoes_teste)

# print(f"Classificação report \n {report_classificacao} \n")


# print("   ===  Previsões de treino ===  ") 
# previsoes_treino = naive.predict(x_train)

# print(f"{previsoes_treino}\n")

# acuracia = accuracy_score(y_train, previsoes_treino) # acuracia da previsão

# print(f"Acurácia: {(acuracia*100):.2f}%  \n")

# matriz_confusao = confusion_matrix(y_train, previsoes_treino) # Matriz de confusão (onde a diagonal principal representa os acertos [0][0] e [1][1])

# print(f"Matriz de confusao \n {matriz_confusao} \n")

# report_classificacao = classification_report(y_train, previsoes_treino)

# print(f"Classificação report \n {report_classificacao} \n")



print("RESULTADOS GERADOS COM OS PREVISORES ESCALONADOS")
previsores_escalonados = StandardScaler().fit_transform(previsores)

x_train, x_test, y_train, y_test = train_test_split(previsores_escalonados, alvo, random_state=0, test_size=0.3) 

# print(f"x_train: {x_train.shape}")
print("as porcentagens resultantes seguem as mesmas para o exemplo")

print("RESULTADOS GERADOS COM AS VARIÁVEIS CATEGÓRICAS TRANSFORMADAS EM NUMERADAS PELO LABELENCODER")
previsores2 = dados_tratados.iloc[:, 0:11].values

previsores2[:, 1] = LabelEncoder().fit_transform(previsores[:, 1])
previsores2[:, 2] = LabelEncoder().fit_transform(previsores[:, 2])
previsores2[:, 6] = LabelEncoder().fit_transform(previsores[:, 6])
previsores2[:, 8] = LabelEncoder().fit_transform(previsores[:, 8])
previsores2[:, 10] = LabelEncoder().fit_transform(previsores[:, 10])

print("as porcentagens resultantes tem aumento nas casas centesimais")

print("RESULTADOS GERADOS COM AS VARIÁVEIS CATEGÓRICAS TRANSFORMADAS EM NUMERADAS PELO LABEL_ENCODER E HOT_ONE_ENCODER SEM ESCALONAR")

previsores3 = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 2, 6, 8, 10])], 
                                remainder='passthrough').fit_transform(previsores2)

x_train, x_test, y_train, y_test = train_test_split(previsores3, alvo, random_state=0, test_size=0.3) 

naive = GaussianNB()

naive.fit(x_train, y_train)

print("RESULTADOS GERADOS COM AS VARIÁVEIS CATEGÓRICAS TRANSFORMADAS EM NUMERADAS PELO LABEL_ENCODER E HOT_ONE_ENCODER E ESCALONADAS")

previsores3_escalonados = StandardScaler().fit_transform(previsores3)

x_train, x_test, y_train, y_test = train_test_split(previsores3_escalonados, alvo, random_state=0, test_size=0.3) 

naive = GaussianNB()

naive.fit(x_train, y_train)

previsoes_teste = naive.predict(x_test)

acuracia = accuracy_score(y_test, previsoes_teste) # acuracia da previsão

print(f"Acurácia: {(acuracia*100):.2f}%  \n")

matriz_confusao = confusion_matrix(y_test, previsoes_teste) # Matriz de confusão (onde a diagonal principal representa os acertos [0][0] e [1][1])

print(f"Matriz de confusao \n {matriz_confusao} \n")

report_classificacao = classification_report(y_test, previsoes_teste)

print(f"Classificação report \n {report_classificacao} \n")


'''
Validação Cruzada
Misturar os dados de treino e teste para saber se a separção feita dos dados foi feita da melhor maneira.
A validação cruzada (ou cross-validation) é uma técnica usada para avaliar o desempenho real de um modelo 
de forma mais justa e confiável.
Ela ajuda a garantir que o modelo não esteja apenas "decorando" os dados de treino, e sim generalizando bem para dados novos.
'''

kfold = KFold(n_splits=30, shuffle=True, random_state=5) # separando os dados em grupos

modelo = GaussianNB()

resultado = cross_val_score(modelo, previsores3_escalonados, alvo, cv=kfold)

print(f"Acurácia da Validação Cruzada: {(resultado.mean()*100):.2f}%  \n")

                            





