import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from bases.dados_tratados import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

'''
MEDV X RM
'''
x_rm = dados_tratados.iloc[:, 0:1]
y = dados_tratados.iloc[:, 3]

x_treino, x_teste, y_treino, y_teste = train_test_split(x_rm, y, test_size=0.3, random_state=10)

reg_linear = LinearRegression()

reg_linear.fit(x_treino, y_treino)

'''
Equação = intercepto (coeficiente linear) + (coeficiente angular) * num de comodos
reg_linear.intercept_
reg_linear.coef_
'''

print(f'coef de determinação de treino: {reg_linear.score(x_treino, y_treino)}')
print(f'coef de determinação de teste: {reg_linear.score(x_teste, y_teste)}')

# print(f'prev de treino: {reg_linear.predict(x_treino)}')
# print(f'prev de teste: {reg_linear.predict(x_teste)}')

# fazendo prev's com valores distintos
# prev_casa = reg_linear.predict([[4]])
# print(prev_casa)

print(mean_absolute_error(y_teste, reg_linear.predict(x_teste)))
print(mean_squared_error(y_teste, reg_linear.predict(x_teste)))

# referencia para comparar com outros modelos
print(np.sqrt(mean_squared_error(y_teste, reg_linear.predict(x_teste))))


print('\nMEDV X LSTAT (melhor)')

x_lstat = dados_tratados.iloc[:, 1:2]

x_treino, x_teste, y_treino, y_teste = train_test_split(x_lstat, y, test_size=0.3, random_state=10)

reg_linear.fit(x_treino, y_treino)

print(f'coef de determinação de treino: {reg_linear.score(x_treino, y_treino)}')
print(f'coef de determinação de teste: {reg_linear.score(x_teste, y_teste)}')

print(f'Erro absoluto {abs((y_teste - reg_linear.predict(x_teste))).mean()}')

print(mean_absolute_error(y_teste, reg_linear.predict(x_teste)))
print(mean_squared_error(y_teste, reg_linear.predict(x_teste)))
print(np.sqrt(mean_squared_error(y_teste, reg_linear.predict(x_teste))))


print('\n Validação cruzada')

kfold = KFold(n_splits=15, shuffle=True, random_state=5)
modelo = LinearRegression()
resultado = cross_val_score(modelo, x_lstat, y, cv=kfold)
print(f'Coef. determinação médio: {resultado.mean()*100.0:.2f}%')