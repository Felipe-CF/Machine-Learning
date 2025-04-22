import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.express as px
from bases.dados_tratados import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


'''
Equação: o polinomial.coef_ gerar 3 valores que devem ser multiplicados, aqueles diferentes de 0, por RM e RM²

Resultados
    coef de determinação de treino: 58.65%
    coef de determinação de teste: 54.68%
    Erro absoluto médio
    87375.95297910202
    Erro quadrático médio
    13369416743.213114
    Raíz do erro quadrático médio
    115626.19401854025
'''
x_rm = dados.iloc[:, 0:1]

y = dados.iloc[:, 3]

x_train, x_test, y_train, y_test = train_test_split(x_rm, y, test_size=0.3, random_state=0)

# pré processamento
grau_polinomial = PolynomialFeatures(degree=2)

x_poly = grau_polinomial.fit_transform(x_train)

polinomial = LinearRegression()

polinomial.fit(x_poly, y_train)

prev = polinomial.predict(x_poly)

n = np.linspace(3, 9.84, 342)

equacao = 1640107 - 568528*n + 60092*pow(n, 2)

print(f'coef de determinação de treino: {polinomial.score(x_poly, y_train)*100:.2f}%')

x_poly_test = grau_polinomial.fit_transform(x_test)

polinomial = LinearRegression().fit(x_poly_test, y_test)

print(f'coef de determinação de teste: {polinomial.score(x_poly_test, y_test)*100:.2f}%')

equacao = 1640107 - 568528*x_test + 60092*pow(x_test, 2)

print('Erro absoluto médio')
print(mean_absolute_error(equacao, y_test))


print('Erro quadrático médio')
print(mean_squared_error(equacao, y_test))

# referencia para comparar com outros modelos
print('Raíz do erro quadrático médio')
print(np.sqrt(mean_squared_error(equacao, y_test)))


