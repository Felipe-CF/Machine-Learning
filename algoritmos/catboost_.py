import numpy as np
from bases.dados_tratados import *
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

'''
coef de determinação de treino: 91.69%

coef de determinação de teste: 83.75%

Raíz do erro quadratico médio: 68660.7963

Validação cruzada
Coef. determinação médio: 83.29%
'''

independentes = dados_tratados.iloc[:, 0:3].values
dependente = dados_tratados.iloc[:, 3].values

x_treino, x_teste, y_treino, y_teste = train_test_split(independentes, dependente, test_size=0.3, random_state=0)

catboost = CatBoostRegressor(max_depth=3, random_state=10, iterations=250, learning_rate=.09)

catboost.fit(x_treino, y_treino)

print(f'coef de determinação de treino: {catboost.score(x_treino, y_treino)*100.0:.2f}%\n')
print(f'coef de determinação de teste: {catboost.score(x_teste, y_teste)*100.0:.2f}%\n')
print(f'Erro absoluto {abs((y_teste - catboost.predict(x_teste))).mean():.4f}\n')
print('Erro absoluto médio: {:.4f}\n'.format(mean_absolute_error(y_teste, catboost.predict(x_teste))))
print('Erro quadratico médio: {:.4f}\n'.format(mean_squared_error(y_teste, catboost.predict(x_teste))))
print('Raíz do erro quadratico médio: {:.4f}\n'.format(np.sqrt(mean_squared_error(y_teste, catboost.predict(x_teste)))))

print('Validação cruzada')
kfold = KFold(n_splits=15, shuffle=True, random_state=5)
modelo =  CatBoostRegressor(max_depth=3, random_state=10, iterations=250, learning_rate=.09)
resultado = cross_val_score(modelo, independentes, dependente, cv=kfold)
print(f'Coef. determinação médio: {resultado.mean()*100.0:.2f}%')


