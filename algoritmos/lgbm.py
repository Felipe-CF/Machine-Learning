import numpy as np
from bases.dados_tratados import *
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

'''
coef de determinação de treino: 88.15%

coef de determinação de teste: 82.18%

Erro absoluto 55114.0931

Erro absoluto médio: 55114.0931

Erro quadratico médio: 5170544146.4396

Raíz do erro quadratico médio: 71906.4959

Validação cruzada
Coef. determinação médio: 82.26%
'''

independentes = dados_tratados.iloc[:, 0:3].values
dependente = dados_tratados.iloc[:, 3].values

x_treino, x_teste, y_treino, y_teste = train_test_split(independentes, dependente, test_size=0.3, random_state=0)

lgbm = lgb.LGBMRegressor(num_leaves=50, random_state=10, n_estimators=50, max_depth=3, learn8ing_rate=.09)

lgbm.fit(x_treino, y_treino)

print(f'coef de determinação de treino: {lgbm.score(x_treino, y_treino)*100.0:.2f}%\n')
print(f'coef de determinação de teste: {lgbm.score(x_teste, y_teste)*100.0:.2f}%\n')
print(f'Erro absoluto {abs((y_teste - lgbm.predict(x_teste))).mean():.4f}\n')
print('Erro absoluto médio: {:.4f}\n'.format(mean_absolute_error(y_teste, lgbm.predict(x_teste))))
print('Erro quadratico médio: {:.4f}\n'.format(mean_squared_error(y_teste, lgbm.predict(x_teste))))
print('Raíz do erro quadratico médio: {:.4f}\n'.format(np.sqrt(mean_squared_error(y_teste, lgbm.predict(x_teste)))))

print('Validação cruzada')
kfold = KFold(n_splits=15, shuffle=True, random_state=5)
modelo =  lgb.LGBMRegressor(num_leaves=50, random_state=10, n_estimators=50, max_depth=3, learning_rate=.09)
resultado = cross_val_score(modelo, independentes, dependente, cv=kfold)
print(f'Coef. determinação médio: {resultado.mean()*100.0:.2f}%')


