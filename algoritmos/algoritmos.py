import os 
import numpy as np
import pandas as pd
from bases.dados_tratados import *
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score  
from sklearn.metrics import mean_absolute_error, mean_squared_error


def catboost():
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

    
def xgboost():
    independentes = dados_tratados.iloc[:, 0:3].values
    dependente = dados_tratados.iloc[:, 3].values

    x_treino, x_teste, y_treino, y_teste = train_test_split(independentes, dependente, test_size=0.3, random_state=0)

    xgb = XGBRegressor(max_depth=3, random_state=10, n_estimators=100, objective='reg:squarederror', learning_rate=.09)

    xgb.fit(x_treino, y_treino)

    print(f'coef de determinação de treino: {xgb.score(x_treino, y_treino)*100.0:.2f}%\n')
    print(f'coef de determinação de teste: {xgb.score(x_teste, y_teste)*100.0:.2f}%\n')
    print(f'Erro absoluto {abs((y_teste - xgb.predict(x_teste))).mean():.4f}\n')
    print('Erro absoluto médio: {:.4f}\n'.format(mean_absolute_error(y_teste, xgb.predict(x_teste))))
    print('Erro quadratico médio: {:.4f}\n'.format(mean_squared_error(y_teste, xgb.predict(x_teste))))
    print('Raíz do erro quadratico médio: {:.4f}\n'.format(np.sqrt(mean_squared_error(y_teste, xgb.predict(x_teste)))))

    print('Validação cruzada')
    kfold = KFold(n_splits=15, shuffle=True, random_state=5)
    modelo =  XGBRegressor(max_depth=3, random_state=10, n_estimators=100, objective='reg:squarederror', learning_rate=.09)
    resultado = cross_val_score(modelo, independentes, dependente, cv=kfold)
    print(f'Coef. determinação médio: {resultado.mean()*100.0:.2f}%')


def lgbm():
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


def arvore_decisao():
    independentes = dados_tratados.iloc[:, 0:3].values
    dependente = dados_tratados.iloc[:, 3].values

    x_treino, x_teste, y_treino, y_teste = train_test_split(independentes, dependente, test_size=0.3, random_state=5)

    arvore = DecisionTreeRegressor(max_depth=4, random_state=5)

    arvore.fit(x_treino, y_treino)

    print(f'coef de determinação de treino: {arvore.score(x_treino, y_treino)*100.0:.2f}%\n')
    print(f'coef de determinação de teste: {arvore.score(x_teste, y_teste)*100.0:.2f}%\n')
    print(f'Erro absoluto {abs((y_teste - arvore.predict(x_teste))).mean():.4f}\n')
    print('Erro absoluto médio: {:.4f}\n'.format(mean_absolute_error(y_teste, arvore.predict(x_teste))))
    print('Erro quadratico médio: {:.4f}\n'.format(mean_squared_error(y_teste, arvore.predict(x_teste))))
    print('Raíz do erro quadratico médio: {:.4f}\n'.format(np.sqrt(mean_squared_error(y_teste, arvore.predict(x_teste)))))

    print('Validação cruzada')
    kfold = KFold(n_splits=15, shuffle=True, random_state=5)
    modelo = DecisionTreeRegressor(max_depth=4, random_state=5)
    resultado = cross_val_score(modelo, independentes, dependente, cv=kfold)
    print(f'Coef. determinação médio: {resultado.mean()*100.0:.2f}%')


def random_forest():
    independentes = dados_tratados.iloc[:, 0:3].values
    dependente = dados_tratados.iloc[:, 3].values

    x_treino, x_teste, y_treino, y_teste = train_test_split(independentes, dependente, test_size=0.3, random_state=0)

    random = RandomForestRegressor(max_depth=5, random_state=10, n_estimators=150, criterion='squared_error')

    random.fit(x_treino, y_treino)

    print(f'coef de determinação de treino: {random.score(x_treino, y_treino)*100.0:.2f}%\n')
    print(f'coef de determinação de teste: {random.score(x_teste, y_teste)*100.0:.2f}%\n')
    print(f'Erro absoluto {abs((y_teste - random.predict(x_teste))).mean():.4f}\n')
    print('Erro absoluto médio: {:.4f}\n'.format(mean_absolute_error(y_teste, random.predict(x_teste))))
    print('Erro quadratico médio: {:.4f}\n'.format(mean_squared_error(y_teste, random.predict(x_teste))))
    print('Raíz do erro quadratico médio: {:.4f}\n'.format(np.sqrt(mean_squared_error(y_teste, random.predict(x_teste)))))

    print('Validação cruzada')
    kfold = KFold(n_splits=15, shuffle=True, random_state=5)
    modelo = RandomForestRegressor(max_depth=5, random_state=10, n_estimators=150, criterion='squared_error')
    resultado = cross_val_score(modelo, independentes, dependente, cv=kfold)
    print(f'Coef. determinação médio: {resultado.mean()*100.0:.2f}%')


def regressao_linear():
    '''
        Equação = intercepto (coeficiente linear) + (coeficiente angular) * num de comodos
        reg_linear.intercept_
        reg_linear.coef_
    '''

    print('\nMEDV X LSTAT')

    reg_linear = LinearRegression()

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


def regressao_polinomial():
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


def svr():
    x_esc = StandardScaler()

    y_esc = StandardScaler()

    independentes = dados_tratados.iloc[:, 0:3].values
    dependentes = dados_tratados.iloc[:, 3].values

    x_treino, x_teste, y_treino, y_teste = train_test_split(independentes, dependentes, test_size=0.3, random_state=0)

    x_treino_esc = x_esc.fit_transform(x_treino)

    y_treino_esc = y_esc.fit_transform(y_treino.reshape(-1, 1))

    x_teste_esc = x_esc.fit_transform(x_teste)

    y_teste_esc = y_esc.fit_transform(y_teste.reshape(-1, 1))

    svr = SVR(kernel='rbf')
    svr.fit(x_treino_esc, y_treino_esc.ravel())

    print(f' score do treino: {svr.score(x_treino_esc, y_treino_esc)*100:.2f}%')

    print(f' score do teste: {svr.score(x_teste_esc, y_teste_esc)*100:.2f}%')

    prev_teste = svr.predict(x_teste_esc)
    prev_teste = y_esc.inverse_transform(prev_teste.reshape(-1, 1))

    y_teste = y_esc.inverse_transform(y_teste_esc)

    print(mean_absolute_error(y_teste, prev_teste))
    print(mean_squared_error(y_teste, prev_teste))

    # referencia para comparar com outros modelos
    print(np.sqrt(mean_squared_error(y_teste, prev_teste)))

    print('\n Validação cruzada')
    independentes_esc = StandardScaler().fit_transform(independentes)
    dependentes_esc = StandardScaler().fit_transform(dependentes.reshape(-1, 1))
    kfold = KFold(n_splits=15, shuffle=True, random_state=5)
    modelo = SVR(kernel='rbf')
    resultado = cross_val_score(modelo, independentes_esc, dependentes_esc.ravel(), cv=kfold)
    print(f'Coef. determinação médio: {resultado.mean()*100.0:.2f}%')


if __name__ == '__main__':
   
   while True:
       
       print('1 - xgboost')
       print('2 - lgbm')
       print('3 - random forest')
       print('4 - regressao linear')
       print('5 - regressao polinomial')
       print('6 - svr')
       print('7 - catboost')
       print('8 - arvore de decisão')
       print('9 - sair')

       i = int(input())

       if i == 1:
        xgboost()

       elif i == 2:
        lgbm()

       elif i == 3:
        random_forest()

       elif i == 4:
        regressao_linear()

       elif i == 5:
        regressao_polinomial()

       elif i == 6:
        svr()

       elif i == 7:
        catboost()

       elif i == 8:
        arvore_decisao()

       elif i == 9:
        break
    
