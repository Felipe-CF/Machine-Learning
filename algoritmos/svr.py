import numpy as np
from bases.dados_tratados import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

'''
Devido a diferença dos valores das colunas de previsores, é necessário realizar um escalonamento dos valores.
 score do treino: 87.15%
 score do teste: 81.88%
53973.72694080933
5256511310.011197
72501.80211561087

 Validação cruzada
Coef. determinação médio: 82.37%
'''
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