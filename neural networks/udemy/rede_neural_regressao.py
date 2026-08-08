import numpy as np
from dataset_housing import *
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


'''
n° de neurônios por camada oculta: 
    N = (nE + nS) / 2 = (11+1)/2 = 6 
    N = (2 * nE) / 3  + nS = (11*2)/3 + 1 = 8 

parâmetros:
    solver = algortimo matemático
        'adam': datasets > 1000 amostras
        'sgd': com descida do gradiente estocástico
        'lbfgs': datasets pequenos

    alpha = regularização dos ajustes do peso (inversamente proporcional ao valor dos pesos) default = .0001

    learning_rate = taxa de aprendizagem
        'constant'
        'invscalling': diminuição gradativa
        'adaptive': diminiu em 5x a cada 2 épocas consectuvias que não diminuíram o erro

    shuffle = embaralha a sequencia de dados (solver 'adam' ou 'sgd')

    momentum = otimizar o 'sgd' default 0.9

    n_iter_no_change = máximo de épocas sem alcançar tolerãncia de melhoria (default=10, 'sgd', 'adam')

resultado:
    Análise de treino
        85.34%

    Análise de testes
        80.63%

    Validação cruzada - Acurácia média: 75.94%   
'''
x_treino, x_teste, y_treino, y_teste = train_test_split(independente, dependente, test_size=0.3, random_state=0)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_treino_esc = x_scaler.fit_transform(x_treino)

y_treino_esc = y_scaler.fit_transform(np.array(y_treino).reshape(-1, 1))

x_teste_esc = x_scaler.fit_transform(x_teste)

y_teste_esc = y_scaler.fit_transform(np.array(y_teste).reshape(-1, 1))

rede_neural = MLPRegressor(hidden_layer_sizes=(6, 6), activation='relu', solver='lbfgs', max_iter=1500,
                            random_state=12, verbose=True)

rede_neural.fit(x_treino_esc, y_treino_esc.ravel())

previsoes = rede_neural.predict(x_treino)

acuracia_treino = rede_neural.score(x_treino_esc, y_treino_esc)

previsoes = rede_neural.predict(x_teste_esc)
acuracia_teste = rede_neural.score(x_teste_esc, y_teste_esc)

kfold = KFold(n_splits=30, shuffle=True, random_state=5)
modelo = MLPRegressor(hidden_layer_sizes=(6, 6), activation='relu', solver='lbfgs', max_iter=1500,
                            random_state=12, verbose=True)
resultado = cross_val_score(modelo, independente, dependente, cv=kfold)

print('\nAnálise de treino')
print(acuracia_treino)

print('\nAnálise de testes')
print(acuracia_teste)

print(f'Validação cruzada - Acurácia média: {resultado.mean()*100:.2f}%\n')

print(f'Erro absoluto {abs((y_teste - rede_neural.predict(x_teste_esc))).mean():.4f}\n')
print('Erro absoluto médio: {:.4f}\n'.format(mean_absolute_error(y_teste, rede_neural.predict(x_teste_esc))))
print('Erro quadratico médio: {:.4f}\n'.format(mean_squared_error(y_teste, rede_neural.predict(x_teste_esc))))
print('Raíz do erro quadratico médio: {:.4f}\n'.format(np.sqrt(mean_squared_error(y_teste, rede_neural.predict(x_teste_esc)))))
