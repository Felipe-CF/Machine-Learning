from dataset_cancer import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


'''
n° de neurônios por camada oculta: 
    N = (nE + nS) / 2 = (30+1)/2 = 15,5 = 16 
    N = (2 * nE) / 3  + nS = 30*2/3 + 1 = 21 
    nMed = (16 + 21) / 2 = 19 

resultado:
    Acurácia de treino: 98.74%

    Acurácia de teste: 97.66%

    Validação cruzada - Acurácia média: 97.36%  
'''

x_treino, x_teste, y_treino, y_teste = train_test_split(previsores_esc, alvo, test_size=0.3, random_state=0)

rede_neural = MLPClassifier(hidden_layer_sizes=(20), activation='tanh', solver='adam', max_iter=800,
                            tol=.0001, random_state=3, verbose=True, alpha=.0004)

rede_neural.fit(x_treino, y_treino)

print('\nAnálise de treino')
previsoes = rede_neural.predict(x_treino)
acuracia_treino = accuracy_score(y_treino, previsoes)
print(f'Classficação:\n {classification_report(y_treino, previsoes)}')
print(f'Matriz de confusão:\n {confusion_matrix(y_treino, previsoes)}')

print('\nAnálise de testes')
previsoes = rede_neural.predict(x_teste)

acuracia_teste = accuracy_score(y_teste, previsoes)

print(f'Classficação:\n {classification_report(y_teste, previsoes)}')
print(f'Matriz de confusão:\n {confusion_matrix(y_teste, previsoes)}')


kfold = KFold(n_splits=30, shuffle=True, random_state=5)
modelo = MLPClassifier(hidden_layer_sizes=(20), activation='tanh', solver='adam', max_iter=800,
                            tol=.0001, random_state=3, verbose=True, alpha=.0004)

resultado = cross_val_score(modelo, previsores_esc, alvo, cv=kfold)


print(f'\nAcurácia de treino: {acuracia_treino*100:.2f}%\n')
print(f'Acurácia de teste: {acuracia_teste*100:.2f}%\n')
print(f'Validação cruzada - Acurácia média: {resultado.mean()*100:.2f}%\n')





