import numpy as np
from sklearn.svm import SVC
from bases.dados.dados_tratados_esc import *
from bases.base_treino_teste import *
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

'''
Kernels e resultados
    ==> o melhor foi o linear com o C = 30
    elevei o c até 10000 e os valores não mudaram

    ==> ele tende a um overfitting quando "rbf", mas com C=2 temos 92% de treino,
        86% de teste, sendo a validação cruzada de 86%

'''

svm = SVC(kernel='rbf', random_state=1, C=2)

svm.fit(x_train, y_train)

previsoes_svm = svm.predict(x_test)

print("   ===  Previsões de teste ===  ") 
acuracia = accuracy_score(y_test, previsoes_svm) # acuracia da previsão

print(f"Acurácia: {(acuracia*100):.2f}%  \n")

matriz_confusao = confusion_matrix(y_test, previsoes_svm) # Matriz de confusão (onde a diagonal principal representa os acertos [0][0] e [1][1])

print(f"Matriz de confusao \n {matriz_confusao} \n")

report_classificacao = classification_report(y_test, previsoes_svm)

print(f"Classificação report \n {report_classificacao} \n")


print("   ===  Previsões de treino ===  ") 
previsoes_svm = svm.predict(x_train)

# print(f"{previsoes_svm}\n")

acuracia = accuracy_score(y_train, previsoes_svm) # acuracia da previsão

print(f"Acurácia: {(acuracia*100):.2f}%  \n")

matriz_confusao = confusion_matrix(y_train, previsoes_svm) # Matriz de confusão (onde a diagonal principal representa os acertos [0][0] e [1][1])

print(f"Matriz de confusao \n {matriz_confusao} \n")

report_classificacao = classification_report(y_train, previsoes_svm)

print(f"Classificação report \n {report_classificacao} \n")


'''
Validação Cruzada
Misturar os dados de treino e teste para saber se a separção feita dos dados foi feita da melhor maneira.
A validação cruzada (ou cross-validation) é uma técnica usada para avaliar o desempenho real de um modelo 
de forma mais justa e confiável.
Ela ajuda a garantir que o modelo não esteja apenas "decorando" os dados de treino, e sim generalizando bem para dados novos.
'''

kfold = KFold(n_splits=30, shuffle=True, random_state=5) # separando os dados em grupos

modelo = SVC(kernel='linear', random_state=1, C=2)

resultado = cross_val_score(modelo, previsores_dummy, alvo, cv=kfold)

print(f"Acurácia da Validação Cruzada: {(resultado.mean()*100):.2f}%  \n")





