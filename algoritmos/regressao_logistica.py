import numpy as np
from sklearn.linear_model import LogisticRegression
from bases.dados.dados_tratados_esc import *
from bases.base_treino_teste import *
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


logistica = LogisticRegression(random_state=1,max_iter=100,penalty='l2', tol=0.0001, 
                                 C=1,solver='lbfgs')

logistica.fit(x_train, y_train)

print(logistica.intercept_)
print(logistica.coef_)

previsoes_logistica = logistica.predict(x_test)

print("   ===  Previsões de teste ===  ") 
acuracia = accuracy_score(y_test, previsoes_logistica) # acuracia da previsão

print(f"Acurácia: {(acuracia*100):.2f}%  \n")

matriz_confusao = confusion_matrix(y_test, previsoes_logistica) # Matriz de confusão (onde a diagonal principal representa os acertos [0][0] e [1][1])

print(f"Matriz de confusao \n {matriz_confusao} \n")

report_classificacao = classification_report(y_test, previsoes_logistica)

print(f"Classificação report \n {report_classificacao} \n")


print("   ===  Previsões de treino ===  ") 
previsoes_logistica = logistica.predict(x_train)

# print(f"{previsoes_logistica}\n")

acuracia = accuracy_score(y_train, previsoes_logistica) # acuracia da previsão

print(f"Acurácia: {(acuracia*100):.2f}%  \n")

matriz_confusao = confusion_matrix(y_train, previsoes_logistica) # Matriz de confusão (onde a diagonal principal representa os acertos [0][0] e [1][1])

print(f"Matriz de confusao \n {matriz_confusao} \n")

report_classificacao = classification_report(y_train, previsoes_logistica)

print(f"Classificação report \n {report_classificacao} \n")



kfold = KFold(n_splits=30, shuffle=True, random_state=5) # separando os dados em grupos

modelo = LogisticRegression(random_state=1)

resultado = cross_val_score(modelo, previsores_dummy_escalonados, alvo, cv=kfold)

print(f"Acurácia da Validação Cruzada: {(resultado.mean()*100):.2f}%  \n")




