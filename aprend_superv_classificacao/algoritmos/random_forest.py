import numpy as np
from sklearn.ensemble import RandomForestClassifier
from bases.dados.dados_tratados_esc import *
from bases.base_treino_teste import *
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


rand_forest = RandomForestClassifier(criterion='entropy', random_state=0, n_estimators=150, max_depth=4)

rand_forest.fit(x_train, y_train)

previsoes_rand_forest = rand_forest.predict(x_test)

print("   ===  Previsões de teste ===  ") 
acuracia = accuracy_score(y_test, previsoes_rand_forest) # acuracia da previsão

print(f"Acurácia: {(acuracia*100):.2f}%  \n")

matriz_confusao = confusion_matrix(y_test, previsoes_rand_forest) # Matriz de confusão (onde a diagonal principal representa os acertos [0][0] e [1][1])

print(f"Matriz de confusao \n {matriz_confusao} \n")

report_classificacao = classification_report(y_test, previsoes_rand_forest)

print(f"Classificação report \n {report_classificacao} \n")


print("   ===  Previsões de treino ===  ") 
previsoes_rand_forest = rand_forest.predict(x_train)

# print(f"{previsoes_rand_forest}\n")

acuracia = accuracy_score(y_train, previsoes_rand_forest) # acuracia da previsão

print(f"Acurácia: {(acuracia*100):.2f}%  \n")

matriz_confusao = confusion_matrix(y_train, previsoes_rand_forest) # Matriz de confusão (onde a diagonal principal representa os acertos [0][0] e [1][1])

print(f"Matriz de confusao \n {matriz_confusao} \n")

report_classificacao = classification_report(y_train, previsoes_rand_forest)

print(f"Classificação report \n {report_classificacao} \n")



kfold = KFold(n_splits=30, shuffle=True, random_state=5) # separando os dados em grupos

modelo = RandomForestClassifier(n_estimators=150, criterion='entropy', random_state=0, max_depth=4)

resultado = cross_val_score(modelo, previsores, alvo, cv=kfold)

print(f"Acurácia da Validação Cruzada: {(resultado.mean()*100):.2f}%  \n")




