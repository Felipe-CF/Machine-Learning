from datetime import datetime
from catboost import CatBoostClassifier
from bases.dados.dados_tratados_esc import *
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



previsores_catboost = dados.iloc[:, 0:11]

alvo_catboost = dados.iloc[:, 11]

x_train, x_test, y_train, y_test = train_test_split(previsores_catboost, alvo_catboost, test_size=0.3, random_state=0)

variaveis_categoricas = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

catboost_class = CatBoostClassifier(task_type='CPU', iterations=100, learning_rate=.1, depth=8, 
                                random_state=5, eval_metric='Accuracy')

catboost_class.fit(x_train, y_train, cat_features=variaveis_categoricas, plot=False, eval_set=(x_test, y_test))

'''
plot=True

A biblioteca traitlets deve está instalada, e o CatBoost precisa dela para mostrar os gráficos de treino.

pip install traitlets ipython ipywidgets

Isso é mostrando independente do valor de Plot
bestTest = 0.8695652174
bestIteration = 54
'''

previsoes_cat = catboost_class.predict(x_test)

print("   ===  Previsões de teste ===  ") 
acuracia = accuracy_score(y_test, previsoes_cat) # acuracia da previsão

print(f"Acurácia: {(acuracia*100):.2f}%  \n")

matriz_confusao = confusion_matrix(y_test, previsoes_cat) # Matriz de confusão (onde a diagonal principal representa os acertos [0][0] e [1][1])

print(f"Matriz de confusao \n {matriz_confusao} \n")

report_classificacao = classification_report(y_test, previsoes_cat)

print(f"Classificação report \n {report_classificacao} \n")


print("   ===  Previsões de treino ===  ") 
previsoes_cat = (catboost_class.predict(x_train) > 0.5).astype(int)

acuracia = accuracy_score(y_train, previsoes_cat) # acuracia da previsão

print(f"Acurácia: {(acuracia*100):.2f}%  \n")

matriz_confusao = confusion_matrix(y_train, previsoes_cat) # Matriz de confusão (onde a diagonal principal representa os acertos [0][0] e [1][1])

print(f"Matriz de confusao \n {matriz_confusao} \n")

report_classificacao = classification_report(y_train, previsoes_cat)

print(f"Classificação report \n {report_classificacao} \n")



kfold = KFold(n_splits=30, shuffle=True, random_state=5) # separando os dados em grupos

modelo = CatBoostClassifier(task_type='CPU', iterations=100, learning_rate=.1, depth=8, 
                                random_state=5, eval_metric='Accuracy')

resultado = cross_val_score(modelo, previsores, alvo, cv=kfold)

print(f"Acurácia da Validação Cruzada: {(resultado.mean()*100):.2f}%  \n")





