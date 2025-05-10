import lightgbm as lgb
from datetime import datetime
from bases.dados.dados_tratados_esc import *
from bases.base_treino_teste import *
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

lgb_dataset = lgb.Dataset(x_train, label=y_train)

parametros = {
    'num_leaves': 150, 
    'objective':'binary',
    'max_depth': 2,
    'learning_rate': 0.05,
    'max_bin': 200
}

inicio = datetime.now()
lgbm = lgb.train(params=parametros, train_set=lgb_dataset, num_boost_round=150)
fim = datetime.now()

tempo = fim - inicio

'''
Se o resultado da previsao for "> 0.5" a condição será True, em seguida
True é convertido para inteiro, 1.
    for i in range(0, 276):
        if previsoes_lgbm[i] >= .5:
            previsoes_lgbm[i] = 1
        else:
            previsoes_lgbm[i] = 0
'''
previsoes_lgbm = (lgbm.predict(x_test) > 0.5).astype(int)
    
print("   ===  Previsões de teste ===  ") 
acuracia = accuracy_score(y_test, previsoes_lgbm) # acuracia da previsão

print(f"Acurácia: {(acuracia*100):.2f}%  \n")

matriz_confusao = confusion_matrix(y_test, previsoes_lgbm) # Matriz de confusão (onde a diagonal principal representa os acertos [0][0] e [1][1])

print(f"Matriz de confusao \n {matriz_confusao} \n")

report_classificacao = classification_report(y_test, previsoes_lgbm)

print(f"Classificação report \n {report_classificacao} \n")


print("   ===  Previsões de treino ===  ") 
previsoes_lgbm = (lgbm.predict(x_train) > 0.5).astype(int)

acuracia = accuracy_score(y_train, previsoes_lgbm) # acuracia da previsão

print(f"Acurácia: {(acuracia*100):.2f}%  \n")

matriz_confusao = confusion_matrix(y_train, previsoes_lgbm) # Matriz de confusão (onde a diagonal principal representa os acertos [0][0] e [1][1])

print(f"Matriz de confusao \n {matriz_confusao} \n")

report_classificacao = classification_report(y_train, previsoes_lgbm)

print(f"Classificação report \n {report_classificacao} \n")



kfold = KFold(n_splits=30, shuffle=True, random_state=5) # separando os dados em grupos

modelo = lgb.LGBMClassifier(num_leaves=150, objective= 'binary', max_depth=  2,
                            learning_rate=  0.05, max_bin=  200)

resultado = cross_val_score(modelo, previsores, alvo, cv=kfold)

print(f"Acurácia da Validação Cruzada: {(resultado.mean()*100):.2f}%  \n")




