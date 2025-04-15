import lightgbm as lgb
from sklearn.svm import SVC
from dados_base.data_d1 import *
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

'''
Dataset referente diagnóstico de tumor benigno/maligno de mama

===> Comparativos dos 5 melhores resultados de teste

| Algoritmo      | Acc (%) | Prec                | Recall              | F1                  | F1-macro | F1-wgt | Val. Cruz.  |
|----------------|---------|---------------------|---------------------|---------------------|----------|--------|-------------|
| LightGBM       | 98.83   | 0.99(0), 0.98(1)    | 0.99(0), 0.98(1)    | 0.99(0), 0.98(1)    | 0.99     | 0.99   | 96.82%      |
| XGBoost        | 97.66   | 0.97(0), 0.98(1)    | 0.99(0), 0.95(1)    | 0.98(0), 0.97(1)    | 0.97     | 0.98   | 97.00%      |
| CatBoost       | 97.08   | 0.97(0), 0.97(1)    | 0.98(0), 0.95(1)    | 0.98(0), 0.96(1)    | 0.97     | 0.97   | 97.16%      |
| Random Forest  | 96.49   | 0.96(0), 0.97(1)    | 0.98(0), 0.94(1)    | 0.97(0), 0.95(1)    | 0.96     | 0.96   | 95.76%      |
| SVM            | 95.32   | 0.95(0), 0.97(1)    | 0.98(0), 0.90(1)    | 0.96(0), 0.93(1)    | 0.94     | 0.95   | 94.88       |
'''

def alg_lgbm():
    x_train, x_test, y_train, y_test = train_test_split(previsores_escalonados, alvo, test_size=0.3, random_state=0)


    lgb_dataset = lgb.Dataset(x_train, label=y_train)

    parametros = {
        'num_leaves': 200, 
        'objective':'binary',
        'max_depth': 4,
        'learning_rate': 0.9,
        'max_bin': 200
    }

    lgbm = lgb.train(params=parametros, train_set=lgb_dataset, num_boost_round=150)

    previsoes_lgbm = (lgbm.predict(x_test) > 0.5).astype(int)

    print("   ===  Previsões de teste ===  ") 
    acuracia = accuracy_score(y_test, previsoes_lgbm)

    print(f"Acurácia: {(acuracia*100):.2f}%  \n")

    matriz_confusao = confusion_matrix(y_test, previsoes_lgbm)

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


    print("   ===   Validação Cruzada ===  ") 
    # kfold = KFold(n_splits=30, shuffle=True, random_state=5) # separando os dados em grupos

    # modelo = lgb.LGBMClassifier(num_leaves=200, objective= 'binary', max_depth=4,
    #                             learning_rate=  0.09, max_bin=200)

    # resultado = cross_val_score(modelo, previsores, alvo, cv=kfold)

    # print(f"Acurácia: {(resultado.mean()*100):.2f}%  \n")


def alg_xgboost():
    x_train, x_test, y_train, y_test = train_test_split(previsores, alvo, test_size=0.3, random_state=0)

    xgboost = XGBClassifier(max_depth=2, learning_rate=0.09, n_estimators=200, objective='binary:logistic', random_state=3)

    xgboost.fit(x_train, y_train)

    previsoes_xgboost = xgboost.predict(x_test)

    print("   ===  Previsões de teste ===  ") 
    acuracia = accuracy_score(y_test, previsoes_xgboost) # acuracia da previsão

    print(f"Acurácia: {(acuracia*100):.2f}%  \n")

    matriz_confusao = confusion_matrix(y_test, previsoes_xgboost) # Matriz de confusão (onde a diagonal principal representa os acertos [0][0] e [1][1])

    print(f"Matriz de confusao \n {matriz_confusao} \n")

    report_classificacao = classification_report(y_test, previsoes_xgboost)

    print(f"Classificação report \n {report_classificacao} \n")


    print("   ===  Previsões de treino ===  ") 
    previsoes_xgboost = xgboost.predict(x_train)

    # print(f"{previsoes_xgboost}\n")

    acuracia = accuracy_score(y_train, previsoes_xgboost) # acuracia da previsão

    print(f"Acurácia: {(acuracia*100):.2f}%  \n")

    matriz_confusao = confusion_matrix(y_train, previsoes_xgboost) # Matriz de confusão (onde a diagonal principal representa os acertos [0][0] e [1][1])

    print(f"Matriz de confusao \n {matriz_confusao} \n")

    report_classificacao = classification_report(y_train, previsoes_xgboost)

    print(f"Classificação report \n {report_classificacao} \n")



    kfold = KFold(n_splits=30, shuffle=True, random_state=5) # separando os dados em grupos

    modelo = XGBClassifier(max_depth=2, learning_rate=0.09, n_estimators=210, objective='binary:logistic', random_state=3)

    resultado = cross_val_score(modelo, previsores, alvo, cv=kfold)

    print(f"Acurácia da Validação Cruzada: {(resultado.mean()*100):.2f}%  \n")


def alg_svm():
    x_train, x_test, y_train, y_test = train_test_split(previsores, alvo, test_size=0.3, random_state=0)

    svm = SVC(kernel='rbf', random_state=1, C=5)

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

    kfold = KFold(n_splits=30, shuffle=True, random_state=5) # separando os dados em grupos

    modelo = SVC(kernel='linear', random_state=1, C=5)

    resultado = cross_val_score(modelo, previsores, alvo, cv=kfold)

    print(f"Acurácia da Validação Cruzada: {(resultado.mean()*100):.2f}%  \n")


def alg_catboost():
    previsores_catboost = dados_frame.iloc[:, 2:32]

    alvo_catboost = dados_frame.iloc[:, 1]

    x_train, x_test, y_train, y_test = train_test_split(previsores_catboost, alvo_catboost, test_size=0.3, random_state=0)

    catboost_class = CatBoostClassifier(task_type='CPU', iterations=100, learning_rate=.1, depth=8, 
                                    random_state=5, eval_metric='Accuracy')

    catboost_class.fit(x_train, y_train, plot=False, eval_set=(x_test, y_test))

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


def alg_rand_forest():
    x_train, x_test, y_train, y_test = train_test_split(previsores_escalonados, alvo, random_state=0, test_size=0.3) 

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


if __name__ == '__main__':

    alg_lgbm()

    # alg_xgboost()

    # alg_svm()

    # alg_catboost()

    # alg_rand_forest()
