import pandas as pd
import lightgbm as lgb
from sklearn.svm import SVC
from datetime import datetime
from dados_base.data_d1 import *
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


x_train, x_test, y_train, y_test = train_test_split(previsores, alvo, test_size=0.3, random_state=0)


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

modelo = lgb.LGBMClassifier(num_leaves=200, objective= 'binary', max_depth=4,
                            learning_rate=  0.09, max_bin=200)

resultado = cross_val_score(modelo, previsores, alvo, cv=kfold)

print(f"Acurácia da Validação Cruzada: {(resultado.mean()*100):.2f}%  \n")
