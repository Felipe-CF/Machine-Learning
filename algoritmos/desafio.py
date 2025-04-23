import os 
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# acessando o dataset
load_dotenv()

archive_dir = os.getenv('OUTPUT_DIR')

dados = pd.read_csv(os.path.join(archive_dir, 'insurance.csv'), delimiter=',',
                    encoding='iso-8859-1')

dados_frame = pd.DataFrame(dados)

# pré-processamento dos dados I
'''
    Devolve todos os valores possíveis da coluna
        print(dados_frame['coluna'].unique())

    'bmi' = indice de massa corporal, sendo o ideal entre 18.5 a 24.9

    'children' = quantidade de dependentes

    'smoker' = fumante, ou não
'''

dados_frame['sex'].replace({
    'female': 0,
    'male': 1,

}, inplace=True)

dados_frame['smoker'].replace({
    'no': 0,
    'yes': 1,
}, inplace=True)

dados_frame['region'].replace({
    'southwest': 0,
    'southeast': 1,
    'northwest': 2,
    'northeast': 3
}, inplace=True)

# pré-processamento dos dados II
previsores = dados_frame.iloc[:, 0:6]

alvo = dados_frame.iloc[:, 6]

previsores_dummy = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [5])],
                                     remainder='passthrough').fit_transform(previsores)

previsores_esc = StandardScaler().fit_transform(previsores_dummy)

# separação de dados para teste e treino
x_treino, x_teste, y_treino, y_teste = train_test_split(previsores_esc, alvo, test_size=0.3, random_state=0)

# criação do modelo de previsão
random = RandomForestRegressor(max_depth=5, n_estimators=150, criterion='squared_error', random_state=10)

# treinamento
random.fit(x_treino, y_treino)

# resultados 
print(f'coef de determinação de treino: {random.score(x_treino, y_treino)*100.0:.2f}%\n')
print(f'coef de determinação de teste: {random.score(x_teste, y_teste)*100.0:.2f}%\n')
print(f'Erro absoluto {abs((y_teste - random.predict(x_teste))).mean():.4f}\n')
print('Erro absoluto médio: {:.4f}\n'.format(mean_absolute_error(y_teste, random.predict(x_teste))))
print('Erro quadratico médio: {:.4f}\n'.format(mean_squared_error(y_teste, random.predict(x_teste))))
print('Raíz do erro quadratico médio: {:.4f}\n'.format(np.sqrt(mean_squared_error(y_teste, random.predict(x_teste)))))

print('Validação cruzada')
kfold = KFold(n_splits=15, shuffle=True, random_state=5)
modelo = RandomForestRegressor(max_depth=5, random_state=10, n_estimators=150, criterion='squared_error')
resultado = cross_val_score(modelo, previsores_dummy, alvo, cv=kfold)
print(f'Coef. determinação médio: {resultado.mean()*100.0:.2f}%')



