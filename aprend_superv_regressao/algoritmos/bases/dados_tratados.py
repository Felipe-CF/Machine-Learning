import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


load_dotenv('C:\\Users\\FelipeCosta\\Desktop\\files\\Machine-Learning\\algoritmos\\bases\\.env')

dir = os.getenv('OUTPUT_DIR')

dados = pd.read_csv(os.path.join(dir, 'housing.csv'),
                    sep=',', encoding='iso-8859-1')

dados_tratados = pd.DataFrame.copy(dados)

previsores = dados_tratados.iloc[:, :3].values

alvo = dados_tratados.iloc[:, 3].values

x_train, x_test, y_train, y_test = train_test_split(previsores, alvo, test_size=0.3, random_state=0)

independente = dados_tratados.iloc[:, 0:3]

dependente = dados_tratados.iloc[:, 3]

np.savetxt(os.path.join(dir, 'independente.csv'), independente,  delimiter=',')

np.savetxt(os.path.join(dir, 'dependente.csv'), dependente,  delimiter=',')
