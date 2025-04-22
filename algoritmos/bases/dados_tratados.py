import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


load_dotenv()

output_dir = os.getenv('OUTPUT_DIR')

dados = pd.read_csv(os.path.join(output_dir, 'housing.csv'),
                    sep=',', encoding='iso-8859-1')

dados_tratados = pd.DataFrame.copy(dados)

previsores = dados_tratados.iloc[:, :3].values

alvo = dados_tratados.iloc[:, 3].values

x_train, x_test, y_train, y_test = train_test_split(previsores, alvo, test_size=0.3, random_state=0)
