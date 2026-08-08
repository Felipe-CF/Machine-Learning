import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler


load_dotenv('C:\\Users\\FelipeCF\\Desktop\\Codigos\\Machine-Learning\\algoritmos\\.env')

dir = os.getenv('OUTPUT_DIR')

dados = pd.read_csv(os.path.join(dir, 'housing.csv'), delimiter=',', encoding='iso-8859-1')

dados_frame = pd.DataFrame(dados)

print(dados_frame.head(5))

independente = dados_frame.iloc[:, 0:3]

dependente = dados_frame.iloc[:, 3]

independente_esc = StandardScaler().fit_transform(independente)





