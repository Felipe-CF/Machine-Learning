import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler


load_dotenv('C:\\Users\\FelipeCF\\Desktop\\Codigos\\Machine-Learning\\algoritmos\\.env')

dir = os.getenv('OUTPUT_DIR')

dados = pd.read_csv(os.path.join(dir, 'data_cancer2.csv'), delimiter=',', encoding='iso-8859-1', index_col=0)

dados_frame = pd.DataFrame(dados)

dados_frame = dados_frame.loc[:, ~dados_frame.columns.str.contains('^Unnamed')]

dados_frame['diagnosis'].replace({
    'B': 0,
    'M': 1
}, inplace=True)

alvo = dados_frame.iloc[:, 0:1]

previsores = dados_frame.iloc[:, 1:31]

previsores_esc = StandardScaler().fit_transform(previsores)



