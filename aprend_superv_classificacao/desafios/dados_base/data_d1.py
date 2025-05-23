import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder



load_dotenv()

data_dir = os.getenv('DESAFIO_1')

dados = pd.read_csv(os.path.join(data_dir, 'data_cancer2.csv'), index_col=None)

dados_frame = pd.DataFrame.copy(dados)

dados_frame_catboost = pd.DataFrame.copy(dados)

print(dados_frame_catboost.head)

dados_frame = dados_frame.loc[:, ~dados_frame.columns.str.contains('^Unnamed')]

dados_frame['diagnosis'].replace(
    {
        'M': 1,
        'B': 0
    }, inplace=True
)

alvo = dados_frame.iloc[:, 1]

previsores = dados_frame.iloc[:, 2:32]

previsores_escalonados = StandardScaler().fit_transform(previsores)


