import os
import numpy as np
import warnings
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore')

load_dotenv('C:\\Users\\FelipeCF\\Desktop\\Codigos\\Machine-Learning\\algoritmos\\.env')

dir = os.getenv('OUTPUT_DIR')

dados = pd.read_csv(os.path.join(dir, 'wine.csv'), delimiter=',', encoding='iso-8859-1')

dados_frame = pd.DataFrame(dados)

scaler = StandardScaler()

dados_frame_esc = scaler.fit_transform(dados_frame)

