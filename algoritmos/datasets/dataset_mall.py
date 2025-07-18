import os
import numpy as np
import warnings
import pandas as pd
from dotenv import load_dotenv


warnings.filterwarnings('ignore')

load_dotenv('C:\\Users\\FelipeCosta\\Desktop\\files\\Machine-Learning\\algoritmos\\.env')

dir = os.getenv('OUTPUT_DIR')

dados = pd.read_csv(os.path.join(dir, 'Mall_Customers.csv'), delimiter=',', encoding='iso-8859-1')

dados_frame = pd.DataFrame(dados)

dados_frame.rename(columns={'Genre': 'genero'}, inplace=True)
dados_frame.rename(columns={'CustomerID': 'cliente_id'}, inplace=True)
dados_frame.rename(columns={'Age': 'idade'}, inplace=True)
dados_frame.rename(columns={'Annual Income (k$)': 'rendimento_anual', 'Spending Score (1-100)': 'pontuacao'}, inplace=True)

print(dados_frame.head(5))
dados_frame['genero'].replace({
    'Female': 0,
    'Male': 1,
}, inplace=True)

