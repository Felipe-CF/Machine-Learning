import os 
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score  
from sklearn.metrics import mean_absolute_error, mean_squared_error


load_dotenv()

archive_dir = os.getenv('OUTPUT_DIR')

dados = pd.read_csv(os.path.join(archive_dir, 'insurance.csv'), delimiter=',',
                    encoding='iso-8859-1')

dados_frame = pd.DataFrame(dados)

print(dados_frame.shape)
print(dados_frame.head())
