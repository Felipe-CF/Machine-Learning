import os 
import numpy as np
import pandas as pd
from bases.dados_tratados import *
from sklearn.svm import SVR
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score  
from sklearn.metrics import mean_absolute_error, mean_squared_error


load_dotenv()

dir = os.getenv('OUTPUT_DIR')

var_independente = pd.read_csv(os.path.join(dir, 'independente.csv'), header=None).values

var_dependente = pd.read_csv(os.path.join(dir, 'dependente.csv'), header=None).values

random = RandomForestRegressor(max_depth=5, random_state=10, n_estimators=150, criterion='squared_error')

random.fit(var_independente, var_dependente)

rm = input('Número de cômodos: ')

lstat = input('Porcentagem de proprietários de baixa renda: ')

ptratio = input('Razão de estudantes e professores: ')

infos = [rm, lstat, ptratio]

valor_casa = random.predict([infos]) # array 2D

print(f'O valor da casa é {valor_casa[0]:.2f}$')





