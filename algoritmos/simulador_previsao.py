import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.express as px
from bases.dados_tratados import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


independente = dados_tratados.iloc[:, 0:3]

dependente = dados_tratados.iloc[:, 3]

load_dotenv()

dir = os.getenv('OUTPUT_DIR')

np.savetxt(os.path.join(dir, 'independente.csv'), independente,  delimiter=',')

np.savetxt(os.path.join(dir, 'dependente.csv'), dependente,  delimiter=',')
