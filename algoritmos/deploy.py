import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.express as px
import os
from bases.dados_tratados import *


load_dotenv()

dir = os.getenv('OUTPUT_DIR')

var_independente = pd.read_csv(os.path.join(dir, 'independente.csv'), header=None).values

var_dependente = pd.read_csv(os.path.join(dir, 'dependente.csv'), header=None).values

print(var_dependente)

