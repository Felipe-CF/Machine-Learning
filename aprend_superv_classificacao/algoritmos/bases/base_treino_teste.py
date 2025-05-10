import numpy as np
import pandas as pd
from .dados.dados_tratados_esc import *
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


x_train, x_test, y_train, y_test = train_test_split(previsores, alvo, test_size=0.3, random_state=0)
