import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
from xgboost import XGBClassifier


load_dotenv()

output_dir = os.getenv('OUTPUT_DIR')

previsores = pd.read_csv(os.path.join(output_dir, 'previsores.csv'), header=None)

alvo = pd.read_csv(os.path.join(output_dir, 'alvo.csv'), header=None)

xgboost = XGBClassifier(max_depth=2, learning_rate=0.05, n_estimators=210, objective='binary:logistic', random_state=3)

xgboost.fit(previsores, alvo)

# Simulando novos pacientes

pacientes = [
    [45, 0, 1, 130.0, 233.0, 1, 0, 150.0, 0, 2.3, 0],
    [60, 1, 3, 140.0, 294.0, 0, 1, 160.0, 1, 1.2, 2],
    [52, 0, 0, 120.0, 260.0, 1, 0, 155.0, 0, 0.6, 1],
    [38, 1, 2, 110.0, 211.0, 0, 1, 170.0, 0, 0.0, 0],
    [66, 0, 3, 150.0, 300.0, 1, 2, 130.0, 1, 2.6, 2],
    [55, 1, 1, 135.0, 250.0, 0, 0, 140.0, 0, 1.0, 1],
    [47, 0, 0, 125.0, 210.0, 1, 0, 165.0, 1, 1.5, 1],
    [61, 1, 2, 145.0, 310.0, 1, 1, 120.0, 0, 3.2, 2],
    [50, 0, 1, 140.0, 275.0, 0, 2, 150.0, 0, 0.5, 0],
    [34, 1, 0, 115.0, 180.0, 0, 0, 180.0, 0, 0.0, 0]
]

pacientes_array = np.array(pacientes).reshape(len(pacientes), -1)

resultados = xgboost.predict(pacientes_array)

for i, resultado in enumerate(resultados):
    status = "possui" if resultado == 1 else "não possui"
    print(f"Paciente {i+1} {status} tendências a problemas cardíacos")




