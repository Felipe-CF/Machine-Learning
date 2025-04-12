# Salvando arquivos e configurando Deploy


## Salvando arquivos - Código

    import numpy as np
    import os
    from dotenv import load_dotenv
    from bases.base_treino_teste import *
    from bases.dados.dados_tratados_esc import *


    load_dotenv()

    output_dir = os.getenv('OUTPUT_DIR')

    np.savetxt(os.path.join(output_dir, 'previsores.csv'), previsores, delimiter=',')

    np.savetxt(os.path.join(output_dir, 'alvo.csv'), alvo, delimiter=',')



## configurando Deploy - Código

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

## Simulando novos pacientes

### Legenda

* Age = int
* Sex = (M=0, F=1)
* Chest Pain Type (tipo de dor no peito) = [TA=0(tangina típica), ATA=1(angina atípica), NAP=2 (dor não anginosa), ASY=3 (assintomático)]
* RestinBP (pressão sanguínea em repouso em mmHg) = float
* Cholesterol (sérico, mg/dl) = float
* Fasting BS (açucar no sangue em jejum mg/dl) = [0 : < 120, 1 >= 120]
* Resting ECG (eletrocardiograma em repouso) = [normal=0, ST(anormalidade da onda)=1, ST-T (hipertrofia ventricular esquerda)]
* Max HR (frequencia cardiaca maxima) = float
* Exercise Angina = [Não=0, Sim=1]
* Old Peak (depressão de ST induzida por exercicio em relação ao repouso)
* ST_Slope (inclinação do segmenet ST) = [Up=0, Flat=1, Down=2]
* Heart Disease = [Não=0, Sim=1]




