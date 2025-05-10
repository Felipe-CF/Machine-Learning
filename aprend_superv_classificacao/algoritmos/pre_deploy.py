import os
import numpy as np
from dotenv import load_dotenv
from bases.base_treino_teste import *
from bases.dados.dados_tratados_esc import *


load_dotenv()

output_dir = os.getenv('OUTPUT_DIR')

np.savetxt(os.path.join(output_dir, 'previsores.csv'), previsores, delimiter=',')

np.savetxt(os.path.join(output_dir, 'alvo.csv'), alvo, delimiter=',')



