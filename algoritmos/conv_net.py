import numpy as np, random
import tensorflow as tf
from activations import *
# from IPython.display import Image




'''
"Se não usarmos padding, as dimensões da imagem diminuem a cada camada convolucional. Isso pode causar uma redução rápida demais no tamanho dos mapas. Para controlar isso e preservar mais detalhes, usamos padding."
'''