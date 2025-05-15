import numpy as np
import tensorflow as tf
from activations import *


'''
        BIAS:
            pelas bias não serem gerados no sizes[0]
            fica entendido que esta será a camada de entrada
        PESOS:
            eles são geradas entre 2 camadas ( entrada -> oculta, oculta -> saída)
            np.random.randn(y, x) → isso cria uma matriz de dimensão (y linhas, x colunas), ou seja:

            1° iteração = [3, 2] --> [[11 12] (6 pesos)
                                      [12  22]
                                      [13  23]]

            2° iteração (matriz W) = [1, 3] --> [[11 12 13]] (3 pesos)

            temos que o peso Wjk é "o peso para conexão do neurônio k-esimo da segunda camada"
            com "o neurônio j-esimo da terceira camada"
'''

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        
        self.sizes = sizes # n° neuronios nas respectivas camadas

        # iniciados aleatoriamente gerando distribuições gaussianas
        # com média de 0 e desvio padrão 1
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        
        return a

        
rede = Network([2, 3, 1])




