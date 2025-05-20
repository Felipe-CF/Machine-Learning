import numpy as np, random
import tensorflow as tf
from activations import *


'''
        BIAS:
            pelas bias não serem gerados no sizes[0], fica entendido que esta será a camada de entrada
            são matrizes 2D numpy 

        PESOS:
            eles são geradas entre 2 camadas ( entrada -> oculta, oculta -> saída)
            np.random.randn(y, x) → isso cria uma matriz de dimensão (y linhas, x colunas), ou seja:

            (entrada -> oculta)                          
            1° iteração = [3, 2] -->    [[11 12] (6 pesos)
                                        [12  22]
                                        [13  23]]

            (oculta -> saída)
            2° iteração (matriz W) = [1, 3] --> [[11 12 13]] (3 pesos)

            temos que o peso Wjk é "o peso para conexão do neurônio k-esimo"
            com "o neurônio j-esimo da camada seguinte"
'''

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        
        self.sizes = sizes # n° neuronios nas respectivas camadas

        # iniciados aleatoriamente gerando distribuições gaussianas (normais)
        # com média de 0 e desvio padrão 1
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)

        return a

    
    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)

        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        
        for j in range(epochs): # em cada epoca...
            random.shuffle(training_data) # randomiza os dados de treino...

            # e reparte eles em mini-lotes apropriados
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            # para cada mini_batch aplicamos um único passo de descida do gradiente
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                print("Epoch {}:{}/{}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} finalizada".format(j))

    # atualiza os pesos e os bias da rede
    # de acordo com uma iteração de descida de gradiente
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]    

        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x

        # lista para armazenar todas as ativações, camada por camada
        activations = [x]

        # lista para armazenar todas os vetores z, camada por camada
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            
            zs.append(z)

            activation = sigmoid(z)

            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        
        nabla_b = delta
        
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z= zs[-l]

            sp = sigmoid_prime(z)

            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            nabla_b[-l] = delta

            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)



        
rede = Network([2, 3, 1])





