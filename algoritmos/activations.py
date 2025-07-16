import numpy as np

def sigmoid(z):
    # cada elemento do vetor Numpy será usado como expoente negativo de e (aprox 2.718)
    return 1.0/(1.0+np.exp(-z))