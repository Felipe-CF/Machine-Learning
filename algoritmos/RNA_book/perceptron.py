import os
import pandas as pd
import numpy as np


class Perceptron:
    def __init__(self, learning_rate, epochs):
        self.weights = np.random.randn(1, 3)
        self.weights = np.insert(self.weights, 0, 0)


    def validation(self, x):
        for x_k in x:

            y = np.dot(x_k, self.weights.reshape(-1, 1))

            prediction =  -1 if y < 0 else 1

            print(prediction)
    
    def update_weights(self, x_k, d_k, y, learning_rate):
        y_diff = (d_k - y) * x_k

        diff_learned = learning_rate * y_diff

        # regra de Hubb - incremento de pesos
        self.weights = self.weights + diff_learned


    def training(self, training_set, outputs, learning_rate, epochs):

        for epoch in range(epochs):
            error = 0

            for x_k, d_k in zip(training_set, outputs):
                x_k = np.insert(x_k, 0, -1)

                u = np.dot(x_k, self.weights.reshape(-1, 1))

                # symmetric hard limiter - função sinal
                y = -1 if u < 0 else 1

                if y != d_k:
                    error += 1
                    self.update_weights(x_k, d_k, y, learning_rate)
            
            print(f'Training Precision in epoch {epoch+1}: {(error/30)*100:.2f}%')
            
 
if __name__ == '__main__':
    perceptron_dir = os.path.dirname(os.path.abspath(__file__))

    training_path = os.path.join(perceptron_dir, "datasets", "training_perceptron.csv")

    validation_path = os.path.join(perceptron_dir, "datasets", "validation_perceptron.csv")

    training_dataset = pd.read_csv(training_path, sep=',', encoding='iso-8859-1')

    validation_dataset = pd.read_csv(validation_path, sep=',', encoding='iso-8859-1')

    training_dataset = pd.DataFrame.copy(training_dataset)

    x = training_dataset.iloc[1:, 1:4]

    y = training_dataset.iloc[:, 4:]

    v = validation_dataset.iloc[:, 1:4]

    learning_rate = float(input('\nlearning_rate '))

    epochs = int(input('epochs '))

    perceptron = Perceptron()

    perceptron.training(x.values, y.values, learning_rate, epochs)

    perceptron.validation(v.values)


    