import os
import pandas as pd
import numpy as np


class Perceptron:
    def __init__(self):
        self.weights = np.random.randn(1, 4)

    def validation(self, x):
        for x_k in x:
            x_k = np.insert(x_k, 0, -1)
            y = np.dot(x_k, self.weights.reshape(-1, 1))

            prediction =  -1 if y < 0 else 1

            print(prediction)
    
    def update_weights(self, x_k, d_k, y, learning_rate):
        diff_learned = learning_rate * (d_k - y) * x_k

        # regra de Hubb - incremento de pesos
        self.weights = self.weights + diff_learned


    def training(self, dataset, learning_rate):
        epoch = 0

        erro = False

        while erro is False:
            error = 0

            for x_k, d_k in dataset:
                x_k = np.insert(x_k, 0, -1)

                u = np.dot(x_k, self.weights.reshape(-1, 1))

                # symmetric hard limiter - função sinal
                y = -1 if u < 0 else 1

                if y != d_k:
                    erro = True

                    error += 1
                    
                    self.update_weights(x_k, d_k, y, learning_rate)
            
            epoch += 1

            print(f'Training Precision in epoch {epoch}: {(error/30)*100:.2f}%')

            if erro is True:
                erro = False

            else:
                break

            
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

    dataset = [(x, y) for x, y in zip(x.values, y.values)]

    learning_rate = float(input('\nlearning_rate '))

    perceptron = Perceptron()

    perceptron.training(dataset, learning_rate)

    perceptron.validation(v.values)


    