import pandas as pd
import numpy as np


class Perceptron:
    def __init__(self, learning_rate, epochs):
        self.weights = np.random.randn(1, 3)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def validation(self, x):
        for x_k in x:

            y = np.dot(x_k, self.weights.reshape(-1, 1))

            prediction =  -1 if y[0] < 0 else 1

            print(prediction)
    
    def update_weights(self, x_k, d_k, y):
        y_diff = (d_k - y) * x_k

        diff_learned = self.learning_rate * y_diff

        self.weights = self.weights + diff_learned


    def training(self, training_set, outputs):
        for epoch in range(self.epochs):
            error = 0

            for x_k, d_k in zip(training_set, outputs):
                u = np.dot(x_k, self.weights.reshape(-1, 1))

                y = -1 if u < 0 else 1

                if y != d_k:
                    error += 1
                    self.update_weights(x_k, d_k, y)
            
            print(f'Training Precision in epoch {epoch+1}: {(error/30)*100:.2f}%')
            
 
if __name__ == '__main__':
    training_dataset = pd.read_csv("datasets\\training_set.csv", sep=',', encoding='iso-8859-1')
    
    validation_dataset = pd.read_csv("datasets\\validation_set.csv", sep=',', encoding='iso-8859-1')

    training_dataset = pd.DataFrame.copy(training_dataset)

    x = training_dataset.iloc[1:, 1:4]

    y = training_dataset.iloc[:, 4:]

    v = validation_dataset.iloc[:, 1:4]

    learning_rate = input('learning_rate ')

    epochs = input('epochs ')

    perceptron = Perceptron(learning_rate, epochs)

    perceptron.training(x.values, y.values)

    perceptron.validation(v.values)


    