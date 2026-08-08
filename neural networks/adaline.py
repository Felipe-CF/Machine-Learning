import os
import pandas as pd
import numpy as np

def open_dataset():
    adaline_dir = os.path.dirname(os.path.abspath(__file__))

    training_path = os.path.join(adaline_dir, "datasets", "training_adaline.csv")

    validation_path = os.path.join(adaline_dir, "datasets", "validation_adaline.csv")

    training_dataset = pd.read_csv(training_path, sep=',', encoding='iso-8859-1')

    validation_dataset = pd.read_csv(validation_path, sep=',', encoding='iso-8859-1')

    training_dataset = pd.DataFrame.copy(training_dataset)

    x = training_dataset.iloc[:, 1:5]

    x.insert(0, 'x0', -1)

    y = training_dataset.iloc[:, 5:]

    v = validation_dataset.iloc[:, 1:]

    v.insert(0, 'x0', -1)

    dataset = [(x, y) for x, y in zip(x.values, y.values)]

    return dataset, v


class Adaline():
    def __init__(self):
        self.weights = np.random.randn(1, 5)

    def validation(self, dataset):
        i = 1
        for x in dataset:
            u = np.dot(self.weights, x.reshape(-1, 1))

            y = -1 if u < 0 else 1

            if y == -1:
                print(f'Validation {i}: A') 
            else:
                print(f'Validation {i}: B') 
            
            i+=1

    def training(self, dataset, learning_rate, precision):
        epochs = 0

        actual_square_error = 0

        while True:
            past_square_error =  self.least_mean_square(dataset)

            for x, y in dataset:
                u = np.dot(self.weights, x.reshape(-1, 1))

                self.weights = self.weights + learning_rate * (y - u) * x
            
            epochs += 1

            actual_square_error = self.least_mean_square(dataset)

            if abs(actual_square_error - past_square_error) <= precision:
                break
        
        print(f'Epochs: {epochs}')

    def least_mean_square(self, dataset):
        square_error = 0

        for x, y in dataset:
            u = np.dot(self.weights, x.reshape(-1, 1))

            square_error += pow((y - u), 2)
        
        square_error /= len(dataset)

        return square_error 

    def predict(self, dataset):
        error_count = 1

        for x, d_k in dataset:
            u = np.dot(self.weights, x.reshape(-1, 1))

            y = -1 if u < 0 else 1

            if d_k != y:
                error_count += 1
        
        print(f'Precision after training: {(error_count/len(dataset))*100:.2f}') 

if __name__ == '__main__':
    dataset, v = open_dataset()

    adaline = Adaline()

    learning_rate = float(input('learning_rate '))

    precision = float(input('precision '))

    adaline.training(dataset, learning_rate, precision)

    adaline.validation(v.values)
