import os, json
import matplotlib.pyplot as plt
from metrics_class import ModelMetrics


def generate_comparison_grafics(cross, no_cross, labels, figsize=(20, 20)):

    plt.figure(figsize=figsize) 

    epochs = [x+1 for x in range(len(no_cross))]

    plt.plot(epochs, cross, label=f'cross_val_{labels}', color='grey', linestyle='-.')

    plt.plot(epochs, no_cross, label=f'no_cross_val_{labels}', color='black', linestyle='-')

    plt.xlabel('Épocas' , fontsize=20)

    plt.ylabel(f"{labels}" , fontsize=20)

    plt.legend()

    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))

    checkpoint_dir = os.path.join(file_dir, 'screening_fit_history')

    history_path = os.path.join(checkpoint_dir, 'comparacao.json')

    with open(history_path, 'r') as file:
        history = json.loads(file.read())


    for key, value in history.items():
        cross=value['com_fold']
        
        no_cross=value['sem_fold']

        no_cross=no_cross[0:49]

        generate_comparison_grafics(
            labels=key,
            cross=cross,
            no_cross=no_cross
        )

        x=2
    
    x=2
    
