import os, json
import matplotlib.pyplot as plt
from metrics_class import ModelMetrics


def generate_metrics_grafic(history, labels, fold, figsize=(40, 40)):

    plt.figure(figsize=figsize) 

    epochs = [x+1 for x in range(len(history))]

    linestyles = ['solid', 'dashed', 'dashed']

    alphas = [0.4, 0.6, 1]

    for label, linestyle, alpha in zip(labels, linestyles, alphas):
        metric = history[label][fold]

        epochs = [x+1 for x in range(len(metric))]

        color = 'grey'

        if linestyle == 'dotted':
            color = 'black'

            linestyle = 'dashed'

        plt.plot(
            epochs, 
            history[label][fold], 
            label=f'val_{label}', 
            color=color, 
            linestyle=linestyle,
            alpha=alpha,
            linewidth=2
            )

    plt.xlabel('Épocas' , fontsize=20)

    plt.ylabel("Métricas" , fontsize=20)

    plt.legend()

    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))

    checkpoint_dir = os.path.join(file_dir, 'screening_fit_history')

    history_path = os.path.join(checkpoint_dir, 'comparacao.json')

    with open(history_path, 'r') as file:
        history = json.loads(file.read())

    folds = ['sem_fold', 'com_fold']

    for fold in folds: 
        generate_metrics_grafic(
            labels=["AUC","Accuracy"],
            history=history,
            fold=fold
        )

        x=3

    
    x=2
    
