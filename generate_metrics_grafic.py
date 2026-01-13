import os, json
import matplotlib.pyplot as plt
from metrics_class import ModelMetrics


def generate_metrics_grafic(history, labels, fold, figsize=(40, 40)):

    plt.figure(figsize=figsize) 

    epochs = [x+1 for x in range(len(history))]

    linestyles = ['solid', '-.']

    for label, linestyle, in zip(labels, linestyles):
        metric = history[label][fold]

        epochs = [x+1 for x in range(len(metric))]

        color = 'grey'

        if linestyle == 'solid':
            color = 'black'

        plt.plot(
            epochs, 
            history[label][fold], 
            label=f'{label}', 
            color=color, 
            linestyle=linestyle,
            linewidth=3
            )

    plt.xlabel('Épocas' , fontsize=20)

    plt.ylabel("Métricas" , fontsize=20)

    plt.legend(fontsize=18, loc='lower right')

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
            labels=["AUC","Acuracia"],
            history=history,
            fold=fold
        )

        x=3

    
    x=2
    
