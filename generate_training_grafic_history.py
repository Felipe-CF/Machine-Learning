import os, json
import matplotlib.pyplot as plt
from metrics_class import ModelMetrics


def generate_training_grafic_history(metrics,  labels, figsize=(20, 20)):

    plt.figure(figsize=figsize) 

    epochs = [x+1 for x in range(len(metrics[0]))]

    # for metric, label in zip(metrics, labels):

    
    plt.plot(epochs, metrics[0], label=f'{labels[0]}', color='black', linewidth=3,linestyle='solid')

    plt.plot(epochs, metrics[1], label=f'{labels[1]}', color='grey', linewidth=3, linestyle='solid')

    # plt.plot(epochs, metrics[2], label=f'cross_val_{labels[2]}', color='grey', linestyle='solid')
    
    plt.xlabel('Épocas' , fontsize=25)

    plt.ylabel("Métricas" , fontsize=25)

    plt.legend(fontsize=18)

    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))

    checkpoint_dir = os.path.join(file_dir, 'screening_fit_history')

    history_path = os.path.join(checkpoint_dir, 'cross_validation_history.json')

    with open(history_path, 'r') as file:
        history = json.loads(file.read())

    metrics = [history['val_AUC'], history['val_Accuracy']]

    labels = ["AUC","Acuracia"]

    generate_training_grafic_history(
        labels=labels,
        metrics=metrics,
    )

    x=2
    
