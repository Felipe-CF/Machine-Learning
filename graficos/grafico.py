import matplotlib.pyplot as plt
import re, os, json
from graficos.metric import Metrics



def generate_grafics(test_regularization, params, epochs, label_line, figsize=(6, 12), colors = ['red', 'green', 'blue', 'orange']):
    fit_metrics = Metrics(test_regularization)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize)

    axes = axes.ravel()

    plt.subplots_adjust(hspace=0.9, wspace=0.3) 

    fig.suptitle('Histórico de Testes de Regularização', fontsize=18)

    for i, param in enumerate(params):
        metrics = fit_metrics.get_metrics(param)

        for metric_color_label in zip(metrics, colors, label_line):
            metric = metric_color_label[0]

            color=metric_color_label[1]

            label=metric_color_label[2]

            axes[i].plot(epochs, metric, label=label, color=color, linestyle='-')
        
        axes[i].set_title(f'Comparativo {param}')
        
        axes[i].set_xlabel('Épocas')

        axes[i].set_ylabel(f'{param}')

        axes[i].legend()

        axes[i].grid(True)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    plt.show()


if __name__ == "__main__":

    file_dir = os.path.dirname(os.path.abspath(__file__))

    history_path = os.path.join(file_dir, 'test_regularization.json')

    test_regularization = {}

    with open(history_path, 'r') as file:
        test_regularization = json.loads(file.read())

    params = ["val_AUC","val_Accuracy", "val_F1_score", "val_Precision", "val_Recall", "val_loss"]

    label_line = ["Baseline","L1", "L2", "Dropout"] 

    epochs = test_regularization["L1"]["epoch"]

    generate_grafics(test_regularization, params, epochs=epochs, figsize=(6, 18), label_line=label_line)
