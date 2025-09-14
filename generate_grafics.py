import os, json
import matplotlib.pyplot as plt
from metrics_class import ModelMetrics


def generate_grafics(model, figsize=(6, 12), colors = ['red', 'green', 'blue', 'orange']):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize)

    axes = axes.ravel()

    plt.subplots_adjust(hspace=0.9, wspace=0.3, left=0.3) 

    fig.suptitle('Histórico de Testes', fontsize=18)

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


if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))

    checkpoint_dir = os.path.join(file_dir, 'screening_fit_history')

    history_path = os.path.join(checkpoint_dir, 'fit_history_auc_0.9166_val_auc_0.8587.json')

    with open(history_path, 'r') as file:
        history = json.loads(file.read())

    model_net = ModelMetrics(history)

    figsize=(6, 12)

    colors = ['red', 'green', 'blue', 'orange']

    generate_grafics(
        model=model_net, 
        figsize=figsize, 
        colors=colors
        )


    pass
