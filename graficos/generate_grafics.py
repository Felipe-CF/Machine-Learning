import os, json
import matplotlib.pyplot as plt
from metrics_class import ModelMetrics


def generate_grafics(model, figsize=(6, 12), colors = ['red','blue']):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize)

    axes = axes.ravel()

    plt.subplots_adjust(hspace=0.9, wspace=0.3, left=0.3) 

    params = ["AUC","Accuracy", "F1_score", "Precision", "Recall", "loss"]

    fig.suptitle('Histórico de Testes', fontsize=18)

    metrics = model.get_metrics()

    for i, param in enumerate(metrics):
        train, val = param

        title = params[i]

        epochs = [x+1 for x in range(len(train))]

        axes[i].plot(epochs, train, label=f'train_{title}', color='red', linestyle='-')

        axes[i].plot(epochs, val, label=f'val_{title}', color='blue', linestyle='-')
        
        axes[i].set_title(f'Comparativo {title}')
        
        axes[i].set_xlabel('Épocas')

        axes[i].set_ylabel(f'{title}')

        axes[i].legend()

        axes[i].grid(True)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    plt.show()


def set_train():
    pass

def set_val():
    pass

if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))

    checkpoint_dir = os.path.join(file_dir, 'screening_fit_history')

    history_path = os.path.join(checkpoint_dir, 'cross_validation_history.json')

    with open(history_path, 'r') as file:
        history = json.loads(file.read())

    model_net = ModelMetrics(history)

    figsize=(6, 12)

    colors = ['red', 'blue']

    generate_grafics(
        model=model_net, 
        figsize=figsize, 
        colors=colors
        )

    x=2
    
