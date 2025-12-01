import os, json
import matplotlib.pyplot as plt
from metrics_class import ModelMetrics


def generate_grafic(model, metric, figsize=(8, 12), colors = ['red','blue']):

    plt.figure(figsize=figsize) 

    metrics = model.get_metrics(metric)

    epochs = [x+1 for x in range(len(metrics[0]))]

    plt.plot(epochs, metrics[0], label=f'train_{metric}', color='grey', linestyle='-.')

    plt.plot(epochs, metrics[1], label=f'val_{metric}', color='black', linestyle='-')

    plt.xlabel('Épocas' , fontsize=20)

    plt.ylabel(f"{metric}" , fontsize=20)

    plt.grid(True)

    plt.show()


def generate_grafics(model, figsize=(8, 12), colors = ['red','blue']):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    axes = axes.ravel()

    plt.subplots_adjust(hspace=0.3, wspace=0.3, left=0.1, right=0.9) 

    # params = ["AUC","Accuracy", "F1_score", "Precision", "Recall", "loss"]
    params = ["Accuracy", "F1_score", "Precision", "Recall"]

    # metrics = model.get_metrics()

    metrics = []

    metrics.append(model.get_metrics('Accuracy'))
    metrics.append(model.get_metrics('F1_score'))
    metrics.append(model.get_metrics('Precision'))
    metrics.append(model.get_metrics('Recall'))

    for i, param in enumerate(metrics):
        train, val = param

        title = params[i]

        epochs = [x+1 for x in range(len(train))]

        axes[i].plot(epochs, train, label=f'train_{title}', color='grey', linestyle='-.')

        axes[i].plot(epochs, val, label=f'val_{title}', color='black', linestyle='-')

        axes[i].set_xlabel('Épocas', fontsize=16)

        axes[i].set_ylabel(f'{title}', fontsize=18)

        axes[i].legend()

        axes[i].grid(True)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])

    plt.show()


def set_train():
    pass

def set_val():
    pass

if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))

    checkpoint_dir = os.path.join(file_dir, 'screening_fit_history')

    history_path = os.path.join(checkpoint_dir, 'fit_history_auc_0.8672_val_auc_0.8722.json')

    with open(history_path, 'r') as file:
        history = json.loads(file.read())

    model_net = ModelMetrics(history)

    figsize=(6, 12)

    colors = ['red', 'blue']

    generate_grafic(
        model=model_net,
        metric='AUC'
    )

    generate_grafics(
        model=model_net, 
        figsize=figsize, 
        colors=colors
        )

    x=2
    
