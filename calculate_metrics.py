import json, os
import matplotlib.pyplot as plt
import tabulate as tbt


def generate_mean():
    pass
    

def generate_std_desviation():
    pass


def generate_IC():
    pass


if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))

    checkpoint_dir = os.path.join(file_dir, 'screening_fit_history')

    history_path = os.path.join(checkpoint_dir, 'kfolds_history.json')

    with open(history_path, 'r') as file:
        history = json.loads(file.read())

    for key, values in history.items():

        val = values['val']

        auc = val['val_AUC']

        i = auc.index(max(auc))

        print(key, end=' ')

        # json_metrics = {
        #     "AUC": [],
        #     "Accuracy": [],
        #     "F1_score": [],
        #     "Precision": [],
        #     "Recall": [],
        #     "loss": []
        # }

        # headers = ["AUC","Accuracy","F1_score","Precision","Recall","Loss", "Mean", "Standard Desviation", "IC(95%)"]

        for k, v in val.items():
            m = v[i]
            print(f'{k}: {m*100:.2f} ', end=' ')

        print()

    x=2
    
