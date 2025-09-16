import os, json


def save_history(file_dir, history):
    file_dir = os.path.dirname(os.path.abspath(__file__))

    history_path = os.path.join(file_dir, 'screening_fit_history')

    val_auc = history.history['val_AUC']

    auc = val_auc.index(max(val_auc))

    auc = history.history['AUC'][auc]

    val_auc = max(val_auc)

    history_path = history_path + f'\\fit_history_auc_{auc:.4f}_val_auc_{val_auc:.4f}.json'

    with open(history_path, 'w') as file:   
        file.write(json.dumps(history.history))

    print('Last history of training saved sucessfull!')