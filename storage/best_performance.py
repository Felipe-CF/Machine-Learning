import os, json


if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))

    checkpoint_dir = os.path.join(file_dir, 'screening_fit_history')

    history_path = os.path.join(checkpoint_dir, 'fit_history_auc_0.9166_val_auc_0.8587.json')

    with open(history_path, 'r') as file:
        history = json.loads(file.read())

    val_auc = history['val_AUC']

    max_aauc = val_auc.index(max(val_auc))

    for key, values in history.items():
        print(f'{key} {values[max_aauc]}')

