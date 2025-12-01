import json, os


file_dir = os.path.dirname(os.path.abspath(__file__))

checkpoint_dir = os.path.join(file_dir, 'screening_fit_history')

# history_path = os.path.join(checkpoint_dir, 'kfold_1_fit_history_val_auc_0.9226.json')
# history_path = os.path.join(checkpoint_dir, 'kfold_2_fit_history_val_auc_0.9356.json')
# history_path = os.path.join(checkpoint_dir, 'kfold_3_fit_history_val_auc_0.9339.json')
# history_path = os.path.join(checkpoint_dir, 'kfold_4_fit_history_val_auc_0.9372.json')
history_path = os.path.join(checkpoint_dir, 'kfold_5_fit_history_val_auc_0.9322.json')

with open(history_path, 'r') as file:
    history = json.loads(file.read())

x = 2