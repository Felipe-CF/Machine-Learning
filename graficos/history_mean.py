import os, json, statistics

file_dir = os.path.dirname(os.path.abspath(__file__))

checkpoint_dir = os.path.join(file_dir, 'screening_fit_history')

fold_history = [
    'kfold_1_fit_history_val_auc_0.9226.json', 
    'kfold_2_fit_history_val_auc_0.9356.json',  
    'kfold_3_fit_history_val_auc_0.9339.json',  
    'kfold_4_fit_history_val_auc_0.9372.json',  
    'kfold_5_fit_history_val_auc_0.9322.json'
    ]

json_values = {
    "val_AUC": {
        "mean": [],
        "std_desv": [],
    },
    "val_Accuracy": {
        "mean": [],
        "std_desv": [],
    },
    "val_Precision": {
        "mean": [],
        "std_desv": [],
    },
    "val_Recall": {
        "mean": [],
        "std_desv": [],
    },
    "val_F1_score": {
        "mean": [],
        "std_desv": [],
    },
}

epochs = 236

for key, values in json_values.items():

    for _path in fold_history:
        history_path = os.path.join(checkpoint_dir, _path)
        
        with open(history_path, 'r') as file:
            history = json.loads(file.read())
        
        values['mean'].extend(history[key])
        
        values['std_desv'].extend(history[key])

    values['mean'] = round(sum(values['mean']) / len(values['mean']), 4)
    
    values['std_desv'] = round(statistics.stdev(history[key]), 4)

    x=2

save_path = checkpoint_dir + '\\mean_metrics.json'

with open(save_path, 'w') as file:
    file.write(json.dumps(json_values))

print("Média e Desvio Padrão salvos com sucesso!")

