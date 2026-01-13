import json, os, time
from datetime import datetime

file_dir = os.path.dirname(os.path.abspath(__file__))

checkpoint_dir = os.path.join(file_dir, 'screening_fit_history')

# history_path = os.path.join(checkpoint_dir, 'kfold_1_fit_history_val_auc_0.9226.json')
# history_path = os.path.join(checkpoint_dir, 'kfold_2_fit_history_val_auc_0.9356.json')
# history_path = os.path.join(checkpoint_dir, 'kfold_3_fit_history_val_auc_0.9339.json')
# history_path = os.path.join(checkpoint_dir, 'kfold_4_fit_history_val_auc_0.9372.json')
history_path = os.path.join(checkpoint_dir, 'kfold_5_fit_history_val_auc_0.9322.json')

with open(history_path, 'r') as file:
    history = json.loads(file.read())



start = time.time()

time.sleep(5)

end = time.time()

a = end - start

total_prediction_time = end.seconds + (end.microseconds / 1e+6)

image_prediction_time = total_prediction_time / 1000

total_minutes, total_seconds = divmod(total_prediction_time, 60)

image_minutes, image_seconds = divmod(image_prediction_time, 60)

print(f'Tempo de inferência por imagem da CrohNet: {int(image_minutes)} minutos e {image_seconds:.2f} segundos')

print(f'Tempo de inferências da CrohNet: {int(total_minutes)} minutos e {total_seconds:.2f} segundos')