import os, json
import matplotlib.pyplot as plt


file_dir = os.path.dirname(os.path.abspath(__file__))

checkpoint_dir = os.path.join(file_dir, 'screening_fit_history')

history_path = os.path.join(checkpoint_dir, 'comparacao.json')

with open(history_path, 'r') as file:
    history = json.loads(file.read())

folds = ['sem_fold', 'com_fold']


auc_sem = history['AUC']['sem_fold'][0:49]

auc_com = history['AUC']['com_fold']

accuracy_sem = history['Accuracy']['sem_fold'][0:49]

accuracy_com = history['Accuracy']['com_fold']

epochs = [x+1 for x in range(len(auc_sem))]

plt.figure(figsize=(20, 20)) 

plt.plot(
    epochs, 
    auc_sem, 
    label=f'AUC (Sem validação cruzada)', 
    color='grey', 
    linestyle='solid',
    )

plt.plot(
    epochs, 
    accuracy_sem, 
    label=f'Accuracy (Sem validação cruzada)', 
    color='grey',
    linestyle='-.',
    )

plt.plot(
    epochs, 
    auc_com, 
    label=f'AUC (Com validação cruzada)', 
    color='black', 
    alpha=0.8, 
    linestyle='solid',
    )

plt.plot(
    epochs, 
    accuracy_com, 
    label=f'Accuracy (Com validação cruzada)', 
    color='black', 
    alpha=0.8, 
    linestyle='-.',
    )

plt.xlabel('Épocas' , fontsize=20)

plt.ylabel("Métricas" , fontsize=20)

plt.legend()

plt.grid(True)

plt.show()


x=2
    
