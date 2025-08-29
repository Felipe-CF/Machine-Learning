import keras
from util import *
from create_net_no_res import *
import numpy as np, random
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint



if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))

    conv_net = create_load_net()

    training_set, validation_set = create_sets()

    checkpoint_dir = os.path.join(file_dir, 'model_checkpoints')

    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'crohnv_net_val_accuracy_{val_accuracy:.4f}.keras'),
        mode='max', # adequado para val_accuracy
        save_best_only=True, # salvar quando a métrica melhora
        save_weights_only=False, # somente os pesos
        monitor='val_accuracy', # métrica balizadora do armazenamento (precisão da validação)
        verbose=1 # logs de salvamento
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.1,
        patience=10,
        mode='auto',
        start_from_epoch=30,
        restore_best_weights=True,
    )

    conv_net.fit(
        training_set, 
        steps_per_epoch=87, 
        epochs=100,
        validation_data=validation_set,
        validation_steps=20, 
        callbacks=[model_checkpoint, early_stop]
    )





