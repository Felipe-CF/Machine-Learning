import keras
from hyperparameters import *
import numpy as np, random
from create_pathology_net import *
import matplotlib.pyplot as plt
from keras.optimizers import SGD


if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))

    res_net = create_load_net()
    # res_net = create_load_net(file_dir)

    res_net.compile(
        optimizer=SGD(learning_rate=0.025, momentum=0.99), 
        loss=keras.losses.BinaryCrossentropy(), 
        metrics=[keras.metrics.BinaryCrossentropy(), keras.metrics.AUC(name='auc')]
    )

    training_set, validation_set = create_sets()

    checkpoint_dir = os.path.join(file_dir, 'pathology_checkpoints')

    print(res_net.summary())

    res_net.fit(
        training_set, 
        steps_per_epoch=174, 
        epochs=100,
        validation_data=validation_set,
        validation_steps=43, 
        callbacks=[model_checkpoint(checkpoint_dir), early_stopping(), learning_rate_plateau()]
    )

    save_history(history=res_net.history, file_dir=file_dir)


