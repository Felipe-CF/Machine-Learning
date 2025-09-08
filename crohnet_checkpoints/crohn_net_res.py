import keras
from util import *
from hyperparameters import *
import numpy as np, random
from create_crohn_net import *
from keras.optimizers import SGD, Adam


if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))

    res_net = create_load_net()
    # res_net = create_load_net(file_dir)

    res_net.compile(
        optimizer=SGD(learning_rate=0.025, momentum=0.99, name='SGD'), 
        loss=keras.losses.BinaryCrossentropy(), 
        metrics=metrics()
    )

    training_set, validation_set = create_sets()

    checkpoint_dir = os.path.join(file_dir, 'crohnet_checkpoints')

    # print(res_net.summary())

    res_net.fit(
        training_set, 
        steps_per_epoch=174, 
        epochs=100,
        validation_data=validation_set,
        validation_steps=43,
        class_weight=class_weights(), 
        callbacks=[model_checkpoint(checkpoint_dir), early_stopping(), learning_rate_plateau()]
    )

    save_history(history=res_net.history, file_dir=file_dir)


