import keras
from hyperparameters import *
import numpy as np, random
from create_screening_net import *
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.losses import BinaryCrossentropy


if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))

    screening_net = create_load_net()
    # screening_net = create_load_net(file_dir)

    screening_net.compile(
        optimizer=SGD(learning_rate=0.0025, momentum=0.99, name='SGD'), 
        loss=BinaryCrossentropy(), 
        metrics=screening_metrics()
    )

    training_set, validation_set = create_sets()

    checkpoint_dir = os.path.join(file_dir, 'screening_checkpoints')

    screening_net.fit(
        training_set, 
        steps_per_epoch=174, 
        epochs=100,
        validation_data=validation_set,
        validation_steps=43,
        verbose=1,
        class_weight=screening_class_weights(), 
        callbacks=[model_checkpoint(checkpoint_dir), early_stopping(), learning_rate_plateau()]
    )

    save_history(history=screening_net.history, file_dir=file_dir)


