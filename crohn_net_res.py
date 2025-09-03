import keras
from util import *
import numpy as np, random
from create_crohn_net import *
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint



if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))

    res_net = create_load_net()

    training_set, validation_set = create_sets()

    checkpoint_dir = os.path.join(file_dir, 'resnet_checkpoints')

    print(res_net.summary())

    res_net.fit(
        training_set, 
        steps_per_epoch=174, 
        epochs=100,
        validation_data=validation_set,
        validation_steps=43, 
        callbacks=[model_checkpoint(checkpoint_dir), early_stopping()]
    )
