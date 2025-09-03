from util import *
from create_net_no_res import *
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))

    dataframe_preprocessing(file_dir)

    conv_net = create_load_net(file_dir)
    # conv_net = create_load_net()

    training_set, validation_set = create_sets()

    checkpoint_dir = os.path.join(file_dir, 'model_checkpoints')

    print(conv_net.summary())

    conv_net.fit(
        training_set, 
        steps_per_epoch=174, 
        epochs=100,
        validation_data=validation_set,
        validation_steps=43, 
        callbacks=[model_checkpoint(checkpoint_dir), early_stopping()]
    )




