import os
import numpy as np

# ❌ REMOVE backend torch
# os.environ['KERAS_BACKEND'] = 'torch'

import keras

from util.sets import *
from util.history import *
from util.hyperparameters import *

from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.losses import BinaryCrossentropy

# =========================
# MOBILE NET
# =========================
def create_screening_mobilenet():

    base_model = MobileNetV2(
        weights=None,
        include_top=False,
        input_shape=(320, 320, 3)
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    output = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    return model


# =========================
# MAIN
# =========================
if __name__ == '__main__':

    file_dir = os.path.dirname(os.path.abspath(__file__))

    kfolds = dataframe_preprocessing()

    for _ in range(5):

        print("\n==============================")

        training_set, validation_set, fold_test_n = create_sets(kfolds)

        checkpoint_dir = os.path.join(file_dir, 'checkpoints_mobile')

        print(f'KFOLD {fold_test_n}')

        # ✅ cria modelo NOVO por fold
        screening_net = create_screening_mobilenet()

        screening_net.compile(
            optimizer=SGD(
                learning_rate=0.001,
                momentum=0.9,
                nesterov=True,
                weight_decay=0.0001
            ),
            loss=BinaryCrossentropy(),
            metrics=screening_metrics()
        )

        print(screening_net.summary())

        steps_per_epoch = training_set.n // 16
        validation_steps = validation_set.n // 16

        history = screening_net.fit(
            training_set,
            steps_per_epoch=steps_per_epoch,
            epochs=100,
            validation_data=validation_set,
            validation_steps=validation_steps,
            verbose=1,
            class_weight=screening_class_weights(),
            callbacks=[
                model_checkpoint(checkpoint_dir),
                learning_rate_plateau(),
                early_stopping()
            ]
        )

        save_history(
            history=history,
            file_dir=file_dir,
            fold_test_n=fold_test_n,
            history_dir_name='mobile_fit_history'
        )