import torch, os
os.environ['KERAS_BACKEND'] = 'torch'

import keras
import numpy as np
from dotenv import load_dotenv
from util.sets import *
from util.history import *
from util.hyperparameters import *
from create_screening_net import *
from keras.optimizers import SGD
from keras.losses import BinaryCrossentropy
from keras.applications import MobileNetV2

def create_screening_mobilenet():

    base_model = MobileNetV2(
        weights=None,
        include_top=False,
        input_shape=(320, 320, 3)
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # 👇 igual sua ResNet
    output = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=SGD(
            learning_rate=0.001,
            momentum=0.9,
            nesterov=True,
            weight_decay=0.0001
        ),
        loss=BinaryCrossentropy(),  # 👈 mantido como você fez
        metrics=screening_metrics()
    )

    return model

if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))

    kfolds = dataframe_preprocessing()

    for _ in range(5):
        training_set, validation_set, fold_test_n = create_sets(kfolds)

        checkpoint_dir = os.path.join(file_dir, 'screening_checkpoints')

        print(f'KFOLD {fold_test_n}')

        screening_net = create_screening_mobilenet()

        screening_net.compile(
            optimizer=SGD(
                learning_rate=0.001,
                momentum=0.9,
                name='SGD',
                weight_decay=0.0001,
                nesterov=True
            ),
            loss=BinaryCrossentropy(),
            metrics=screening_metrics()
        )

        steps_per_epoch = training_set.n // 16
        validation_steps = validation_set.n // 16

        screening_net.fit(
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

    save_history(history=screening_net.history, file_dir=file_dir, fold_test_n=fold_test_n)
