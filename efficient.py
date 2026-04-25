import os
import numpy as np
import keras
from util.sets import *
from util.history import *
from util.hyperparameters import *
from keras.applications import MobileNetV2, EfficientNetB0
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.losses import BinaryCrossentropy, CategoricalCrossentropy


def create_screening_efficient():

    base_model = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(320, 320, 3)
    )

    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    output = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    return model


if __name__ == '__main__':

    file_dir = os.path.dirname(os.path.abspath(__file__))

    dataframe_preprocessing()

    fold_test = 1

    for _ in range(5):

        print("\n==============================")

        path_folds = 'db\\DataCrohnIPI_2021_03\\dados.json'

        training_set, validation_set, fold_test = create_sets(path_folds, fold_test)

        checkpoint_dir = os.path.join(file_dir, 'checkpoints_efficient')

        print(f'KFOLD{fold_test}')

        # ✅ cria modelo NOVO por fold
        screening_net = create_screening_efficient()

        screening_net.compile(
            optimizer=SGD(
                learning_rate=0.001,
                momentum=0.9,
                nesterov=True,
                weight_decay=0.0001
            ),
            loss=CategoricalCrossentropy(),
            metrics=screening_metrics()
        )

        steps_per_epoch = training_set.n // 16
        validation_steps = validation_set.n // 16

        history = screening_net.fit(
            training_set,
            steps_per_epoch=steps_per_epoch,
            epochs=100,
            validation_data=validation_set,
            validation_steps=validation_steps,
            verbose=2,
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
            fold_test_n=fold_test,
            history_dir_name='efficient_fit_history'
        )

        fold_test += 1