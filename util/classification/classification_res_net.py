import keras
from classification_util import *
from hyperparameters import *
from create_classification_net import *
from keras.optimizers import SGD


if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))

    res_net = create_load_net()
    # res_net = create_load_net(file_dir)

    res_net.compile(
        optimizer=SGD(learning_rate=0.0001, momentum=0.9, name='SGD', weight_decay=0.0001), 
        loss=keras.losses.BinaryCrossentropy(), 
        metrics=classification_metrics()
    )

    training_set, validation_set = create_sets()

    checkpoint_dir = os.path.join(file_dir, 'classification_checkpoints')

    # print(res_net.summary())

    res_net.fit(
        training_set, 
        steps_per_epoch=136, 
        epochs=100,
        validation_data=validation_set,
        validation_steps=34,
        class_weight=classification_class_weights(), 
        callbacks=[model_checkpoint(checkpoint_dir), early_stopping(), learning_rate_plateau()]
    )

    save_history(history=res_net.history, file_dir=file_dir)


