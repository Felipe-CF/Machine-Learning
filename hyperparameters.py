import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.metrics import AUC, F1Score, Recall, Precision, BinaryAccuracy


def early_stopping():

    return EarlyStopping(
        monitor='val_AUC',
        min_delta=0.001,
        patience=10,
        mode='max',
        start_from_epoch=70,
        restore_best_weights=True,
    )


def model_checkpoint(checkpoint_dir):

    return ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'crohn_net_{val_AUC:.4f}.keras'),
        mode='max', # detecta automaticamente
        save_best_only=True, # salvar quando a métrica melhora
        save_weights_only=False, # somente os pesos
        monitor='val_AUC', # métrica balizadora do armazenamento (Accuracy da Validation)
        verbose=1 # logs de salvamento
    )


def learning_rate_plateau():

    return ReduceLROnPlateau(
        monitor='val_AUC',
        mode='max',
        factor=0.1, 
        patience=10,
        min_delta=0.001,
        cooldown=5,
        verbose=1
    )


def metrics():

    return [
        AUC(name='AUC', curve='ROC', multi_label=True, num_labels=7),
        Precision(name='Precision', thresholds=0.5),
        Recall(name='Recall', thresholds=0.5),
        BinaryAccuracy(name='Accuracy', threshold=0.5),
        F1Score(name='F1_score')
    ]


def class_weights():

    classes = {
        "0": 1.983,
        "1": 3.982,
        "2": 0.234,
        "3": 3.336,
        "4": 3.828,
        "5": 1.218,
        "6": 1.678
    }

    classes_weight = {int(key): value for key, value in classes.items()}

    return classes_weight







