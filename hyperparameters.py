import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.metrics import AUC, F1Score, Recall, Precision, BinaryAccuracy


def early_stopping():

    return EarlyStopping(
        monitor='val_AUC',
        min_delta=0.1,
        patience=10,
        mode='max',
        start_from_epoch=50,
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
        patience=5,
        min_delta=0.1,
        cooldown=0,
        verbose=1
    )


def screening_metrics():

    return [
        AUC(name='AUC', curve='ROC'),
        Precision(name='Precision', thresholds=0.5),
        Recall(name='Recall', thresholds=0.5),
        BinaryAccuracy(name='Accuracy', threshold=0.5),
        F1Score(name='F1_score', threshold=0.5, average='weighted')
    ]


def screening_class_weights():

    classes = {
        "0": 0.8234,
        "1": 1.2861,
    }

    classes_weight = {int(key): value for key, value in classes.items()}

    return classes_weight


def classification_metrics():

    return [
        AUC(name='AUC', curve='ROC', multi_label=True, num_labels=6),
        Precision(name='Precision', thresholds=0.5),
        Recall(name='Recall', thresholds=0.5),
        BinaryAccuracy(name='Accuracy', threshold=0.5),
        F1Score(name='F1_score')
    ]


def classification_class_weights():

    classes = {
        "0": 0.903,
        "1": 1.813,
        "2": 1.521,
        "3": 1.744,
        "4": 0.556,
        "5": 0.763,
    }

    classes_weight = {int(key): value for key, value in classes.items()}

    return classes_weight

