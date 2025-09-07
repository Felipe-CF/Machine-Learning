import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.metrics import AUC, F1Score, Recall, Precision, BinaryAccuracy


def early_stopping():

    return EarlyStopping(
        monitor='val_auc',
        min_delta=0.001,
        patience=10,
        mode='max',
        start_from_epoch=70,
        restore_best_weights=True,
    )


def model_checkpoint(checkpoint_dir):

    return ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'crohn_net_{val_auc:.4f}.keras'),
        mode='max', # detecta automaticamente
        save_best_only=True, # salvar quando a métrica melhora
        save_weights_only=False, # somente os pesos
        monitor='val_auc', # métrica balizadora do armazenamento (Accuracy da Validation)
        verbose=1 # logs de salvamento
    )


def learning_rate_plateau():

    return ReduceLROnPlateau(
        monitor='val_auc',
        mode='max',
        factor=0.5, 
        patience=10,
        min_delta=0.001,
        cooldown=10,
        verbose=1
    )


def metrics():

    return [
        AUC(name='AUC', curve='ROC', multi_label=True, num_labels=7),
        Precision(name='Precision', thresholds=0.5),
        Recall(name='Recall', thresholds=0.5),
        BinaryAccuracy(name='Accuracy', threshold=0.5),
        F1Score(name='F1 score')
    ]






