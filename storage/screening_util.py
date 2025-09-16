import os, json
import pandas as pd
import matplotlib as plt
from PIL import Image, UnidentifiedImageError
from keras_preprocessing.image import ImageDataGenerator
from util.preprocessing import dataframe_preprocessing


def generate_grafics(history):
    auc = history['AUC']

    val_auc = history['val_AUC']

    loss = history['binary_crossentropy']

    val_loss = history['val_binary_crossentropy']

    binary_crossentropy = history['binary_crossentropy']

    val_binary_crossentropy = history['val_binary_crossentropy']

    epochs = history['epoch']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Training History - CrohnNet', fontsize=16)
    
    # Gráfico de AUC (o equivalente a acurácia no seu log)
    axes[0].plot(epochs, auc, label='Traning AUC', color='blue')
    axes[0].plot(epochs, val_auc, label='Validation AUC', color='red', linestyle='--')
    axes[0].set_title('Accuracy - AUC vs. Epochs')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('AUC')
    axes[0].legend()
    axes[0].grid(True)
    
    # Gráfico de Binary Cross-Entropy
    axes[1].plot(epochs, binary_crossentropy, label='Traning BCE', color='blue')
    axes[1].plot(epochs, val_binary_crossentropy, label='Validation BCE', color='red', linestyle='--')
    axes[1].set_title('Binary Cross-Entropy vs. Epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Binary Cross-Entropy')
    axes[1].legend()
    axes[1].grid(True)

    # Gráfico de Loss
    axes[2].plot(epochs, loss, label='Traning Loss', color='blue')
    axes[2].plot(epochs, val_loss, label='Validation Loss', color='red', linestyle='--')
    axes[2].set_title('Loss Fuction vs. Epochs')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

