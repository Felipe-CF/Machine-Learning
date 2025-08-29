import keras
from util import *
from keras import layers
from keras.models import Sequential
from keras.losses import CategoricalCrossentropy
from keras.optimizers import SGD
from keras.layers import Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dense, Dropout, Activation


def create_load_net(file_dir=None):
    conv_net = None
    
    if not file_dir:
       
        conv_net.add(layers.GlobalAveragePooling2D())
        
        conv_net.add(Dense(units=1000, activation=layers.LeakyReLU(alpha=0.01)))

        conv_net.add(Dense(units=7, activation='softmax'))

        conv_net.compile(
            optimizer=SGD(momentum=0.99), 
            loss=CategoricalCrossentropy(), 
            metrics=['accuracy']
            )
        
    else:
        checkpoint_dir = os.path.join(file_dir, 'model_checkpoints')

        best_model_path = os.path.join(checkpoint_dir, 'conv_net_accuracy_0.69.keras')

        conv_net = keras.saving.load_model(best_model_path, compile=True, safe_mode=True, custom_objects=None)
    
    return conv_net


def create_conv_layer2D(conv_net, kernel_size=(3, 3), resource_map_size=256, strides=(1, 1), input_shape=None, n_layer=1, activation_layer=layers.LeakyReLU(alpha=0.01)):
    
    for _ in range(n_layer):

        if not input_shape:
            conv_net.add(Conv2D(
                resource_map_size, 
                kernel_size=kernel_size, 
                strides=strides
                ))

        else:
            conv_net.add(Conv2D(
                resource_map_size,
                kernel_size=kernel_size, 
                input_shape=(320, 320, 3), 
                strides=strides
            ))

        conv_net.add(BatchNormalization())
        
        conv_net.add(activation_layer)
        
    return conv_net


