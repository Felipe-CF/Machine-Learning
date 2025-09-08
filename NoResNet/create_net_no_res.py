import keras
from util_classification_net import *
from keras import layers
from keras.models import Sequential
from keras.optimizers import SGD
from keras.initializers import TruncatedNormal
from keras.layers import Conv2D, BatchNormalization, Dense, Dropout, Activation


def create_load_net(file_dir=None):
    conv_net = None
    
    if not file_dir:
        conv_net = Sequential()

        # ConvLayer2D = 7x7 conv, 64, /2
        conv_net = create_conv_layer2D(
            conv_net, 
            kernel_size=(7, 7),
            resource_map_size=32,
            input_shape=(320, 320, 3),
            )
        
        # ConvLayer2D = 3x3 conv, 64, /2
        conv_net = create_conv_layer2D(
            conv_net, 
            strides=(2, 2),
            resource_map_size=64, 
            )
        
        # ConvLayer2D = 3x3 conv, 64 x2
        conv_net = create_conv_layer2D(
            conv_net, 
            n_layer=2,
            resource_map_size=64, 
            )
        
        # ConvLayer2D = 3x3 conv, 128, /2
        conv_net = create_conv_layer2D(
            conv_net,
            resource_map_size=128, 
            strides=(2, 2)
            )
        
        # ConvLayer2D = 3x3 conv, 256, /2
        conv_net = create_conv_layer2D(
            conv_net,
            resource_map_size=256, 
            strides=(2, 2)
            )
        
        conv_net = create_conv_layer2D(
            conv_net,
            resource_map_size=512, 
            strides=(2, 2),
            )
        
        conv_net.add(layers.GlobalAveragePooling2D())

        conv_net.add(
            Dense(
                units=7, 
                activation='sigmoid',
                kernel_initializer=TruncatedNormal(mean=0.0, stddev=1.0)
                )
            )

        conv_net.compile(
            optimizer=SGD(momentum=0.99), 
            loss=keras.losses.BinaryCrossentropy(), 
            metrics=[keras.metrics.BinaryCrossentropy(), keras.metrics.AUC(name='auc')]
            )
        
    else:
        checkpoint_dir = os.path.join(file_dir, 'model_checkpoints')

        best_model_path = os.path.join(checkpoint_dir, 'crohnv_net_val_auc_0.9056.keras')

        conv_net = keras.saving.load_model(best_model_path, compile=True, safe_mode=True, custom_objects=None)
    
    return conv_net


def create_conv_layer2D(conv_net, kernel_size=(3, 3), resource_map_size=64, strides=(1, 1), input_shape=None, n_layer=1, activation_layer=layers.LeakyReLU(alpha=0.01)):
    
    for _ in range(n_layer):

        if not input_shape:
            conv_net.add(Conv2D(
                resource_map_size, 
                kernel_size=kernel_size,
                kernel_initializer=TruncatedNormal(mean=0.0, stddev=1.0),
                strides=strides
                ))

        else:
            conv_net.add(Conv2D(
                resource_map_size,
                kernel_size=kernel_size, 
                input_shape=input_shape,
                kernel_initializer=TruncatedNormal(mean=0.0, stddev=1.0),
                strides=strides
            ))

        conv_net.add(BatchNormalization())
        
        conv_net.add(activation_layer)
        
    return conv_net


