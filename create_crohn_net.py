import keras
from util import *
from keras.regularizers import L2
from keras.layers import BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Dense, add
from keras.models import Model  
from keras.initializers import TruncatedNormal
from keras.layers import Conv2D, BatchNormalization, Dense, Dropout


def create_load_net(file_dir=None):
    conv_net = None
    
    if not file_dir:

        inputs = keras.Input(shape=(320, 320, 3))

        res_net_layers =  Conv2D(
            kernel_size=(7, 7), 
            strides=2, 
            filters=64, 
            padding='same',
            kernel_initializer=TruncatedNormal(mean=0.0, stddev=1.0)
            )(inputs)

        res_net_layers = BatchNormalization()(res_net_layers)

        res_net_layers = LeakyReLU(alpha=0.01)(res_net_layers)

        res_net_layers = add_identity_block(res_net_layers, filters=64)

        res_net_layers = add_projection_block(res_net_layers, filters=64)

        res_net_layers = add_projection_block(res_net_layers, filters=128)

        res_net_layers = add_projection_block(res_net_layers, filters=256)

        res_net_layers = add_projection_block(res_net_layers, filters=512)
       
        res_net_layers = GlobalAveragePooling2D()(res_net_layers)

        outputs = Dense(
            units=7, 
            activation='softmax',
            kernel_initializer=TruncatedNormal(mean=0.0, stddev=1.0))(res_net_layers)
        
        return Model(inputs, outputs)
        
    else:
        checkpoint_dir = os.path.join(file_dir, 'resnet_checkpoints')

        best_model_path = os.path.join(checkpoint_dir, 'crohn_net_0.9130.keras')

        conv_net = keras.saving.load_model(best_model_path, compile=True, safe_mode=True, custom_objects=None)
    
    return conv_net


def add_identity_block(res_net_layers, filters=32, kernel_size=(3, 3)):
    skip_connection = res_net_layers

    # 1st layer
    res_net_layers = Conv2D( 
        filters=filters, 
        kernel_size=kernel_size, 
        padding="same",
        kernel_initializer=TruncatedNormal(mean=0.0, stddev=1.0))(res_net_layers)

    res_net_layers = BatchNormalization()(res_net_layers)

    res_net_layers = LeakyReLU(alpha=0.01)(res_net_layers)

    # 2nd layer
    res_net_layers = Conv2D( 
        filters=filters, 
        kernel_size=kernel_size, 
        padding="same",
        kernel_initializer=TruncatedNormal(mean=0.0, stddev=1.0))(res_net_layers)

    res_net_layers = BatchNormalization()(res_net_layers)

    res_net_layers = LeakyReLU(alpha=0.01)(res_net_layers)

    # adding residual connection
    res_net_layers = add([res_net_layers, skip_connection])

    res_net_layers = LeakyReLU(alpha=0.01)(res_net_layers)

    res_net_layers = Dropout(0.2)(res_net_layers)

    return res_net_layers


def add_projection_block(res_net_layers, filters=32,kernel_size=(3, 3)):
    skip_connection = res_net_layers

    # 1st layer
    res_net_layers = Conv2D( 
        filters=filters, 
        kernel_size=kernel_size, 
        padding="same", 
        strides=2,
        kernel_initializer=TruncatedNormal(mean=0.0, stddev=1.0))(res_net_layers)

    res_net_layers = BatchNormalization()(res_net_layers)

    res_net_layers = LeakyReLU(alpha=0.01)(res_net_layers)

    # 2nd layer
    res_net_layers = Conv2D( 
        filters=filters, 
        kernel_size=kernel_size, 
        padding="same",
        kernel_initializer=TruncatedNormal(mean=0.0, stddev=1.0))(res_net_layers)

    res_net_layers = BatchNormalization()(res_net_layers)

    # adapting the channels differents sizes
    projection_connection = Conv2D(
        filters=filters, 
        kernel_size=(1, 1),
        strides=2, 
        padding='same',
        kernel_initializer=TruncatedNormal(mean=0.0, stddev=1.0))(skip_connection)

    projection_connection = BatchNormalization()(projection_connection)

    # adding residual connection
    res_net_layers = add([res_net_layers, projection_connection])

    res_net_layers = LeakyReLU(alpha=0.01)(res_net_layers)

    res_net_layers = Dropout(0.2)(res_net_layers)

    return res_net_layers
