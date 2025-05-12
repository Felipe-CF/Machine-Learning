import numpy as np
import tensorflow as tf

# carregar o MNIST
# 60000 imagens      10000 imagens
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(x_train.shape)
print(y_train.shape)
print()
print()
print()
print()


