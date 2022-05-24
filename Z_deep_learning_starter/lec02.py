# Dense layer 구현

import tensorflow as tf

from tensorflow.keras.layers import Dense

N, n_feature = 1, 10
X = tf.random.normal(shape=(N, n_feature))

n_neuron = 3
dense = Dense(units = n_neuron, activation = 'sigmoid')
Y = dense(X)

