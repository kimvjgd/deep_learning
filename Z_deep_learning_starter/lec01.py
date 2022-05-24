# Affine function
import tensorflow as tf
from tensorflow.keras.layers import Dense

x = tf.constant([[10.]])
print(x)
dense = Dense(units=1, activation='linear')

y_tf = dense(x)
print(y_tf)
W, B = dense.get_weights()
print(W,B)
y_man = tf.linalg.matmul(x, W) + B



# params initialization
from tensorflow.keras.layers import Dense
from tensorflow.keras.intitializers import Constant

x = tf.constant([[10.]])

w, b = tf.constant(10.), tf.constant(20.)
w_init, b_init = Constant(w), Constant(b)

dense = Dense(units=1,
              activation='linear',
              kernel_initializer=w_init,
              bias_initializer=b_init)

y_tf = dense(x)
print(y_tf)
W, B = dense.get_weights()


x = tf.random.uniform(shape=(1, 10), minval=0, maxval=10)
dense = Dense(units = 1)
y_tf = dense(x)
W, B = dense.get_weights()

y_man = tf.linalg.matmul(x, W) + B



# 이제는 affine function에다가 activation function을 붙여보자

# activation만 만들어보기

import tensorflow as tf
from tensorflow.math import exp, maximum
from tensorflow.keras.layers import Activation
x=tf.random.normal(shape=(1,5))

sigmoid = Activation('sigmoid')
tanh = Activation('tanh')
relu = Activation('relu')

y_sigmoid_tf = sigmoid(x)
y_tanh_tf = tanh(x)
y_relu_tf = relu(x)




# Activation in Dense Layer
import tensorflow as tf

from tensorflow.keras.layers import Dense

x = tf.random.normal(shape=(1,5))
dense_sigmoid = Dense(units = 1, activation = 'sigmoid')
dense_tanh = Dense(units = 1, activation = 'tanh')
dense_relu = Dense(units = 1, activation = 'relu')

y_sigmoid = dense_sigmoid(x)
y_tanh = dense_tanh(x)
y_relu = dense_relu(x)




# Minibatches
import tensorflow as tf
from tensorflow.keras.layers import Dense

N, n_feature = 8, 10
x = tf.random.normal(shape=(N, n_feature))
print(x.shape)

dense = Dense(units=1, activation='relu')
y = dense(x)

W, B = dense.get_weights()


# tensorflow & manual comparison

x = tf.random.normal(shape=(N, n_feature))

dense = Dense(units=1, activation='sigmoid')
y_tf = dense(x)

W, B = dense.get_weights()

y_man = tf.linalg.matmul(x, W) + B
y_man = 1/(1+tf.math.exp(-y_man))
