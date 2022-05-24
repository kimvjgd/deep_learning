# How to make toy data set

import tensorflow as tf

N, n_feature = 8, 5

t_weights = tf.constant([1,2,3,4,5], dtype=tf.float32)
t_bias = tf.constant([10], dtype=tf.float32)

X = tf.random.normal(mean=0, stddev=1, shape=(N, n_feature))

tem_Y = t_weights*X
Y = t_weights*X+t_bias
print(X.shape, '\n', X)
print(tem_Y.shape, '\n', tem_Y)
print(Y.shape, '\n', Y)

Y = tf.cast(Y>5, tf.int32)

# make multi class data set

import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use('seaborn')

N, n_feature = 30, 2
n_class = 5

X = tf.zeros(shape=(0, n_feature))    # 0~5
Y = tf.zeros(shape=(0,1), dtype=tf.int32)

fig, ax = plt.subplots(figsize=(5,5))
for class_idx in range(n_class):
  center = tf.random.uniform(minval=-15, maxval=15, shape=(2,))
  
  x1 = center[0] + tf.random.normal(shape=(N,1))
  x2 = center[1] + tf.random.normal(shape=(N,1))
  
  ax.scatter(x1.numpy(), x2.numpy())
  
  x = tf.concat((x1, x2), axis=1)
  y = class_idx*tf.ones(shape=(N, 1), dtype=tf.int32)
  print(x.shape, y.shape)
  
  
  
# 
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use('seaborn')

N, n_feature = 8, 2
n_class = 3

X = tf.zeros(shape=(0, n_feature))
Y = tf.zeros(shape=(0, ), dtype=tf.int32)

fig, ax = plt.subplots(figsize=(5,5))
for class_idx in range(n_class):
  center = tf.random.uniform(minval=-15, maxval=15, shape=(2,))
  
  x1 = center[0] + tf.random.normal(shape=(N, 1))
  x2 = center[1] + tf.random.normal(shape=(N, 1))
  
  x = tf.concat((x1, x2), axis=1)
  y = class_idx*tf.ones(shape=(N, ), dtype=tf.int32)
  
  ax.scatter(x[:, 0].numpy(), x[:, 1].numpy(), alpha=0.3)
  
  X = tf.concat((X, x), axis = 0)
  Y = tf.concat((Y, y), axis = 0)
  
print('X(shape/dtype/data): {} \n {}\n {}\n'.format(X.shape, X.dtype, X.numpy()))
print('Y(shape/dtype/data): {} \n {}\n {}\n'.format(Y.shape, Y.dtype, Y.numpy()))
  

# one-hot encoding
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use('seaborn')

N, n_feature = 8, 2
n_class = 3

X = tf.zeros(shape=(0, n_feature))
Y = tf.zeros(shape=(0, ), dtype=tf.int32)

fig, ax = plt.subplots(figsize=(5,5))
for class_idx in range(n_class):
  center = tf.random.uniform(minval=-15, maxval=15, shape=(2,))
  
  x1 = center[0] + tf.random.normal(shape=(N, 1))
  x2 = center[1] + tf.random.normal(shape=(N, 1))
  
  x = tf.concat((x1, x2), axis=1)
  y = class_idx*tf.ones(shape=(N, ), dtype=tf.int32)
  
  ax.scatter(x[:, 0].numpy(), x[:, 1].numpy(), alpha=0.3)
  
  X = tf.concat((X, x), axis = 0)
  # Y = tf.concat((Y, y), axis = 0)
  Y = tf.one_hot(Y, depth=n_class, dtype=tf.int32)
  
print('X(shape/dtype/data): {} \n {}\n {}\n'.format(X.shape, X.dtype, X.numpy()))
print('Y(shape/dtype/data): {} \n {}\n {}\n'.format(Y.shape, Y.dtype, Y.numpy()))
  

# Dataset Objects
import tensorflow as tf
N, n_feature = 100, 5
batch_size = 32
t_weights = tf.constant([1,2,3,4,5], dtype=tf.float32)
t_bias = tf.constant([10], dtype=tf.float32)

X = tf.random.normal(mean=0, stddev=1, shape=(N, n_feature))
Y = tf.reduce_sum(t_weights*X, axis=1) + t_bias

for batch_idx in range(N//batch_size):
  x = X[batch_idx * batch_size : (batch_idx + 1)*batch_size, ...]
  y = Y[batch_idx * batch_size : (batch_idx + 1)*batch_size, ...]

  print(x.shape, y.shape)

# simple way
import tensorflow as tf
N, n_feature = 100, 5
batch_size = 32
t_weights = tf.constant([1,2,3,4,5], dtype=tf.float32)
t_bias = tf.constant([10], dtype=tf.float32)

X = tf.random.normal(mean=0, stddev=1, shape=(N, n_feature))
Y = tf.reduce_sum(t_weights*X, axis=1) + t_bias

dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset = dataset.batch(batch_size)

for x, y in dataset:
  print(x.shape, y.shape)



# MSE Calculation
import tensorflow as tf

from tensorflow.keras.losses import MeanSquaredError

loss_object = MeanSquaredError()

batch_size = 32
predictions = tf.random.normal(shape=(batch_size, 1))
labels = tf.random.normal(shape=(batch_size, 1))

print(predictions.shape, labels.shape)

# (32,1) (32,1)
mse = loss_object(labels, predictions)
print(mse.numpy())
# y_hat & y 의 차이


# 
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError

N, n_feature = 100, 5

X = tf.random.normal(shape=(N, n_feature))
Y = tf.random.normal(shape=(N, 1))
dataset = tf.data.Dataset.from_tensor_slices((X, Y))    # X, Y 넣어주고(동피셜)
dataset = dataset.batch(batch_size)         # batch로 나눠 (동피셜)

model = Dense(units=1, activation = 'linear')
loss_object = MeanSquaredError()

for x, y in dataset:
  predictions = model(x)
  loss = loss_object(y, predictions)
  print(loss.numpy())

# 
import tensorflow as tf

from tensorflow.keras.losses import BinaryCrossentropy
batch_size = 32
n_class = 2

predictions = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=1, dtype=tf.float32)
labels = tf.random.uniform(shape=(batch_size,1), minval=0, maxval=n_class, dtype=tf.int32)

loss_object = BinaryCrossentropy()
loss = loss_object(labels, predictions)

lables = tf.cast(labels, tf.float32)


# minibatch에 따라 각각의 loss들이 나온다.

import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

N, n_feature = 100, 5
t_weights = tf.constant([1,2,3,4,5], dtype=tf.float32)
t_bias = tf.constant([10], dtype=tf.float32)

X = tf.random.normal(mean=0, stddev=1, shape=(N, n_feature))
Y = tf.reduce_sum(t_weights*X, axis=1) + t_bias
Y = tf.cast(Y>5, tf.int32)

dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset = dataset.batch(batch_size)

model = Dense(units=1, activation='sigmoid')

loss_object = BinaryCrossentropy()

for x, y in dataset:
  predictions = model(x)
  loss = loss_object(y, predictions)
  print(loss.numpy())
  

# 
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

batch_size, n_class = 16, 5

predictions = tf.random.uniform(shape=(batch_size, n_class), 
                                minval=0, maxval=1, 
                                dtype=tf.float32)

pred_sum = tf.reshape(tf.reduce_sum(predictions, axis=1), (-1,1))

print(predictions.shape, pred_sum.shape)

predictions = predictions/pred_sum

labels = tf.random.uniform(shape=(batch_size, ), minval=0, maxval=n_class, dtype=tf.int32)

loss_object = SparseCategoricalCrossentropy()
loss = loss_object(labels, predictions)

print(loss.numpy())

ce = 0
for label, prediction in zip(labels, predictions):
  print(label.shape, prediction.shape)
  print(label, prediction)
  

# SCCE with Model/Dataset

import tensorflow as tf

from tensorflow.keras.layers import Dense

from tensorflow.keras.losses import SparseCategoricalCrossentropy

N, n_feature = 100,2
n_class =5

X = tf.zeros(shape=(0, n_feature))
Y = tf.zeros(shape=(0, 1), dtype=tf.int32)

for class_idx in range(n_class):
  center = tf.random.uniform(minval=-15, maxval=15, shape=(2, ))

  x1 = center[0] + tf.random.normal(shape=(N,1))
  x2 = center[1] + tf.random.normal(shape=(N,1))
  
  x = tf.concat((x1, x2), axis=1)
  y = class_idx*tf.ones(shape=(N, 1), dtype=tf.int32)
  
  X = tf.concat((X, x), axis=0)
  Y = tf.concat((Y, y), axis=0)
  
dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset = dataset.batch(batch_size)

model = Dense(units=n_class, activation='softmax')
loss_object = SparseCategoricalCrossentropy()

for x, y in dataset:
  predcitions = model(x)
  loss = loss_object(y, predictions)
  print(loss.numpy())
  



#
from tensorflow.keras.losses import CategoricalCrossentropy
batch_size, n_class = 16 ,5
predictions = tf.random.uniform(shape=(batch_size, n_class), minval=0, maxval=1, dtype=tf.float32)
pred_sum = tf.reshape(tf.reduce_sum(predictions, axis=1), (-1,1))
predicitons = predictions/pred_sum

labels = tf.random.uniform(shape=(batch_size, ), minval=0, maxval=n_class, dtype=tf.int32)

labels = tf.one_hot(labels, n_class)

loss_object = CategoricalCrossentropy()
loss = loss_object(labels, predictions)
print(loss.numpy())


