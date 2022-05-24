# Dense layer 구현

import tensorflow as tf

from tensorflow.keras.layers import Dense

N, n_feature = 1, 10
X = tf.random.normal(shape=(N, n_feature))

n_neuron = 3
dense = Dense(units = n_neuron, activation = 'sigmoid')
Y = dense(X)



# multi dense layers

import tensorflow as tf
from tensorflow.keras.layers import Dense

N, n_feature = 4, 10
X = tf.random.normal(shape=(N, n_feature))

n_neurons = [3, 5]
dense1 = Dense(units = n_neurons[0], activation='sigmoid')
dense2 = Dense(units = n_neurons[1], activation='sigmoid')

A1 = dense1(X)
Y = dense2(A1)

# Dense Layers with Python List
import tensorflow as tf

from tensorflow.keras.layers import Dense

N, n_feature = 4, 10
X = tf.random.normal(shape=(N, n_feature))

n_neurons = [1,2,3,4,5,6,76,8,8,10]

dense_layers = list()
for n_neuron in n_neurons:
  dense = Dense(units=n_neuron, activation='relu')
  dense_layers.append(dense)
  print(len(dense_layers))

for dense in dense_layers:
  X = dense(X)
  
# or
for dense_idx, dense in enumerate(dense_layeres):
  X = dense(X)
  print(dense_idx+1)
  print(X.shape)
Y=X

# Sequential 은 dense layer가 몇개든 포함해준다.
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

n_neurons = [3,4,5,6]

# list
model = list()
for n_neuron in n_neurons:
  model.append(Dense(units=n_neuron, activation='sigmoid'))

# Sequential
model = Sequential()
for n_neuron in n_neurons:
  model.add(Dense(units=n_neuron, activation = 'sigmoid'))
  
# Sequential 노가다
model = Sequential()
model.add(Dense(units=10, activation='sigmoid'))
model.add(Dense(units=20, activation='sigmoid'))


# Model implementation with model-subclassing
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

class TestModel(Model):
  def __init__(self):
    super(TestModel, self).__init__()
    
    self.dense1 = Dense(units=10, activation='sigmoid')
    self.dense2 = Dense(units=20, activation='sigmoid')
  
  def call(self, x):
    x = self.dense1(x)
    x = self.dense2(x)
    return x
  
model = TestModel()
Y = model(X)

#
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

X = tf.random.normal(shape=(4,10))

model = Sequential()
model.add(Dense(units=10, activation='sigmoid'))
model.add(Dense(units=20, activation='sigmoid'))

Y = model(X)

print(type(model.layers))
print(model.layers)         # dense layer들이 model.layers안에 들어가 있다.

# Trainable Variables in Models
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

X = tf.random.normal(shape=(4, 10))

model = Sequential()
model.add(Dense(units=10, activation='sigmoid'))
model.add(Dense(units=20, activation='sigmoid'))

Y = model(X)
print(type(model.trainable_variables))
print(len(model.trainable_variables))

for train_var in model.trainable_variables:
  print(train_var.shape)    # weight & bias 가 들어가 있다.
  