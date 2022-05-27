# Modules of Classifier

from multiprocessing import pool
import tensorflow as tf

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten

N, n_H, n_W, n_C = 32, 28, 28, 3
n_conv_filter = 5
k_size = 3
pool_size, pool_strides = 2, 2
batch_size = 32

x = tf.random.normal(shape=(N, n_H, n_W, n_C))

conv1 = Conv2D(filters=n_conv_filter, kernel_size = k_size,
               padding='same', activation = 'relu')
conv1_pool = MaxPooling2D(pool_size=pool_size, strides=pool_strides)

conv2 = Conv2D(filters=n_conv_filter, kernel_size = k_size,
               padding='same', activation = 'relu')
conv2_pool = MaxPooling2D(pool_size=pool_size, strides=pool_strides)

flatten = Flatten()

print(x.shape())
x = conv1(x)
x = conv1_pool(x)
x = conv2(x)
x = conv2_pool(x)

x = flatten(x)

# 
from tensorflow.keras.layers import Dense
n_neurons = [50, 25, 10]
dense1 = Dense(units=n_neurons[0], activation='relu')
dense2 = Dense(units=n_neurons[1], activation='relu')
dense3 = Dense(units=n_neurons[2], activation='softmax')

x = dense1(x)
W, B = dense1.get_weights()
x = dense2(x)
W, B = dense2.get_weights()

# Shapes in the loss Function
from tensorflow.keras.losses import CategoricalCrossentropy

y = tf.random.uniform(minval=0, maxval=10,
                      shape=(32, ),
                      dtype=tf.int32)

loss_object = CategoricalCrossentropy()
loss = loss_object(y, x)
print(loss.shape)
print(loss)


# 
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

N, n_H, n_W, n_C = 4, 28, 28, 3
n_conv_neurons = [10, 20, 30]
n_dense_neurons = [50, 30, 10]
k_size, padding = 3, 'same'
activation='relu'
pool_size, pool_strides = 2, 2

x = tf.random.normal(shape=(N, n_H, n_W, n_C))

model = Sequential()
model.add(Conv2D(filters=n_conv_neurons[0], kernel_size=k_size, padding=padding,
                 activation=activation))
model.add(MaxPooling2D(pool_size=pool_size, strides=pool_strides))
model.add(Conv2D(filters=n_conv_neurons[1], kernel_size=k_size, padding=padding,
                 activation=activation))
model.add(MaxPooling2D(pool_size=pool_size, strides=pool_strides))
model.add(Conv2D(filters=n_conv_neurons[2], kernel_size=k_size, padding=padding,
                 activation=activation))
model.add(MaxPooling2D(pool_size=pool_size, strides=pool_strides))
model.add(Flatten())

model.add(Dense(units=n_dense_neurons[0],activation=activation))
model.add(Dense(units=n_dense_neurons[1],activation=activation))
model.add(Dense(units=n_dense_neurons[2],activation='softmax'))

predictions = model(x)
print(predictions.shape)



model = Sequential()
for n_conv_neuron in n_conv_neurons:
  model.add(Conv2D(filters=n_conv_neuron, kernel_size=k_size, padding=padding, activation=activation))
  model.add(MaxPooling2D(pool_size=pool_size, strides = pool_strides))
model.add(Flatten())

for n_dense_neuron in n_dense_neurons:
  model.add(Dense(units=n_dense_neuron, activation=activation))
model.add(Dense(units=n_dense_neurons[-1], activation='softmax'))


# 
n_dense_neurons=[30,20,10]
k_size, padding = 3, 'same'
activation = 'relu'
pool_size, pool_strides = 2, 2

x = tf.random.normal(shape=(N, n_H, n_W, n_C))

class TestCNN(Model):
  def __init__(self):
    super(TestCNN, self).__init__()
    
    self.conv1 = Conv2D(filters=n_conv_neurons[0], kernel_size=k_size, padding=padding, acitvation=activation)
    self.conv1_pool = MaxPooling2D(pool_size = pool_size, strides=pool_strides)
    
    self.conv2 = Conv2D(filters=n_conv_neurons[1], kernel_size=k_size, padding=padding, activation=activation)
    self.conv2_pool = MaxPooling2D(pool_size=pool_size, strides=pool_strides)
    
    self.conv3 = Conv2D(filters=n_conv_neurons[2], kernel_size=k_size, padding=padding, activation=activation)
    self.conv3_pool = MaxPooling2D(pool_size=pool_size, strides=pool_strides)

    self.flatten = Flatten()
    
    self.dense1 = Dense(units=n_dense_neurons[0], activation = activation)
    self.dense2 = Dense(units=n_dense_neurons[1], activation = activation)
    self.dense3 = Dense(units=n_dense_neurons[2], activation = 'softmax')


  def call(self, x):
    x = self.conv1(x)
    x = self.conv1_pool(x)
    
    x = self.conv2(x)
    x = self.conv2_pool(x)
    
    x = self.conv3(x)
    x = self.conv3_pool(x)
    
    x = self.flatten(x)
    
    x = self.dense1(x)
    x = self.dense2(x)
    x = self.dense3(x)
    return x
  
x = tf.random.normal(shape=(N, n_H, n_W, n_C))
model = TestCNN()
y = model(x)


# 
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

class MyConv(Layer):
  def __init__(self, n_neuron):
    super(MyConv, self).__init__()
    self.conv = Conv2D(filters=n_neuron, kernel_size=k_size, padding=padding, activation=activation)
    
    self.conv_pool = MaxPooling2D(pool_size = pool_size, strides=pool_strides)
  
  def call(self, x):
    x = self.conv(x)
    x = self.conv_pool(x)
    return x
model = Sequential()
model.add(MyConv(n_conv_neurons[0]))
model.add(MyConv(n_conv_neurons[1]))
model.add(MyConv(n_conv_neurons[2]))
model.add(Flatten())

model.add(Dense(units=n_dense_neurons[0], activation=activation))
model.add(Dense(units=n_dense_neurons[1], activation=activation))
model.add(Dense(units=n_dense_neurons[2], activation='softmax'))


class MyConv(Layer):
  def __init__(self, n_neuron):
    super(MyConv, self).__init__()
    
    self.conv = Conv2D(filters=n_neuron, kernel_size=k_size, padding = padding, activation=activation)
    self.conv_pool = MaxPooling2D(pool_size = pool_size, strides=pool_strides)
  
  def call(self, x):
    x = self.conv(x)
    x = self.conv_pool(x)
    return x
class TestCNN(Model):
  def __init__(self):
    super(TestCNN, self).__init__()
    self.conv1 = MyConv(n_conv_neurons[0])
    self.conv2 = MyConv(n_conv_neurons[1])
    self.conv2 = MyConv(n_conv_neurons[2])
    self.flatten = Flatten()
    
    self.dense1 = Dense(units=n_dense_neurons[0], activation=activation)
    self.dense1 = Dense(units=n_dense_neurons[1], activation=activation)
    self.dense1 = Dense(units=n_dense_neurons[2], activation='softmax')
  
  def call(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    
    x = self.dense1(x)
    x = self.dense2(x)
    x = self.dense3(x)
    return x


#
class TestCNN(Model):
  def __init__(self):
    super(TestCNN, self).__init__()
    self.fe = Sequential()
    
    self.fe.add(MyConv(n_conv_neurons[0]))
    self.fe.add(MyConv(n_conv_neurons[1]))
    self.fe.add(MyConv(n_conv_neurons[2]))
    self.fe.add(Flatten())
    
    self.classifier = Sequential()
    self.classifier.add(Dense(units=n_dense_neurons[0], activation=activation))
    self.classifier.add(Dense(units=n_dense_neurons[1], activation=activation))
    self.classifier.add(Dense(units=n_dense_neurons[2], activation='softmax'))
    
    self.dense1 = Dense(units=n_dense_neurons[0], activation=activation)
    self.dense1 = Dense(units=n_dense_neurons[1], activation=activation)
    self.dense1 = Dense(units=n_dense_neurons[2], activation='softmax')
    
  def call(self, x):
    x = self.fe(x)
    x = self.classifier(x)
    
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling2D

class LeNet(Model):
  def __init__(self):
    super(LeNet, self).__init__()
    
    self.conv1 = Conv2D(filters=6, kernel_size=5, padding='smae', activation='tanh')
    self.conv1_pool = AveragePooling2D(pool_size=2, strides=2)
    self.conv2 = Conv2D(filters=16, kernel_size=5, padding='valid', activation='tanh')  # valid는 default : 없는 것과 마찬가지
    self.conv2_pool = AveragePooling2D(pool_size=2, strides=2)
    self.conv3 = Conv2D(filters=120, kernel_size=5, padding='valid', activation='tanh')
    self.flatten = Flatten()
    
    self.dense1 = Dense(units=84, activation='tanh')
    self.dense2 = Dense(units=10, activation='softmax')
  
  def call(self, x):
    x = self.conv1(x)
    x = self.conv1_pool(x)
    
    x = self.conv2(x)
    x = self.conv2_pool(x)
    
    x = self.conv3(x)
    x = self.flatten(x)
    
    x = self.dense1(x)
    x = self.dense2(x)
    return x
    