# binary classifiers 구현

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
plt.style.use('seaborn')

p_np = np.linspace(0.01, 0.99, 100)
p_tf = np.linspace(0.01, 0.99, 100)

# odds
odds_np = p_np/(1-p_np)
odds_tf = p_tf/(1-p_tf)

# logit
logit_np = np.log(odds_np)
logit_tf = np.log(odds_tf)

fig, axes = plt.subplots(2,1,figsize=(15,10), sharex=True)

axes[0].plot(p_np, odds_np)
axes[1].plot(p_np, logit_np)

xticks = np.arange(0, 1.1, 0.2)
axes[0].tick_params(labelsize=15)
axes[0].set_xticks(xticks)
axes[0].set_ylabel('Odds', fontsize=20, color='darkblue')
axes[1].tick_params(labelsize=15)
axes[1].set_xticks(xticks)
axes[1].set_ylabel('Probability', fontsize=20, color='darkblue')
axes[1].set_ylabel('Logit', fontsize=20, color='darkblue')

#
import tensorflow as tf
from tensorflow.keras.layers import Activation

X = tf.linspace(-10, 10, 100)
sigmoid = Activation('sigmoid')(X)
fig, ax = plt.subplots(figsize = (10,5))
ax.plot(X.numpy(), sigmoid.numpy())


#
import matplotlib.pyplot as splt
import tensorflow as tf
from tensorflow.keras.layers import Dense

plt.style.use('seaborn')

X = tf.random.normal(shape=(100,1))
dense = Dense(units=1, activation='sigmoid')

Y = dense(X)
fig, ax = plt.subplots(figsize=(7,7))
ax.scatter(X.numpy().flatten(), Y.numpy().flatten())

#
import tensorflow as tf

from tensorflow.keras.layers import Activation

logit = tf.random.uniform(shape=(2, 5), minval=-10, maxval=10)

softmax_value = Activation('softmax')(logit)
softmax_sum = tf.reduce_sum(softmax_value, axis=1)

print("Logits: \n", logit.numpy())
print("Probabilities: \n", softmax_value.numpy())
print("Sum of softmax values: \n", softmax_sum)


# 
import tensorflow as tf
from tensorflow.keras.layers import Dense

logit = tf.random.uniform(shape=(8,5), minval = -10, maxval=10)
dense = Dense(units=8, activation='softmax')
Y = dense(logit)
print(tf.reduce_sum(Y, axis=1))


#
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

class TestModel(Model):
  def __init__(self):
    super(TestModel, self).__init__()
    
    self.dense1 = Dense(units=8, activation='relu')
    self.dense2 = Dense(units=5, activation='relu')
    self.dense3 = Dense(units=3, activation='softmax')
    
  def call(self, x):
    print("X: {}\n{}\n".format(x.shape,x.numpy()))

    x = self.dense1(x)
    print("A1: {}\n{}\n".format(x.shape, x.numpy()))

    x = self.dense2(x)
    print("A2: {}\n{}\n".format(x.shape, x.numpy()))

    x = self.dense3(x)
    print("Y: {}\n{}\n".format(x.shape, x.numpy()))
    print("Sum of vectors: {}\n".format(tf.reduce_sum(x, axis=1)))
    return X
model = TestModel()
X = tf.random.uniform(shape=(8,5), minval = -10, maxval = 10)
Y = model(X)
    
          