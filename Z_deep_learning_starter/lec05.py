import tensorflow as tf

from tensorflow.keras.layers import Conv2D

N, n_H, n_W, n_C = 1, 28, 28, 1
n_filter = 1
f_size = 3

images = tf.random.uniform(minval=0, maxval=1, shape=((N, n_H, n_W, n_C)))

print(images.shape)

#
k_size = 3
images = tf.random.uniform(minval=0, maxval=1, shape=((N, n_H, n_W, n_C)))
conv = Conv2D(filters=n_filter, kernel_size=k_size)

