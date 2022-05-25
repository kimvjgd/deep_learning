# Max Pooling
 
import numpy as np
import tensorflow as tf
 
from tensorflow.keras.layers import MaxPooling1D
 
L, f, s = 10, 2, 1
 
x = tf.random.normal(shape=(1, L, 1))
pool_max = MaxPooling1D(pool_size=f, strides=s)
pooled_max = pool_max(x)
 
# manually
x = x.numpy().flatten()
pooled_max_man = np.zeros((L - f + 1, ))
for i in range(L-f+1):
  window = x[i:i+f]
  pooled_max_man[i] = np.max(window)
  
# max -> avg
from tensorflow.keras.layers import AveragePooling1D
# 위와 거의 똑ㅏ은데 MaxPooling1D => AveragePooling1D

# 2D
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import MaxPooling2D

N, n_H, n_W, n_C = 1, 5, 5, 1
f, s = 2, 1

x = tf.random.normal(shape=(N, n_H, n_W, n_C))
pool_max = MaxPooling2D(pool_size=f, strides=s)
pooled_max = pool_max(x)


# manually 
x = x.numpy().squeeze()

pooled_max_man = np.zeros(shape=(n_H -f +1, n_W -f +1))
for i in range(n_H -f + 1):
  for j in range(n_W -f + 1):
    window = x[i:i+f, j:j+f]
    pooled_max_man[i,j] = np.nax(window)

# average => 거의 똑같이

import math
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import MaxPooling2D

N, n_H, n_W, n_C = 1, 5, 5, 3
f, s = 2, 2

x = tf.random.normal(shape=(N, n_H, n_W, n_C))
print(np.transpose(x.numpy().squeeze(), (2,0,1)))

# ZeroPadding2D layer
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import ZeroPadding2D

images = tf.random.normal(shape=(1,5,5,3))
print(np.transpose(images.numpy().squeeze(), (2,0,1)))

zero_padding = ZeroPadding2D(padding=1)
y = zero_padding(images)

print(np.transpose(y.numpy().squeeze()),(2,0,1))



