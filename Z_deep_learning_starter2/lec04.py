import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
plt.style.use('seaborn')

# set params
N = 100
lr = 0.01
t_w, t_b = 5, -3
w, b = np.random.uniform(-3, 3, 2)

# generate dataset
x_data = np.random.randn(N, )
y_data = x_data*t_w + t_b
y_data += 0.1*np.random.randn(N, )

# visualize dataset
cmap = cm.get_cmap('rainbow', lut=N)
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x_data, y_data)
w_track, b_track = list(), list()

x_range = np.array([x_data.min*(), x_data.max()])
J_track = list()

# train model and visualized updated models
for data_idx, (x, y) in enumerate(x_data, y_data):
  w_track.append(w)
  b_track.append(b)

  # visualize updated model
  y_range = w*x_range + b
  ax.plot(x_range, y_range, color=cmap(data_idx), alpha=0.5)

  # forward propagation
  pred = x*w + b
  J = (y - pred)**2
  J_track.append(J)
  
  # jacobian
  dJ_dpred = -2*(y-pred)
  dpred_dw = x
  dpred_db = 1
  
  # backpropagation
  dJ_dw = dJ_dpred * dpred_dw
  dJ_db = dJ_dpred * dpred_db
  
  
# visualize results
fig, axes = plt.subplots(2, 1, figsize=(20, 10))
axes[0].plot(J_track)
axes[1].plot(w_track, color='darkred')
axes[1].plot(b_track, color='darkblue')

axes[0].set_ylabel('MSE', fontsize=30)
axes[0].tick_params(labelsize=20)

axes[1].axhline(y=t_w, color='darkred', linestyle=':')
axes[1].axhline(y=t_b, color='darkred', linestyle=':')
axes[1].tick_params(labelsize=20)




# set params
N, n_feature = 100, 3
lr = 0.1
t_W = np.random.uniform(-3, 3, (n_feature, 1))
t_b = np.random.uniform(-3, 3, (1, ))

W = np.random.uniform(-3, 3, (n_feature, 1))
b = np.random.uniform(-3, 3, (1, ))

# generate dataset
x_data = np.random.randn(N, n_feature)
y_data = x_data @ t_W + t_b

J_track = list()
W_track, b_track = list(), list()
for data_idx, (X, y) in enumerate(zip(x_data, y_data)):
  W_track.append(W)
  b_track.append(b.reshape(1, 1))
  
  # forward propagation
  X = X.reshape(1, -1)    # row로 미리 바꿔줌
  pred = X @ W + b
  J = (y - pred)**2
  J_track.append(J.squeeze())   # J는 (1,1) 인데 스칼라로 바꿔준다.
  
  # jacobians
  dJ_dpred = -2*(y - pred)
  dpred_dW = X
  dpred_db = 1
  
  # backpropagation
  dJ_dW = dJ_dpred * dpred_dW
  dJ_db = dJ_dpred * dpred_db
  
  print(dJ_dW.shape, dJ_db.shape)
  
  # parameter update
  W = W - lr*dJ_dW.T
  b = b - lr*dJ_db
  
  W_track = np.hstack(W_track)      # (3, 1)을 horizontal하게 쌓음
  b_track = np.concatenate(b_track).flatten() # 이어
  
  # visualize results
  fig, axes = plt.subplots(figsize=(20, 10))
  
axes[0].plot(J_track)
axes[0].set_ylabel('MSE', fontsize=30)
axes[0].tick_params(labelsize=20)

cmap = cm.get_cmap('rainbow', lut=n_feature)
for w_idx, (t_w, w) in enumerate(zip(t_W, W_track)):
  axes[1].axhline(y=t_w, color=cmap(w_idx), linestyle=':')
  axes[1].plot(w)
axes[1].axhline(y=t_b, color='black', linestyle=':')
axes[1].plot(b_track)
axes[1].tick_params(labelsize=20)

# Logistic Regression

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
np.random.seed(1)
plt.style.use('seaborn')


# set params
N = 100
lr = 0.1
t_w = np.random.uniform(-3, 3, (1,))
t_b = np.random.uniform(-3, 3, (1, ))

w = np.random.uniform(-3, 3, (1, ))
b = np.random.uniform(-3, 3, (1, ))

# generate dataset
x_data = np.random.randn(N, )
y_data = x_data * t_w + t_b
y_Data = 1/(1+np.exp(-y_data))
y_data = (y_data > 0.5).astype(np.int)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x_data, y_data)

for data_idx, (x, y) in enumerate(zip(x_data, y_data)):
  w_track.append(w)
  b_track.append(b)
  
  # visualize updated model
  y_range = w*x_range + b
  y_range = 1/(1+np.exp(-y_range))
  ax.plot(x_range, y_range, color=cmap(data_idx), alpha=0.5)
  
  # forward propagation
  z = x*w + b
  pred = 1/(1+np.exp(-z))
  J = -(y*np.log(pred) + (1-y)*np.log(1-pred))
  J_track.append(J)
  
  break