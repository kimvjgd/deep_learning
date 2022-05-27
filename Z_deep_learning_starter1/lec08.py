import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# set params
N = 100
lr = 0.01
t_w, t_b = 5, -3
w, b = np.random.uniform(-3,3,2)

# generate dataset
x_data = np.random.randn(N, )
y_data = x_data*t_w + t_b

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(x_data, y_data)
