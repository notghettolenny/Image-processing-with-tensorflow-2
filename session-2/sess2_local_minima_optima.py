from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

plt.style.use('ggplot')

# Single parameter gradient descent
# TODO: Search as to why the iteration stops at the nearest local minima
""" Local minima/optima 
Gradient descent functions contain "minima" which is the lowest value
or point and maxima or the highest value. Local minima/maxima means the 
lowest/highest value of the local wave/trough
"""
fig = plt.figure(figsize=(10, 6))
ax = fig.gca()
x = np.linspace(-1, 1, 200)
hz = 10
cost = np.sin(hz*x)*np.exp(-x)
# Get the difference between every value
gradient = np.diff(cost)
ax.plot(x, cost)
ax.set_ylabel('Cost')
ax.set_xlabel('Some Parameter')
# Number of iterations means the number of which the running/learning repeats
n_iterations = 500
cmap = plt.get_cmap('coolwarm')
c_norm = colors.Normalize(vmin=0, vmax=n_iterations)
scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)
""" init_p means the location of the nth(init_p) number in the linear space or
where should the point start, down to the nearest local minima (if the gradient 
is negative)
"""
init_p = 120 # np.random.randint(len(x)*0.2, len(x)*0.8)
# Means how the iteration should move forward
learning_rate = 1.0
# Implement a negative gradient of the function
for iter_i in range(n_iterations):
	# The dots must go down since it is decremented
	init_p -=learning_rate * gradient[int(init_p)]
	# Parameters x, y, circle color, transparency, color
	ax.plot(x[int(init_p)], cost[int(init_p)], 'ro', alpha=(iter_i + 1) / n_iterations, color=scalar_map.to_rgba(iter_i))
plt.show()

# Two-parameter gradient descent
# TODO: Understand this concept
fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection='3d')
x, y = np.mgrid[-1:1:0.02, -1:1:0.02]
X, Y, Z = x, y, np.sin(hz*x)*np.exp(-x)*np.cos(hz*y)*np.exp(-y)
ax.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.75, cmap='jet', shade=False)
ax.set_ylabel('Some Parameter 2')
ax.set_xlabel('Some Parameter 1')
ax.set_zlabel('Cost')
plt.show()

# Plot different learning rates
fig, axs = plt.subplots(1, 3, figsize=(20, 6))
for rate_i, learning_rate in enumerate([0.01, 1.0, 500.0]):
	ax = axs[rate_i]
	x = np.linspace(-1, 1, 200)
	hz = 10
	cost = np.sin(hz*x)*np.exp(-x)
	gradient = np.diff(cost)
	ax.plot(x, cost)
	ax.set_ylabel('Cost')
	ax.set_xlabel('Some Parameter')
	ax.set_title(str(learning_rate))
	n_iterations = 500
	cmap = plt.get_cmap('coolwarm')
	c_norm = colors.Normalize(vmin=0, vmax=n_iterations)
	init_p = 120
	for iter_i in range(n_iterations):
		init_p -= learning_rate * gradient[int(init_p)]
		ax.plot(x[int(init_p)], cost[int(init_p)], 'ro', alpha=(iter_i + 1) / n_iterations, color=scalar_map.to_rgba(iter_i))
plt.show()