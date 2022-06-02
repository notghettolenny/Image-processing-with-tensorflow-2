from skimage import data
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os

# Minimize console warnings by tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

""" Create a linspace using numpy.linspace (start_val, end_val, 
number of contents in the array)
"""
x = np.linspace(-3.0, 3.0, 100)
"""print(x)
print(x.shape)
print(x.dtype)
"""

# Create a linspace using tensorflow.linspace
x = tf.linspace(-3.0, 3.0, 100)
# print(x)

# Inspect the graph
g = tf.get_default_graph()

# Create a session first before we can evaluate tensors
sess = tf.Session()
# Using sess without run won't work
computed_x = sess.run(x)
""" Alternatively we can use
compute_x = x.eval(session=sess)
We tell the tensor to evaluate itself using this session
"""

# Tell the session which graph to manage
sess = tf.Session(graph=g)

# Create a new graph
g2 = tf.Graph()

# Interactive session we need not to tell the eval about the session
sess = tf.InteractiveSession()
x.eval()

# sess.close() to close the session

# Use get_shape() to know the shape of your tensor
# print(x.get_shape())
# A more friendly version uses []
# print(x.get_shape().as_list())

# Gaussian/normal curve formula implementation
# f(x) = (1/(sqrt(2pi))exp(-((mean-x)^2/(2sigma(^2))))
# mean is mu
# standard deviation is sigma
# We use z as a representation for a z-test
mean = 0.0
sigma = 1.0
z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
						(2.0 * tf.pow(sigma, 2.0)))) * 
			(1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))

# Evaluate the tensor for z
gaussian_graph = z.eval()
plt.plot(gaussian_graph)

# TODO: Understand more of this concept
# Creating a 2-D Gaussian Kernel
# (N, 1) x (1, N) inner dimensions should match, N are the result of matrix mult
# Store the number of values from our Gaussian curve
ksize = z.get_shape().as_list()[0]
"""Multiply the transposed matrix to the new shape with respect to the number of contents
of the Gaussian curve
"""
# tf.reshape arguments (tensor, shape, name)
# Multiply matrix to the transform of the matrix to get 2-D
z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))
plt.imshow(z_2d.eval())

# 4-D image convolution
# skimage data has some images to manipulate with
img = data.camera().astype(np.float32)
plt.imshow(img, cmap='gray')
# print(img.shape)

# Reshaping in 4-D means 1xHxWx1 or 1 x img.shape[0] x img.shape[1] x 1
img_4d = tf.reshape(img, [1, img.shape[0], img.shape[1], 1])
# print(img_4d.get_shape().as_list())

"""Reshape Gaussian kernel (z_2d) to required tensorflow 4-D format HxWxIXO
where H is height, W is width, I is input channel, O is output channel. We 
already have the size of the kernel which is ksize so we will use that
"""
z_4d = tf.reshape(z_2d, [ksize, ksize, 1, 1])
# print(z_4d.get_shape().as_list())

# TODO: Understand more of this concept, this part takes a lot of time to compile
# Convolve/filter an image using previous Gaussian Kernel
"""Stride means how much the window shifts by in each dimension
Value of strides can be [1,1,1,1] or [1,2,2,1]
"""
convolved = tf.nn.conv2d(img_4d, z_4d, strides=[1,1,1,1], padding='SAME')
result = convolved.eval()
# print(result.shape)

"""matplotlib can't handle showing 4-D images, we either squeeze it using numpy.squeeze()
or selecting a portion of the dimensions
"""
plt.imshow(np.squeeze(result), cmap='gray')
# plt.show()

# Create a convolution using Gabor kernel using sine wave activation
"""NOTE: The Gaussian that I've created after normalization is on the
range -3 to 3 so we use the tf.linspace() to print the linearspace
in reference to the kernel size (ksize)
"""
xs = tf.linspace(-3.0, 3.0, ksize)
ys = tf.sin(xs)
plt.plot(ys.eval())
# For multiplication, we need to convert 1-D vector into a matrix (Nx1)
ys = tf.reshape(ys, [ksize, 1])
# Repeat this wave across the matrix by mutliplication of ones
ones = tf.ones((1, ksize))
wave = tf.matmul(ys, ones)
plt.imshow(wave.eval(), cmap='gray')
# Multiply our old 2-D gaussian kernel to wave to get gabor kernel
gabor = tf.multiply(wave, z_2d)
plt.imshow(gabor.eval(), cmap='gray')
plt.show()