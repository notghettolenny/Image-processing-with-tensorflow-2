from skimage import data
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os

# Minimize console warnings by tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

sess = tf.InteractiveSession()

img_dataset = [os.path.join('celeba_dataset_minified', img_i)
			   for img_i in os.listdir('celeba_dataset_minified')
			   if '.jpg' in img_i]

img_dataset_read = [plt.imread(img_i)
					for img_i in img_dataset]

img_data = np.array(img_dataset_read)

img_data_mean = np.mean(img_data, axis=0)
img_data_std = np.std(img_data, axis=0)
img_normalized = ((img_data[0] - img_data_mean) / img_data_std)

"""plt.hist(img_normalized.ravel(), 20)
print(img_normalized.shape)
plt.show()
"""

# The image tensor
img = tf.placeholder(tf.float32, shape=[None, None], name='img')

# Make 2-D to 3-D (HxW) to (HxWx1)
"""tf.expand_dims() takes two parameters, the base tensor and the column where
we want to insert the new dimension
"""
# Insert new dimension to column two [x: y: <here>] cf. [0: 1: 2]
img_3d = tf.expand_dims(img, 2)
dims = img_3d.get_shape()
print(dims)

# Insert new dimension to column zero or the start [<here>: y, z, a] cf. [0: 1: 2: 3]
img_4d = tf.expand_dims(img_3d, 0)
print(img_4d.get_shape().as_list())

# Create placeholders for gabor's params
mean = tf.placeholder(tf.float32, name='mean')
sigma = tf.placeholder(tf.float32, name='sigma')
ksize = tf.placeholder(tf.int32, name='ksize')

# Redo set of operations for creation of gabor kernel
# Linspace
x = tf.linspace(-3.0, 3.0, ksize)
# Gaussian curve or normal distrib curve
z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
                   (2.0 * tf.pow(sigma, 2.0)))) *
      (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))
# 2-D matrix [Nx1] x [1xN]
z_2d = tf.matmul(
	tf.reshape(z, tf.stack([ksize, 1])),
	tf.reshape(z, tf.stack([1, ksize])))
ys = tf.sin(x)
ys = tf.reshape(ys, tf.stack([ksize, 1]))
ones = tf.ones(tf.stack([1, ksize]))
# Repeatedly multiply one to ys
wave = tf.matmul(ys, ones)
gabor = tf.multiply(wave, z_2d)
gabor_4d = tf.reshape(gabor, tf.stack([ksize, ksize, 1, 1]))

# The convolution part takes a little longer time to compile
# Convolve the two
convolved = tf.nn.conv2d(img_4d, gabor_4d, strides=[1, 1, 1, 1], padding='SAME', name='convolved')
convolved_img = convolved[0, :, :, 0]

# Show result
result = convolved_img.eval(feed_dict={
	img: data.camera(),
	mean: 0.0,
	sigma: 1.0,
	ksize: 100
	})
plt.imshow(result, cmap='gray')
plt.show()