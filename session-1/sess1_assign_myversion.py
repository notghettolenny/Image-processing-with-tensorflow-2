from libs import utils
from skimage import data
from skimage.transform import resize
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

image_files = [os.path.join('celeba_dataset_session1_assign', image_i)
               for image_i in os.listdir('celeba_dataset_session1_assign')
               if '.jpg' in image_i]

image_read = [plt.imread(image_i)[..., :3]
               for image_i in image_files]
image_read = [utils.imcrop_tosquare(image_i)
              for image_i in image_read]
image_read = [resize(image_i, (100, 100))
              for image_i in image_read]
images = np.array(image_read).astype(np.float32)

sess = tf.Session()

mean_images_4d = tf.reduce_mean(images, axis=0, keep_dims=True)
subtract = images - mean_images_4d

# Do not use this type of deviation since it is not simplified to the nearest tens
# Using std_real (min, max) -48.3935 47.8217
# Using std_modified (min, max) -4.83935 4.78217
std_real = sess.run(tf.sqrt(tf.reduce_mean(subtract * subtract, axis=0) / images.shape[0]))
std_real_show = std_real / np.max(std_real)

std_mod = sess.run(tf.sqrt(tf.reduce_mean(subtract * subtract, axis=0)))
std_mod_show = std_mod / np.max(std_mod)

fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True, sharex=True)
axs[0].imshow(std_real_show)
axs[0].set_title('Real deviation ' + str(std_real_show.shape))
axs[1].imshow(std_mod_show)
axs[1].set_title('Modified deviation ' + str(std_mod_show.shape))
plt.show(fig.show)

# 0-1 normalization: (x - min(x)) / (max(x) - min(x))
normalize = sess.run(tf.divide(subtract, std_mod))
print(np.min(normalize), np.max(normalize))
# normalize_show = tf.divide(tf.subtract(normalize, np.min(normalize)), tf.subtract(np.max(normalize), np.min(normalize)))
normalize_show = tf.divide(normalize - np.min(normalize), np.max(normalize) - np.min(normalize))
# normalize_show = (normalize - np.min(normalize)) / (np.max(normalize) - np.min(normalize))
plt.imshow(utils.montage(normalize_show, 'normalized.png'))
plt.show()

# Convolution
ksize = 16
kernel = np.concatenate([utils.gabor(ksize) [:, :, np.newaxis] for i in range(3)], axis=2)
kernel_4d = sess.run(tf.reshape(kernel, [ksize, ksize, 3, 1]))
# Input channel is 3 (rgb) output channel is just 1
assert(kernel_4d.shape == (ksize, ksize, 3, 1))
plt.figure(figsize=(5, 5))
plt.imshow(kernel_4d[:, :, 0, 0], cmap='gray')
# plt.imsave(arr=kernel_4d[:, :, 0, 0], fname='kernel.png', cmap='gray')
plt.show()

convolved = sess.run(tf.nn.conv2d(mean_images_4d, kernel_4d, strides=[1, 1, 1, 1], padding='SAME'))
# convolved_show = (convolved - min(convolved)) / (np.max(convolved) - np.min(convolved))
# print(convolved_show.shape)
plt.figure(figsize=(10, 10))
plt.imshow(utils.montage(convolved[...,0], 'convolved.png'), cmap='gray')
plt.show()