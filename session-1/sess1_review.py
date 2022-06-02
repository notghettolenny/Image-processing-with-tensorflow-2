import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

image_files = [os.path.join('celeba_dataset_minified', image_files_i)
               for image_files_i in os.listdir('celeba_dataset_minified')
               if '.jpg' in image_files_i]

images = [plt.imread(image_files_i) for image_files_i in image_files]

plt.show(plt.imshow(images[0]))