import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from libs import utils

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def get_image_files():
	files = [os.path.join('celeba_dataset_minified', files_i)
	         for files_i in os.listdir('celeba_dataset_minified')
	         if '.jpg' in files_i]
	return files

def read_image_files():
	return [plt.imread(image_i) for image_i in get_image_files()]

images = read_image_files()




