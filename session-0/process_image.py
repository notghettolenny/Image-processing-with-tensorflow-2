from libs import utils
import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt 
from scipy.misc import imresize

plt.style.use('ggplot')

files = [os.path.join('img_align_celeba_small', file_i)
		 for file_i in os.listdir('img_align_celeba_small')
		 if '.jpg' in file_i]

read_files = [plt.imread(file_i)
			  for file_i in files]

data = np.array(read_files)

mean_data = np.mean(data, axis=0)
"""plt.imshow(mean_data.astype(np.uint8))
plt.show()
"""

std_data = np.std(data, axis=0)
"""plt.imshow(std_data.astype(np.uint8))
plt.show()
"""

# Generate a heat map by getting the mean of the std of all channels
# plt.imshow(np.mean(std_data, axis=2).astype(np.uint8))
# plt.show()

# Flatten the batch
flattened = data.ravel()
# print(flattened[:10])

# Show a histogram of all the rgbs from the flattened values with bin val 255
# plt.hist(values, bin or number of lines)
# plt.hist(flattened.ravel(), 255)

""" Normalization means to select an image and subtract it to the mean of all
the images contained inside the batch dimension and dividing the answer to the 
standard deviation of all the images contained inside the batch dimension. Do not
forget to flatten using the ravel() function
"""

# figsize (w, h) tuple how large the ouput figure will be
# sharey, sharex (bool) means the x and y axis be shared by all subplots
bins = 20
fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True, sharex=True)
"""axs[0].hist((data[0]).ravel(), bins)
axs[0].set_title('img distribution')
axs[1].hist((mean_data).ravel(), bins)
axs[1].set_title('mean distribution')
"""
axs[0].hist((data[0] - mean_data).ravel(), bins)
axs[0].set_title('(img - mean) distribution')
axs[1].hist((std_data).ravel(), bins)
axs[1].set_title('std distribution')
axs[2].hist(((data[0] - mean_data) / std_data).ravel(), bins)
axs[2].set_title('((img - mean) / std) distribution')
# Uncomment to disregard set_xlim()
# plt.show(fig.show())

# Look at a different scale set_xlim() means to zoom on x range a to b (a, b)
axs[2].set_xlim([-150, 150])
axs[2].set_xlim([-100, 100])
axs[2].set_xlim([-50, 50])
axs[2].set_xlim([-10, 10])
axs[2].set_xlim([-5, 5])
plt.show(fig.show())