import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

males = np.load('gender_based/data_arrays_and_labels/males_image_data2.npy')
females = np.load('gender_based/data_arrays_and_labels/females_image_data2.npy')

plt.imshow(Image.fromarray(males[2]))
plt.show()
plt.imshow(Image.fromarray(females[2]))
plt.show()

