import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# from keras.preprocessing.image import array_to_img

images = np.load('image_data.npy')
names = np.load('image_names2.npy')
male_names = np.loadtxt('male_names.txt', dtype='str', delimiter='\n')
female_names = np.loadtxt('female_names.txt',dtype='str', delimiter='\n')


# img1 = Image.fromarray(images[12555])
# img1 = array_to_img(images[0])

print(names[12555])
# plt.imshow(img1)
# plt.show()
