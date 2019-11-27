import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# from keras.preprocessing.image import array_to_img

images = np.load('image_data.npy')
names = np.load('image_names.npy')
img1 = Image.fromarray(images[12555])
# img1 = array_to_img(images[0])

print(names[12555])
plt.imshow(img1)
plt.show()
