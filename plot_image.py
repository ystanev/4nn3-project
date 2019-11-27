import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

images = np.load('image_data.npy')
img1 = Image.fromarray(images[5000])
plt.imshow(img1)
plt.show()
