import glob
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np

filelist = glob.glob('data/*.jpg')

X = np.array([img_to_array(load_img(fname, grayscale=True))
              for fname in filelist])

np.save('keras_image_data', X)

print("DONE")
