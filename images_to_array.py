# loads images in `./data/*.jpg` into an array as grayscale and save it to the disk -> file://image_data.npy
#        axis: [  0,    1,   2 ] -> https://www.w3resource.com/w3r_images/python-data-type-list-excercise-13.svg
# array shape: [13233, 255, 255]

import numpy as np
from PIL import Image, ImageOps
import glob
filelist = glob.glob('data/*.jpg')

X = np.array([np.array(ImageOps.grayscale(Image.open(fname)))
              for fname in filelist])  # array of images

np.save('image_data', X)

print("DONE")
