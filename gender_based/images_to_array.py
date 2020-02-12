# loads images in `./data/*.jpg` into an array as grayscale and save it to the disk -> file://image_data.npy
#        axis: [  0,    1,   2 ] -> https://www.w3resource.com/w3r_images/python-data-type-list-excercise-13.svg
# array shape: [13233, 255, 255]

import numpy as np
from PIL import Image, ImageOps
import glob
# from shutil import copyfile
filelist_males = glob.glob(
    '/home/yury.stanev/4nn3-project/gender_based/lwf_gender_labeled_data/male/*.jpg')
filelist_females = glob.glob(
    '/home/yury.stanev/4nn3-project/gender_based/lwf_gender_labeled_data/female/*.jpg')


# Images
X_males = np.array([np.array(ImageOps.grayscale(Image.open(fname).crop((40, 8, 210, 205))))
                    for fname in filelist_males])  # array of images
np.save('males_image_data2', X_males)

X_females = np.array([np.array(ImageOps.grayscale(Image.open(fname).crop((40, 8, 210, 205))))
                      for fname in filelist_females])  # array of images
np.save('females_image_data2', X_females)

print("DONE")
