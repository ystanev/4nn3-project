# loads images in `./data/*.jpg` into an array as grayscale and save it to the disk -> file://image_data.npy
#        axis: [  0,    1,   2 ] -> https://www.w3resource.com/w3r_images/python-data-type-list-excercise-13.svg
# array shape: [13233, 255, 255]

import numpy as np
# from PIL import Image, ImageOps
import glob
from shutil import copyfile
filelist = glob.glob('data/*.jpg')

# Images
# X = np.array([np.array(ImageOps.grayscale(Image.open(fname)))
#               for fname in filelist])  # array of images
# np.save('image_data', X)

# File Names
# y = np.array([fname.split("/")[1] for fname in filelist])
# np.save('image_names2', y)

male_names = np.loadtxt('male_names.txt', dtype='str', delimiter='\n')
female_names = np.loadtxt('female_names.txt', dtype='str', delimiter='\n')

for fname in filelist:
    for male in male_names:
        for female in female_names:
            if fname.split("/")[1] == male:
                copyfile(
                    fname, '/home/yury.stanev/Downloads/lwf_labels_data/male/'+fname.split("/")[1])
                print(fname.split("/")[1])
            elif fname.split("/")[1] == female:
                copyfile(
                    fname, '/home/yury.stanev/Downloads/lwf_labels_data/female/'+fname.split("/")[1])
                print(fname.split("/")[1])


print("DONE")
