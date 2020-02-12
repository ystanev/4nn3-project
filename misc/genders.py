import glob
# from shutil import copyfile
filelist = glob.glob('data/*.jpg')
import numpy as np
# from shutil import copyfile

male_names = np.loadtxt('male_names.txt', dtype='str', delimiter='\n')
female_names = np.loadtxt('female_names.txt', dtype='str', delimiter='\n')

for file in filelist:
    # print(file)
    for male in male_names:
        # for female in female_names:
        if file.split("/")[1] != male:
            print(male)

print('DONE')     