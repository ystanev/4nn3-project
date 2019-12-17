import glob
import os
import numpy as np
from PIL import Image, ImageOps
import time
import keras
from keras.utils import to_categorical
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# from keras.models import load_model
from shutil import copyfile, move


# * WORKS
def move_images_to_one_folder(scr, dst):
    """
    Moves all images in dataset into one folder to simplify data processing.

    Args:
        scr: location of the dataset
        dst: the folder to extract the images into
    """
    if not os.path.exists(dst):
        os.makedirs(dst)

    for root, subdirs, files in os.walk(scr):
        for file in files:
            print(file)
            path = os.path.join(root, file)
            move(path, dst)


# * WORKS
def split_by_gender(image_data, save_path_male_images, save_path_female_images, male_names_path, female_names_path):
    """
    Compares the file name against gender labels provided with the dataset and splits images into gender folders.

    Args:
          image_data: the location of the folder with all the images ( `dst` from move_images_to_one_folder() )
          save_path_female_images: location to save images of males
          save_path_male_images: location to save female images
          male_names_path: location of file with containing `file name` of male images
          female_names_path: location of file with containing `file name` of female images
    """
    if not os.path.exists(save_path_male_images):
        os.makedirs(save_path_male_images)
    elif not os.path.exists(save_path_female_images):
        os.makedirs(save_path_female_images)

    filelist = glob.glob(image_data + '/*.jpg')
    male_names = np.loadtxt(male_names_path, dtype='str', delimiter='\n')
    female_names = np.loadtxt(female_names_path, dtype='str', delimiter='\n')

    for file in filelist:
        for male in male_names:
            if file.split("/")[-1] == male:
                print(male)
                copyfile(file, save_path_male_images + male)
        for female in female_names:
            if file.split("/")[-1] == female:
                print(female)
                copyfile(file, save_path_female_images + female)


# * WORKS
def save_images_to_array(save_path_male_images, save_path_female_images, save_male_array, save_female_array):
    """
    The function deals with data preparation. It performs the following operation on the images:
        1. crop the image apprx. to face area
        2. convert image image to grayscale ( remove RGB channels )
        3. convert image data into a numpy array

    Args:
        save_path_male_images: the location containing images of males
        save_path_female_images: the location containing images of females
        save_male_array: location to save `.npy` file with male image data
        save_female_array: location to save `.npy` file with female image data
    """
    if not os.path.exists(save_male_array):
        os.makedirs(save_male_array)
    elif not os.path.exists(save_female_array):
        os.makedirs(save_female_array)

    filelist_males = glob.glob(save_path_male_images + '/*.jpg')
    filelist_females = glob.glob(save_path_female_images + '/*.jpg')

    # Images
    X_males = np.array([np.array(ImageOps.grayscale(Image.open(fname).crop((40, 8, 210, 205))))
                        for fname in filelist_males])  # array of images
    np.save(os.path.join(save_male_array, 'males.npy'), X_males)

    X_females = np.array([np.array(ImageOps.grayscale(Image.open(fname).crop((40, 8, 210, 205))))
                          for fname in filelist_females])  # array of images

    np.save(os.path.join(save_female_array, 'females.npy'), X_females)


# * WORKS
def cnn(saved_male_array, saved_female_array, save_model_path, save_predictions_path):
    """
    Builds the convolution neural network model for image classification.

    Args:
        saved_male_array: location of the `.npy` file with male image data
        saved_female_array: location of the `.npy` file with female image data
        save_model_path: path to save `.h5` file ( cnn model ) for later usage
        save_predictions_path: path to save the file containing model class predictions

    Outputs:
        1. confusion matrix
        2. time elapsed for model training and testing
    """
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    elif not os.path.exists(save_predictions_path):
        os.makedirs(save_predictions_path)

    batch_size = 128
    num_classes = 2  # Males & Females
    epochs = 10

    # input image dimensions
    img_x, img_y = 170, 197

    # Split by Gender
    X_males = np.load(saved_male_array)
    y_males = np.ones(10268)

    X_females = np.load(saved_female_array)
    y_females = np.zeros(2965)

    # Joined
    x = np.concatenate((X_males, X_females), axis=0)
    y = np.concatenate((y_males, y_females), axis=0)

    # Slipt into Train and Test datasets
    # if you use random_state=some_number, then you can guarantee that your split will be always the same.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42)

    # Duplicated Test Labels for Confusion Matrix
    conf_y_test = y_test

    # reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
    # because the data is greyscale, we only have a single channel - RGB colour images would have 3
    x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
    x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
    input_shape = (img_x, img_y, 1)

    # convert the data to the right type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices - this is for use in the categorical_crossentropy loss
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # The core core for CNN
    model = Sequential()

    # Convolutional Layers

    # 1
    model.add(Conv2D(32, kernel_size=(7, 7), strides=(1, 1),
                     activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # 2
    model.add(Conv2D(32, (7, 7), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 3
    model.add(Conv2D(32, (7, 7), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Configures the model for training.
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    # Timing The Code
    start_time = time.time()

    # Train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(x_test, y_test))

    # Test the model
    score = model.evaluate(x_test, y_test, verbose=1)

    elapsed_time = time.time() - start_time

    # save as .h5 file
    model.save(os.path.join(save_model_path, 'cnn_model.h5'))
    print("Model Saved to Disk")
    print("Elapsed Time: ", elapsed_time)

    # Make predictions ans save them to file
    # model = load_model(save_model_path) # * may not be needed
    rounded_predictions = model.predict_classes(x_test, verbose=1)
    np.savetxt(os.path.join(save_predictions_path, 'classes.txt'),
               rounded_predictions, fmt="%d")

    # Confusion Matrix
    y_pred = np.loadtxt(os.path.join(save_predictions_path, 'classes.txt'))
    print(confusion_matrix(conf_y_test, y_pred))


def main():
    """
    A wrapper around function calls backs.
    """
    # Parameters Data Type: String -> PATH
    scr = r'/home/yury.stanev/Downloads/lfw-deepfunneled/'
    dst = r'/home/yury.stanev/4nn3-project/clean_cnn_outputs/data/'

    image_data = '/home/yury.stanev/4nn3-project/clean_cnn_outputs/data/'
    save_path_male_images = '/home/yury.stanev/4nn3-project/clean_cnn_outputs/data_by_gender/males/'
    save_path_female_images = '/home/yury.stanev/4nn3-project/clean_cnn_outputs/data_by_gender/females/'

    male_names_path = '/home/yury.stanev/4nn3-project/cleanup/lwf_gender_labeled_data/male_names.txt'
    female_names_path = '/home/yury.stanev/4nn3-project/cleanup/lwf_gender_labeled_data/female_names.txt'

    save_male_array = '/home/yury.stanev/4nn3-project/clean_cnn_outputs/data_arrays/'
    save_female_array = '/home/yury.stanev/4nn3-project/clean_cnn_outputs/data_arrays/'
    saved_male_array = save_male_array + 'males.npy'  # ! FILE
    saved_female_array = save_female_array + 'females.npy'  # ! FILE

    save_model_path = '/home/yury.stanev/4nn3-project/clean_cnn_outputs/models/'
    save_predictions_path = '/home/yury.stanev/4nn3-project/clean_cnn_outputs/outputs/'

    move_images_to_one_folder(scr, dst)

    split_by_gender(image_data, save_path_male_images,
                    save_path_female_images, male_names_path, female_names_path)

    save_images_to_array(save_path_male_images, save_path_female_images,
                         save_male_array, save_female_array)

    cnn(saved_male_array, saved_female_array,
        save_model_path, save_predictions_path)


main()
