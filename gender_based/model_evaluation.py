import numpy as np
import time
import keras
from keras.utils import to_categorical
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import load_model


batch_size = 128
num_classes = 2  # Males & Females
epochs = 10


# input image dimensions
img_x, img_y = 170, 197


# Split by Gender
X_males = np.load('gender_based/data_arrays_and_labels/males_image_data2.npy')
y_males = np.ones(10268)

X_females = np.load(
    'gender_based/data_arrays_and_labels/females_image_data2.npy')
y_females = np.zeros(2965)

# Joined
x = np.concatenate((X_males, X_females), axis=0)
y = np.concatenate((y_males, y_females), axis=0)

# Slipt into Train and Test datasets
# if you use random_state=some_number, then you can guarantee that your split will be always the same.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the data is greyscale, we only have a single channel - RGB colour images would have 3
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255  # Why?
x_test /= 255  # Why?
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the categorical_crossentropy loss
# y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)

# model = load_model("gender_based/models/male_female_model.h5")
# score = model.evaluate(x_test, y_test, verbose=1)
# print(score[1]*100)

# y_pred = model.predict(x_test, verbose=1)
# rounded_predictions = model.predict_classes(x_test, verbose=1)
# np.savetxt('rounded_predictions.txt', rounded_predictions, fmt="%d")

# np.savetxt('y_pred.txt', y_pred)
# print(y_pred)
# y_pred_2 = np.argmax(np.loadtxt('gender_based/y_pred.txt'), axis=0)
# y_test_2 = np.argmax(y_test, axis=0)
# print(confusion_matrix(y_test_2, y_pred_2[:, 0]))
y_pred = np.loadtxt('gender_based/rounded_predictions.txt')
print(confusion_matrix(y_test, y_pred))
