import numpy as np
import time
import keras
from keras.utils import to_categorical
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


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

#Timing The Code
start_time = time.time()  

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))

# Test the model
score = model.evaluate(x_test, y_test, verbose=1)

elapsed_time = time.time() - start_time

model.save('gender_based/models/male_female_model.h5')
print("Model Saved to Disk")

# Confusion Matrix
print("Elapsed Time: ", elapsed_time)
# y_pred = np.argmax(score, axis=0)
# print("Confusion Matrix:\n", confusion_matrix(y, y_pred))


print("DONE")