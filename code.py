# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 13:47:36 2020

@author: Acer
"""


from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation, BatchNormalization, Dropout, Flatten, Dense
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os, cv2
import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
# from keras.models import load_model
import numpy as np
from keras.models import model_from_json
from keras.models import load_model
from numpy.core.multiarray import ndarray

epochs = 2
img_data_list = []
labels_list = []
PATH = os.getcwd()
data_path = PATH + '/car'
data_dir_list = os.listdir(data_path)
for i, label in enumerate(data_dir_list):
    cur_path = data_path + '/' + label
    #print("current path")
    #print(cur_path)
    a = sorted(glob.glob(cur_path))
    j = 1  # type: int
    #print("a")
    #print(a)

    for img in a:
        #print(img)
        try:
            input_img = cv2.imread(img,0)
            input_img_resize = cv2.resize(input_img, (64, 64))
        except Exception as e:
            print(str(e))
        #print("inp-img")
        #print(input_img)
        #print(input_img.shape)
        #input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        #input_img_resize = cv2.resize(input_img, (64, 64))

        #print(input_img_resize.shape)
        #
        if img[10] == 'C':
            labels_list.append('0')
        else:
            labels_list.append('1')
        img_data_list.append(input_img_resize)
        # labels_list.append(label)
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print(img_data.shape)
# print(labels_list)
labels = np.array(labels_list)
# print the count of number of samples for different classes
print(np.unique(labels, return_counts=True))
# convert class labels to on-hot encoding
# Y = np_utils.to_categorical(labels, num_classes)
# Shuffle the dataset
x, y = shuffle(img_data, labels, random_state=4)
x = np.expand_dims(x, axis=1).reshape(-1, 64, 64, 1)
# Split the dataset
training_set, test_set, train_label, test_label = train_test_split(x, y, test_size=0.2, random_state=2)

# loaded_model=load_model('model.hdf5')
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.1))
# Adding a second convolutional layer(changed to 64)
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.15))
# i added a convolution layer cos accuracy wasnt changing
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
classifier.add(BatchNormalization())
classifier.add(Dropout(0.2))
# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(training_set, train_label, batch_size=256, epochs=14, validation_data=(test_set, test_label))

#classifier.save('classifier.h5')
#classifier.save_weights("classifier.h5")
#print("Saved model to disk")
#
# import json
#
# # lets assume `model` is main model
# model_json = classifier.to_json()
# with open("model_in_json.json", "w") as json_file:
#     json.dump(model_json, json_file)
#
# classifier.save_weights("model_weights.h5")
# print("saved model")
#
# from keras.models import load_model
# from keras.models import model_from_json
# import json
#
# with open('model_in_json.json','r') as f:
#     model_json = json.load(f)
#
# model = model_from_json(model_json)
# model.load_weights('model_weights.h5')
#
# print("loaded model")

# serialize model to JSON
model_json = classifier.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model1.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
print("Loaded model from disk")

classifier.save('model1.hdf5')
loaded_model=load_model('model1.hdf5')
#training_set.save('train3.h5')
print("done")