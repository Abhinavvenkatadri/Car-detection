


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


#
# # loaded_model=load_model('model.hdf5')
# # Initialising the CNN
# classifier = Sequential()
# # Step 1 - Convolution
# classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
# # Step 2 - Pooling
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(BatchNormalization())
# classifier.add(Dropout(0.1))
# # Adding a second convolutional layer(changed to 64)
# classifier.add(Conv2D(64, (3, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(BatchNormalization())
# classifier.add(Dropout(0.15))
# # i added a convolution layer cos accuracy wasnt changing
# classifier.add(Conv2D(64, (3, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# # Step 3 - Flattening
# classifier.add(Flatten())
# classifier.add(BatchNormalization())
# classifier.add(Dropout(0.2))
# # Step 4 - Full connection
# classifier.add(Dense(units=128, activation='relu'))
# classifier.add(Dense(units=1, activation='sigmoid'))
#
# classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model = load_model('model1.hdf5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model =load_model('model_weights.h5')
# print("model loaded")
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
PATH = os.getcwd()

data_path = PATH + '/tes4'
data_dir_list = os.listdir(data_path)

count = 1
for i, label in enumerate(data_dir_list):
    cur_path = data_path + '/' + label
    #print(cur_path)
    a = sorted(glob.glob(cur_path))
    j = 1  # type: int
    print(a)

    for image_path in a:

#        img = image.load_img(image_path, target_size=(128,128,1))
        input_img = cv2.imread(image_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img, (64,64))
        #img_data = np.array(img_data_list)
        img_data = input_img_resize.astype('float32')
        img_data /= 255

        x = np.expand_dims(img_data, axis=0)
        x = np.expand_dims(x, axis=0)
        x= x.reshape((-1,64,64,1))

        y = model.predict(x)
        print(y)
        #training_set.class_indices
        if y < 0.5:
            prediction = 'Car'  # type: str
        else:
            prediction = 'Non-car'
        print(prediction)
        print ("[INFO] processed - " + str(j))
        print ("===============================")
    # noinspection PyUnboundLocalVariable
    j += 1
print ('[INFO] completed label - {0}'.format(label))
