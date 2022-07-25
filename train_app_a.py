import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import sys
#import keras_tuner as kt
import os
import time

def model_builder(hp, shpe):
    # Initialising the CNN
    cnn = tf.keras.models.Sequential()
    #print('MODEL_BUILDER: shape:: ' , shpe)
    
    cnn.add(Conv2D(256, (3, 3), activation="relu", input_shape=[shpe[0], shpe[1], 3]))
    cnn.add(MaxPooling2D(2, 2))
    cnn.add(Conv2D(128, (3, 3), activation="relu"))
    cnn.add(MaxPooling2D(2, 2))
    cnn.add(Conv2D(128, (3, 3), activation="relu"))
    cnn.add(MaxPooling2D(2, 2))
    cnn.add(Conv2D(128, (3, 3), activation="relu"))
    cnn.add(MaxPooling2D(2, 2))
    cnn.add(Flatten())
    cnn.add(Dropout(0.5))
    cnn.add(Dense(64, activation="relu"))
    cnn.add(Dense(13, activation="softmax"))
    
    # Compiling the CNN
    cnn.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam") #optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return cnn


# Build class A
l = 128
b = 96
def return_files(class_x, next_class=False):
    global l
    global b
    if(os.path.exists(class_x + '/' + class_x + '/pca.img/')):
        path = class_x + '/' + class_x + '/pca.img/'
    else:
        path = class_x + '/' + class_x + '/p/'
    files = os.listdir(path)
    for (i, f) in enumerate(files):
        files[i] = path + f
        img = cv2.imread(files[i])
        if(i == 0 and next_class == False):
            l = img.shape[0]
            b = img.shape[1]
            lf = 1
            bf = 1
            # TF Opt.
            if(img.shape[0] > 128 and img.shape[1] > 96):
                l = 128
                b = 96
                lf = 128/img.shape[0]
                bf = 96/img.shape[1]
            elif(imf.shape[0] > 128):
                l = 128
                lf = 128/img.shape[0]
            elif(img.shape[1] > 96):
                b = 96
                bf = 96/img.shape[1]

        img = cv2.resize(img, (b,l))
        if(i == 0):
            files_ar = tf.expand_dims(img,0)
        else:
            files_ar = np.concatenate((files_ar, tf.expand_dims(img,0)))
    
    return files_ar

clsses = ['','','']
def train():
    global clsses
    clsses = ['biker', 'bird1', 'blurbody', 'blurcar2', 'bolt', 'cardark', 'football', 'human3', 'human6', 'human9', 'panda', 'walking', 'walking2']
    
    for (i,cls) in enumerate(clsses):
        if(i==0):
            files = np.array(return_files(cls))
            size = len(files)
            labels_0 = np.zeros((size,1))
            labels_1 = np.ones((size,1))
            for j in range(0, len(clsses)):
                if(j == 0):
                    labels = labels_1
                else:
                    labels = np.concatenate((labels, labels_0), axis=1)
        else:
            tmp = return_files(cls, next_class=True)
            size = len(tmp)
            labels_0 = np.zeros((size,1))
            labels_1 = np.ones((size,1))
            for j in range(0, len(clsses)):
                if(j == 0):
                    labels_tmp = labels_0
                elif(j == i):
                    labels_tmp = np.concatenate((labels_tmp, labels_1), axis=1)
                else:
                    labels_tmp = np.concatenate((labels_tmp, labels_0), axis=1)

            files = np.concatenate((files, tmp))
            labels = np.concatenate((labels, labels_tmp))
    
    

    X_train, X_test, y_train, y_test = train_test_split(files, labels, test_size=0.3, random_state=4)
    train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
    test_generator = ImageDataGenerator(rescale=1/255)
    train_generator = train_generator.flow(np.array(X_train), y_train, shuffle=False)
    test_generator = test_generator.flow(np.array(X_test), y_test, shuffle=False)
    cnn = model_builder(True, (l, b))
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    history = cnn.fit(X_train, y_train,  epochs=100, verbose=1, validation_split=0.2, callbacks=[stop_early])
    cnn.save('simple_detection.h5')
