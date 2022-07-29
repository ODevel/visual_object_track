# Importing the libraries
import cv2
import numpy as np
from PIL import Image
import os

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys
import time
import matplotlib.pyplot as plt

def model_builder(shpe):
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
    cnn.add(Dense(1, activation="relu"))
    
    # Compiling the CNN
    cnn.compile(loss="mean_squared_error", metrics=["mean_squared_error"], optimizer="adam")
    return cnn
    

def train(test):
    print('Test: ' , test)
    start_time = time.time()
    
    
    statt_time = time.time()
    fp = open(test + '/' + test + '/pos.txt')
    line = fp.readline()
    line_arr = []
    while line:
        line_arr.append(line.split(' '))
        line = fp.readline()
    
    # Prepare data
    i = 0
    for lin in line_arr:
        path = test + '/' + test+'/p/' +lin[0][2:]
        if(not os.path.exists(path)):
            #print(path, ': image not found')
            continue
        #print('Image found')
        img = cv2.imread(path)
        if(i == 0):
            l = img.shape[0]
            b = img.shape[1]
            lf = 1
            bf = 1
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
    
            print('l:', l, 'b:', b, img.shape)
        img = cv2.resize(img, (b,l))
        print(img.shape)
        x_0 = int(int(lin[2])*bf)
        x_1 = int(int(lin[3])*lf)
        y_0 = int(int(lin[4])*bf)
        y_1 = int(int(lin[5])*lf)
    
        #cv2.rectangle(img,(x_0,x_1), (y_0+x_0,y_1+x_1), (0,255,0), 2)
        ##cv2.rectangle(img,(x_0,x_0), (x_1,y_1), (0,255,0), 2)
        #cv2.imwrite('cam' + str(i) + '.jpg', img)
        ##input()
    
        point00 = np.array([[x_0]])#, x_1, x_0 + y_0, x_1 + y_1]])
        point01 = np.array([[x_1]])
        point10 = np.array([[x_0+y_0]])
        point11 = np.array([[x_1+y_1]])
        if(i == 0):
            X = np.array([img])
            y00 = point00
            y01 = point01
            y10 = point10
            y11 = point11
        else:
            X = np.concatenate((X , np.array([img])))
            y00 = np.concatenate((y00, point00))
            y01 = np.concatenate((y01, point01))
            y10 = np.concatenate((y10, point10))
            y11 = np.concatenate((y11, point11))
        i += 1
    
    X_train00, X_test00, y_train00, y_test00 = train_test_split(X, y00, test_size=0.3, random_state=42)
    train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
    test_generator = ImageDataGenerator(rescale=1/255)
    train_generator = train_generator.flow(np.array(X_train00), y_train00, shuffle=False)
    test_generator = test_generator.flow(np.array(X_test00), y_test00, shuffle=False)
    cnn00 = model_builder((l, b))
    history00 = cnn00.fit(X_train00, y_train00,  epochs=200, verbose=1, validation_split=0.2)
    cnn00.save('checkpoints/'+test +'_00.h5')
    
    history = history00
    accuracy = history.history['mean_squared_error']
    val_accuracy = history.history['val_mean_squared_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(accuracy))
    
    plt.ion()
    plt.figure()
    plt.plot(epochs, accuracy, "b", label="trainning accuracy")
    plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
    plt.legend()
    plt.show()
    plt.figure()
    
    plt.plot(epochs, loss, "b", label="trainning loss")
    plt.plot(epochs, val_loss, "r", label="validation loss")
    plt.legend()
    plt.show()
    
    
    X_train01, X_test01, y_train01, y_test01 = train_test_split(X, y01, test_size=0.3, random_state=42)
    train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
    test_generator = ImageDataGenerator(rescale=1/255)
    train_generator = train_generator.flow(np.array(X_train01), y_train01, shuffle=False)
    test_generator = test_generator.flow(np.array(X_test01), y_test01, shuffle=False)
    cnn01 = model_builder((l, b))
    history01 = cnn01.fit(X_train01, y_train01,  epochs=200, verbose=1, validation_split=0.2)
    cnn01.save('checkpoints/'+test +'_01.h5')
    
    history = history01
    accuracy = history.history['mean_squared_error']
    val_accuracy = history.history['val_mean_squared_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(accuracy))
    
    plt.ion()
    plt.figure()
    plt.plot(epochs, accuracy, "b", label="trainning accuracy")
    plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(epochs, loss, "b", label="trainning loss")
    plt.plot(epochs, val_loss, "r", label="validation loss")
    plt.legend()
    plt.show()
    
    
    X_train10, X_test10, y_train10, y_test10 = train_test_split(X, y10, test_size=0.3, random_state=42)
    train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
    test_generator = ImageDataGenerator(rescale=1/255)
    train_generator = train_generator.flow(np.array(X_train10), y_train10, shuffle=False)
    test_generator = test_generator.flow(np.array(X_test10), y_test10, shuffle=False)
    cnn10 = model_builder((l, b))
    history10 = cnn10.fit(X_train10, y_train10,  epochs=200, verbose=1, validation_split=0.2)
    cnn10.save('checkpoints/'+test +'_10.h5')
    
    history = history10
    accuracy = history.history['mean_squared_error']
    val_accuracy = history.history['val_mean_squared_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(accuracy))
    
    plt.ion()
    plt.figure()
    plt.plot(epochs, accuracy, "b", label="trainning accuracy")
    plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(epochs, loss, "b", label="trainning loss")
    plt.plot(epochs, val_loss, "r", label="validation loss")
    plt.legend()
    plt.show()
    
    
    X_train11, X_test11, y_train11, y_test11 = train_test_split(X, y11, test_size=0.3, random_state=42)
    train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
    test_generator = ImageDataGenerator(rescale=1/255)
    train_generator = train_generator.flow(X_train11, y_train11, shuffle=False)
    test_generator = test_generator.flow(X_test11, y_test11, shuffle=False)
    cnn11 = model_builder((l, b))
    history11 = cnn11.fit(X_train11, y_train11,  epochs=200, verbose=1, validation_split=0.2)
    cnn11.save('checkpoints/' +test +'_11.h5')
    
    history = history11
    accuracy = history.history['mean_squared_error']
    val_accuracy = history.history['val_mean_squared_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(accuracy))
    
    plt.ion()
    plt.plot(epochs, accuracy, "b", label="trainning accuracy")
    plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
    plt.legend()
    plt.show()
    plt.figure()
    
    plt.plot(epochs, loss, "b", label="trainning loss")
    plt.plot(epochs, val_loss, "r", label="validation loss")
    plt.legend()
    plt.show()
    
    
    
    y_pred00 = cnn00.predict(X_train00)
    y_pred01 = cnn01.predict(X_train00)
    y_pred10 = cnn10.predict(X_train00)
    y_pred11 = cnn11.predict(X_train00)
    
    boxes = np.concatenate((y_pred00, y_pred01), axis=1)
    boxes = np.concatenate((boxes, y_pred10), axis=1)
    boxes = np.concatenate((boxes, y_pred11), axis=1)
    
    for (i,img) in enumerate(X_train00):
        x,y,w,h = boxes[i]
        cv2.rectangle(img, (int(x),int(y)),(int((x+w)),int((y+h))), (0,255,0), 2)
        cv2.imwrite('cam' +str(i) +'.jpg', img)
        #input()
    
    print('--Traing is done --\n')
       
    end_time = time.time()
    print('total time taken:' , end_time - start_time)
