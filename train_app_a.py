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
    cnn.add(Dense(3, activation="softmax"))
    
    # Compiling the CNN
    cnn.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam") #optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return cnn


# Build class A
l = 128
b = 96
from sklearn.decomposition import PCA
def orient_img(img,shape):
    img1 = np.array(img[0])
    for i in range(1, shape[0]):
        img1 = np.concatenate((img1, img[i]), axis=0)

    return np.array([img1])

def reorient_img(img,shape):
    img1 = np.array([img[0*shape[1] : (0+1)*shape[1]]])
    for i in range(1, shape[0]):
        img_t = img[i*shape[1] : (i+1)*shape[1]]
        img1 = np.concatenate((img1, np.array([img_t])))

    return img1

def return_files(class_x, next_class=False):
    global l
    global b
    path = class_x + '/' + class_x + '/p/'
    files = os.listdir(path)
    for (i, f) in enumerate(files):
        files[i] = path + f
        img = cv2.imread(files[i])
        
        if(i == 0):
            shpe = img.shape
            files_ar = orient_img(img,shpe)
        else:
            files_ar = np.concatenate((files_ar, orient_img(img,shpe)))
    
    # PCA
    pca_model = PCA(n_components = 12288, random_state=4)
    print(files_ar.shape)
    pca_model.fit(files_ar[:,:,0])
    pca_out = pca_model.transform(files_ar[:,:,0])
    #print ("Variance explained by all 30 principal components ", sum(pca_out.explained_variance_ratio * 100))
    img_t = reorient_img(files_ar[0],shpe)
    quit()
    return files_ar

class_a = 'blurbody'
class_b = 'bird1'
class_c = 'panda'
#class_a = sys.argv[-1]
#class_b = sys.argv[-2]
#class_c = sys.argv[-3]

files_a = return_files(class_a)
files_b = return_files(class_b, next_class=True)
files_c = return_files(class_c, next_class=True)

labels = np.array([[0,0,1]])
for i in range(0,len(files_a)-1):
    t = np.array([[0,0,1]])
    labels = np.concatenate((labels, t))

for i in range(0,len(files_b)):
    t = np.array([[0,1,0]])
    labels = np.concatenate((labels, t))

for i in range(0,len(files_c)):
    t = np.array([[1,0,0]])
    labels = np.concatenate((labels, t))

# Concatenate X & y
files = np.concatenate((files_a, files_b))
files = np.concatenate((files, files_c))

X_train, X_test, y_train, y_test = train_test_split(files, labels, test_size=0.3, random_state=4)
train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
test_generator = ImageDataGenerator(rescale=1/255)
train_generator = train_generator.flow(np.array(X_train), y_train, shuffle=False)
test_generator = test_generator.flow(np.array(X_test), y_test, shuffle=False)
cnn = model_builder(True, (l, b))
history = cnn.fit(X_train, y_train,  epochs=100, verbose=1, validation_split=0.2)
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))

import matplotlib.pyplot as plt
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

cnn.save('simple_detection.h5')
