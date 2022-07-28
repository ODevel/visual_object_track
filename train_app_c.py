import cv2
import numpy as np
from PIL import Image
import os

#################################################################
# This script detects objects in two ways -
# 1. Through OpenCV trainer.yml
# 2. Through TensorFlow Learning after cropping by OpenCV cascade
########################################################

# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
import sys
import time


def build_model(shpe):
    print(shpe)
    # Initialising the CNN
    cnn = tf.keras.models.Sequential()
    
    cnn.add(Conv2D(256, (3, 3), activation="relu", input_shape=[shpe[0],shpe[1],3]))
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
    cnn.add(Dense(2, activation="softmax"))
    
    # Compiling the CNN
    cnn.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")
    return cnn 

# function to get the images and label data
def getImagesAndLabelsOpenCV(path, id):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    i = 0
    for imagePath in imagePaths:
        print(imagePath)
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        cv_img = cv2.imread(imagePath)
        img_numpy = np.array(PIL_img,'uint8')
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            if(train_mode == 'opencv'):
                faceSamples.append(img_numpy[y:y+h, x:x+w])
            else:
                faceSamples.append(cv_img[y:y+h,x:x+w, :])
            #input()
            ids.append(id)
        i += 1
    return faceSamples,ids

def getNegativeImg(path, label, shape):
    print(shape)
    files = os.listdir(path)
    i = 0
    y_train = []
    for f in files:
        img = cv2.imread(path + f)
        img = cv2.resize(img, (shape[1], shape[0]))
        if( i == 0):
            X_train = [tf.expand_dims(img,0)]
        else:
            X_train.append(tf.expand_dims(img,0))
        i+=1
        y_train.append(1)
    return X_train, y_train

def getImagesAndLabelsTF(test):
    cnn00 = tf.keras.models.load_model(test +'_00.h5')
    cnn01 = tf.keras.models.load_model(test +'_01.h5')
    cnn10 = tf.keras.models.load_model(test +'_10.h5')
    cnn11 = tf.keras.models.load_model(test +'_11.h5')

    test_p = test +'/' + test +'/'
    files = os.listdir(test_p +'/p/')
    i = 0
    y_train = []
    for f in files:
        f = test_p + '/p/' + f
        img = cv2.imread(f)
        img_orig =img
        bf = 1
        lf = 1
        if(img.shape[1] > 128 and img.shape[0] > 96):
            img = cv2.resize(img, (96,128))
            bf = (img_orig.shape[0]/96)
            lf = (img_orig.shape[1]/128)
        elif (img.shape[0] > 96):
            img = cv2.resize(img, (96, img.shape[1]))
            bf = (img_orig.shape[0]/96)
        elif(img.shape[1] > 128):
            img = cv2.resize(img, (img.shape[0], 128))
            lf = (img_orig.shape[1]/128)
        x = cnn00.predict(tf.expand_dims(img, 0))
        y = cnn01.predict(tf.expand_dims(img, 0))
        w = cnn10.predict(tf.expand_dims(img, 0))
        h = cnn11.predict(tf.expand_dims(img, 0))
    
        x *= bf
        y *= lf
        w *= bf
        h *= lf
        img = img_orig[int(x) : int((x+w)),int(y) : int((y+w))]
        if(img.shape[1] > 128 and img.shape[0] > 96):
            img = cv2.resize(img, (96,128))
        elif(img.shape[0] > 96):
            img = cv2.resize(img, (96, img.shape[1]))
        else:
            img = cv2.resize(img, (img.shape[0],128))
        if(i == 0) :
            X_train = [tf.expand_dims(img, 0)]
        else:
            X_train.append(tf.expand_dims(img,0))
        i+=1
        y_train.append(0)
    return X_train, y_train

def train(test_name):
    start_time = time.time()
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    # Enabling opencv by default
    train_mode= 'opencv'
    # Path for face image database
    path = test_name + '/' + test_name + '/p/'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(test_name +'/'+ test_name + '/cascade/' + 'cascade.xml');

    if(train_mode == 'opencv'):
        faces,ids = getImagesAndLabelsOpenCV(path,0)
        path = './n/'
        faces1,ids1 = getImagesAndLabelsOpenCV(path,1)
        # Negative images
        faces1 = faces1[:300]
        ids1 = ids1[:300]
        for f in faces1:
            faces.append(f)
        for i in ids1:
            ids.append(i)
        recognizer.train(faces, np.array(ids))
        
        # Save the model into trainer/trainer.yml
        if(not os.path.exists(test_name+'/'+test_name+'/trainer')):
            os.mkdir(test_name+'/'+test_name+'/trainer')
        
        recognizer.write(test_name + '/'+test_name+'/trainer/trainer.yml') 
    else:
        # TF - not much accuracy 
        faces, ids = getImagesAndLabelsTF(test_name)
        path = './n/'
        faces1,ids1 = getNegativeImg(path,1, (faces[0][0].shape[0], faces[0][0].shape[1]))
        # Negative images
        faces1 = faces1[:300]
        ids1 = ids1[:300]
        for f in faces1:
            faces.append(f)
        for i in ids1:
            ids.append(i)
        for (i, f) in enumerate(faces):
            #img = cv2.resize(f, (145,145))
            if(i == 0):
                faces_np = np.array(f)
            else:
                faces_np = np.concatenate((faces_np, np.array(f)))
        cnn = build_model((faces[0][0].shape[0], faces[0][0].shape[1]))
    
        ids = np.array(ids)
        ids = np.reshape(ids, (len(ids), 1))
    
        X_train, X_test, y_train, y_test = train_test_split(faces_np, ids, test_size=0.3, random_state=42)
    
        train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
        test_generator = ImageDataGenerator(rescale=1/255)
    
        train_generator = train_generator.flow(np.array(X_train), y_train, shuffle=False)
        test_generator = test_generator.flow(np.array(X_test), y_test, shuffle=False)
        for (i,img) in enumerate(faces_np):
            #im = X_train[i]
            cv2.imwrite('cam'+str(i)+'.jpg', img)
        print(faces_np.shape, ids.shape, len(train_generator), len(test_generator)) 
        history = cnn.fit(train_generator, epochs=40, validation_data=test_generator, shuffle=True, validation_steps=len(y_test))
    
        print('--Traing is done --\n')
        
        # Predict a sample image
        ##im = cv2.imread(test_name +'/' +test_name+'/test/' + os.listdir(test_name +'/' +test_name+'/test/')[0])
        ##im = cv2.resize(im, (145,145))
        ##im = tf.expand_dims(tf, 0)
        ##print(cnn.predict(im))
        ### Save the checkpoint
        cnn.save('model-'+ test_name + '.h5')
    
    end_time = time.time()
    print('total time taken: ', end_time - start_time)
