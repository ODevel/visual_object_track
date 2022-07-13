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

if(len(sys.argv) != 3):
    print('''
    Usage:
    python3 <script> <mode> <test_name>
    where,
    mode: opencv | tf
    ''')
    quit()
    
test_name = sys.argv[-1]
train_mode = sys.argv[1]

def build_model():
    # Initialising the CNN
    cnn = tf.keras.models.Sequential()
    
    cnn.add(Conv2D(256, (3, 3), activation="relu", input_shape=[145,145,3]))
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
    cnn.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam") #optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return cnn 


# Path for face image database
#test_name = 'blurbody'
path = test_name + '/' + test_name + '/p/'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(test_name +'/'+ test_name + '/cascade/' + 'cascade.xml');

#train_mode = 'opencv'
# function to get the images and label data
def getImagesAndLabels(path, id):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    i = 0
    for imagePath in imagePaths:
        print(imagePath)
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        cv_img = cv2.imread(imagePath)
        img_numpy = np.array(PIL_img,'uint8')
        #id = int(os.path.split(imagePath)[-1].split("_")[0])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            if(train_mode == 'opencv'):
                faceSamples.append(img_numpy[y:y+h, x:x+w])
            else:
                faceSamples.append(cv_img[y:y+h,x:x+w, :])
            #cv2.imwrite('cam'+str(i) + '.jpg',cv_img[y:y+h,x:x+w, :])
            #input()
            ids.append(id)
        i += 1
    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path,0)
path = './n/'
faces1,ids1 = getImagesAndLabels(path,1)
# Negative images
faces1 = faces1[:300]
ids1 = ids1[:300]
for f in faces1:
    faces.append(f)
for i in ids1:
    ids.append(i)

if(train_mode == 'opencv'):
    recognizer.train(faces, np.array(ids))
    
    # Save the model into trainer/trainer.yml
    if(not os.path.exists(test_name+'/'+test_name+'/trainer')):
        os.mkdir(test_name+'/'+test_name+'/trainer')
    
    recognizer.write(test_name + '/'+test_name+'/trainer/trainer.yml') 
else:
    cnn = build_model()
    for (i, f) in enumerate(faces):
        img = cv2.resize(f, (145,145))
        if(i == 0):
            faces_np = np.array([img])
        else:
            faces_np = np.concatenate((faces_np, np.array([img])))

    ids = np.array(ids)
    ids = np.reshape(ids, (len(ids), 1))

    X_train, X_test, y_train, y_test = train_test_split(faces_np, ids, test_size=0.3, random_state=42)

    train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
    test_generator = ImageDataGenerator(rescale=1/255)

    train_generator = train_generator.flow(np.array(X_train), y_train, shuffle=False)
    test_generator = test_generator.flow(np.array(X_test), y_test, shuffle=False)
    #train_set = np.concatenate((np.reshape(X_train, (len(X_train),1)), np.reshape(y_train, (len(y_train),1))), axis=1)
    #test_set = np.concatenate((np.reshape(X_test, (len(X_test),1)), np.reshape(y_test, (len(y_test),1))), axis=1)
    print(faces_np.shape, ids.shape, len(train_generator), len(test_generator)) 
    history = cnn.fit(train_generator, epochs=40, validation_data=test_generator, shuffle=True, validation_steps=len(y_test))

    print('--Traing is done --\n')
    
    # Predict a sample image
    im = cv2.imread(test_name +'/' +test_name+'/test/' + os.listdir(test_name +'/' +test_name+'/test/')[0])
    im = cv2.resize(im, (145,145))
    im = tf.expand_dims(tf, 0)
    print(cnn.predict(im))
    # Save the checkpoint
    cnn.save('model-'+ test_name + '.h5')
