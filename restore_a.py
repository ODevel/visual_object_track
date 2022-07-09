# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
#from tensorflow.keras.models import Model
#from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
#import tensorflow as tf
import cv2

#cnn = tf.keras.models.load_model('model-a.h5')

# Preprocessing the Test set
#test_datagen = ImageDataGenerator(rescale = 1./255)
#test_set = test_datagen.flow_from_directory('archive/train',
#                                            target_size = (145, 145),
#                                            batch_size = 32,
#                                            class_mode = 'categorical')

epoch = 0
img_or_lab = 0
#loss, acc = cnn.evaluate(test_set[epoch][img_or_lab], test_set[0][1], verbose=2)


import numpy as np
import os
import time

recognizer = cv2.face.LBPHFaceRecognizer_create()
cascadePath = "boker_cascade.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['Eyes Closed', 'Eyes Open, Okay', 'No-Yawn, Okay', 'Yawn'] 
# consfusion between open, no-yawn

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = cam.get(3)
minH = cam.get(4)

t1 = time.time()
id_list = []

# tmp
img_lis = os.listdir('Bird1_unsucc/Bird1/p/')
for (i, im) in enumerate(img_lis):
#while True:
#if True:
    #ret, img =cam.read()
    img = cv2.imread('Bird1_unsucc/Bird1/p/' + im)
    img1 = cv2.resize(img, (145,145))
    #img = cv2.flip(img, -1) # Flip vertically
    #img = cv2.imread('dataset/1_5.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        img,
        #scaleFactor = 1.2,
        #minNeighbors = 5,
        #minSize = (145, 145),
       )

    
    shp = (1,145,145,3)
    arr_t = np.ones(shp)
    for i in range(0,1):
        for j in range(0, 145):
            for k in range(0,145):
                for l in range(0,3):
                    arr_t[i,j,k,l] = img1[j,k,l]

    #pred_ar = cnn.predict(arr_t)
    #high = -255
    #prediction = -1
    #for i in range(0,4):
    #    if(pred_ar[0,i] > high):
    #        high = pred_ar[0,i]
    #        prediction = i
    if(len(faces) > 0):
        print(im)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        
        #id_list.append(names[prediction])

        cv2.putText(img, 'Biker', (x+5,y-5), font, 1, (255,255,255), 2)
    
    cv2.imshow('camera',img) 
    t2 = time.time()
    #if(t2 - t1 >= 15):
    #    break
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    #if k == 27:
    #    break


#cam.release()
#cv2.destroyAllWindows()
