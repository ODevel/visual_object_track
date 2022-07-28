# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
import cv2
import sys

epoch = 0
img_or_lab = 0
import sys
import numpy as np
import os
import time

def run(test_name):
    cascadePath = test_name + '/'+ test_name +'/cascade/' + 'cascade.xml'
    faceCascade = cv2.CascadeClassifier(cascadePath);
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(test_name + '/' + test_name + '/trainer/trainer.yml')
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    t1 = time.time()
    id_list = []
    img_lis = sorted(os.listdir(test_name + '/' + test_name + '/p/'))
    
    for (i, im) in enumerate(img_lis):
        img = cv2.imread(test_name + '/' + test_name +'/p/'+im)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale( 
            img,
           )
    
        idx = 0
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    
            # Prediction using OpenCV - Part A
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
    
            cv2.putText(img, test_name +str(idx+1), (x+5,y-5), font, 1, (255,255,255), 2)
            #cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0),1)
            idx += 1
        
        cv2.imshow('camera',img) 
        t2 = time.time()
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        #if k == 27:
        #    break
    
    
    cv2.destroyAllWindows()
