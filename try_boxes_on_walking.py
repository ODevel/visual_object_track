import tensorflow as tf
import cv2
import numpy as np
import os
import sys

test = sys.argv[-1]
cnn00 = tf.keras.models.load_model(test +'_00.h5')
cnn01 = tf.keras.models.load_model(test +'_01.h5')
cnn10 = tf.keras.models.load_model(test +'_10.h5')
cnn11 = tf.keras.models.load_model(test +'_11.h5')

test_p = test +'/' + test +'/'
files = os.listdir(test_p +'/p/')
i = 0
for f in files:
    f = test_p + '/p/' + f
    img = cv2.imread(f)
    if(img.shape[0] > 128 and img.shape[1] > 96):
        img = cv2.resize(img, (128,96))
    elif (img.shape[0] > 128):
        img = cv2.resize(img, (128, img.shape[1]))
    elif(img.shape[1] > 96):
        img = cv2.resize(img, (img.shape[0], 96))
    x = cnn00.predict(tf.expand_dims(img, 0))
    y = cnn01.predict(tf.expand_dims(img, 0))
    w = cnn10.predict(tf.expand_dims(img, 0))
    h = cnn11.predict(tf.expand_dims(img, 0))

    cv2.rectangle(img, (int(x),int(y)),(int((x+w)),int((y+h))), (0,255,0), 2)
    cv2.imwrite('cam' +str(i) +'.jpg', img)
    #cv2.imshow('cam' +str(i) +'.jpg', img)
    i+=1
