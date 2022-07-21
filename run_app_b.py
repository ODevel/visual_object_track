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

#test = 'human9'
test_p = test +'/' + test +'/'
files = sorted(os.listdir(test_p +'/p/'))
i = 0
for f in files:
    f = test_p + '/p/' + f
    img = cv2.imread(f)
    img_orig= img
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
    print('x:', x, 'y: ', y, img_orig.shape)
    cv2.putText(img_orig, test, (int(x+5),int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.rectangle(img_orig, (int(x),int(y)),(int((x+w)),int((y+h))), (0,255,0), 2)
    cv2.imwrite('cam' +str(i) +'.jpg', img_orig)
    cv2.imshow('cam' +str(i) +'.jpg', img_orig)
    i+=1
