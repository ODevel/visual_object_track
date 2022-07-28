import tensorflow as tf
import cv2
import numpy as np
import os
import sys

def path_to_image(path):
    img = cv2.imread(path)
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
    elif(img.shape[0] > 128):
        l = 128
        lf = 128/img.shape[0]
    elif(img.shape[1] > 96):
        b = 96
        bf = 96/img.shape[1]

    img = cv2.resize(img, (b,l))
    img = tf.expand_dims(img,0)
    return img, (lf, bf)

def run(image_path):
    cnn_detection = tf.keras.models.load_model('simple_detection.h5')
    clsses = ['biker', 'bird1', 'blurbody', 'blurcar2', 'bolt', 'cardark', 'football', 'human3', 'human6', 'human9', 'panda', 'walking', 'walking2']
    image_path += '/'+ image_path +'/p/'
    im = os.listdir(image_path)[0]
    image_path += '/' + im
    img_orig = cv2.imread(image_path)
    print(image_path)
    img, (lf, bf) = path_to_image(image_path)
    cls = cnn_detection.predict(img)
    idx = np.argmax(cls)
    test = clsses[idx]
    print('Class predicted: ', test)
    cnn00 = tf.keras.models.load_model(test +'_00.h5')
    cnn01 = tf.keras.models.load_model(test +'_01.h5')
    cnn10 = tf.keras.models.load_model(test +'_10.h5')
    cnn11 = tf.keras.models.load_model(test +'_11.h5')
    
    #test = 'human9'
    x = cnn00.predict(img)
    y = cnn01.predict(img)
    w = cnn10.predict(img)
    h = cnn11.predict(img)
    
    x /= bf
    y /= lf
    w /= bf
    h /= lf
    print('x:', x, 'y: ', y, img_orig.shape)
    cv2.putText(img_orig, test, (int(x+5),int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.rectangle(img_orig, (int(x),int(y)),(int((w)),int((h))), (0,255,0), 2)
    #cv2.imwrite('cam' +'001' +'.jpg', img_orig)
    cv2.imshow('cam' +'001' +'.jpg', img_orig)
    cv2.waitKey()
