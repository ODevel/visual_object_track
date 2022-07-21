import tensorflow as tf
import numpy as np
import cv2 
import os


# tests - blurbody, bird, panda

l = 128
b = 96
def return_files(class_x, next_class=False):
    global l
    global b
    path = class_x + '/' + class_x + '/test/'
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

def class_name(val):
    if(np.int_(val)[-1] == 1):
        return 'blurbody'
    elif(np.int_(val)[1] == 1):
        return 'panda'
    else:
        return 'bird'


class_a = 'blurbody'
class_b = 'bird1'
class_c = 'panda'

cnn = tf.keras.models.load_model('simple_detection.h5')
files_a = return_files(class_a)
files_b = return_files(class_b, next_class=True)
files_c = return_files(class_c, next_class=True)

prediction_a = cnn.predict(files_a)
prediction_b = cnn.predict(files_b)
prediction_c = cnn.predict(files_c)

files_a[0] = cv2.putText(files_a[0], class_name(prediction_a[0]), (int(files_a[0].shape[0]//10), int(files_a[0].shape[1]//5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
cv2.imwrite('cam.jpg', files_a[0])
files_b[0] = cv2.putText(files_b[0], class_name(prediction_b[0]), (int(files_b[0].shape[0]//10), int(files_b[0].shape[1]//5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
cv2.imwrite('cam1.jpg', files_b[0])
files_c[0] = cv2.putText(files_c[0], class_name(prediction_c[0]), (int(files_c[0].shape[0]//10), int(files_c[0].shape[1]//5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
cv2.imwrite('cam2.jpg', files_c[0])
