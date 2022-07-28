import run_app_b
import cv2
import os
import tensorflow as tf
import numpy as np

def run(test):
    cnn_detection = tf.keras.models.load_model('simple_detection.h5')
    clsses = ['biker', 'bird1', 'blurbody', 'blurcar2', 'bolt', 'cardark', 'football', 'human3', 'human6', 'human9', 'panda', 'walking', 'walking2']
    test_p = test +'/' + test +'/'
    files = sorted(os.listdir(test_p +'/p/'))
    model_loaded = {}
    models = {}
    for each_cls in clsses:
        model_loaded[each_cls] = False

    for (i,f) in enumerate(files):
        f = test_p + '/p/' + f
        img_orig = cv2.imread(f)
        img, (lf, bf) = run_app_b.path_to_image(f)
        cls = cnn_detection.predict(img)
        idx = np.argmax(cls)
        test = clsses[idx]
        print('Class predicted: ', test)
        if(model_loaded[test] == False):
            models[test] = [tf.keras.models.load_model(test +'_00.h5')]
            models[test].append(tf.keras.models.load_model(test +'_01.h5'))
            models[test].append(tf.keras.models.load_model(test +'_10.h5'))
            models[test].append(tf.keras.models.load_model(test +'_11.h5'))
            model_loaded[test] = True
        cnn00 = models[test][0]
        cnn01 = models[test][1]
        cnn10 = models[test][2]
        cnn11 = models[test][3]
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
        #cv2.imwrite('cam' +str(i) +'.jpg', img_orig)
        cv2.imshow('cam', img_orig)
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        i+=1 
    cv2.destroyAllWindows()
