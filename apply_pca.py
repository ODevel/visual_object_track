import tensorflow as tf
import numpy as np
import cv2
import os
import time
from sklearn.decomposition import PCA

def apply_pca(class_x, next_class=False):
    global l
    global b
    start_time = time.time()
    path = class_x + '/' + class_x + '/p/'
    files = os.listdir(path)
    files_orig = list(files)
    for (i, f) in enumerate(files):
        print('Running PCA on image : ', f)
        files[i] = path + f
        img = cv2.imread(files[i])
        files_ar = np.array(img)
    
        pca_model_r = PCA(n_components = int(img.shape[1]/2), random_state=4)
        pca_model_r.fit(files_ar[:,:,0])
        pca_out_r = pca_model_r.transform(files_ar[:,:,0])
        pca_model_g = PCA(n_components = int(img.shape[1]/2), random_state=4)
        pca_model_g.fit(files_ar[:,:,1])
        pca_out_g = pca_model_g.transform(files_ar[:,:,1])
        pca_model_b = PCA(n_components = int(img.shape[1]/2), random_state=4)
        pca_model_b.fit(files_ar[:,:,2])
        pca_out_b = pca_model_b.transform(files_ar[:,:,2])
        #print ("Variance explained by all 30 principal components ", sum(pca_out.explained_variance_ratio * 100))
        pca_rev_r = tf.expand_dims(pca_model_r.inverse_transform(pca_out_r), 2)
        pca_rev_g = tf.expand_dims(pca_model_g.inverse_transform(pca_out_g), 2)
        pca_rev_b = tf.expand_dims(pca_model_b.inverse_transform(pca_out_b), 2)
        files_ar_t = np.concatenate((pca_rev_r, pca_rev_g), axis=2)
        files_ar_t = np.concatenate((files_ar_t, pca_rev_b), axis=2)
        if(not os.path.exists(class_x + '/' + class_x + '/pca.img/')):
            os.mkdir(class_x + '/' + class_x + '/pca.img')
        cv2.imwrite(class_x + '/' + class_x + '/pca.img/' + files_orig[i], files_ar_t)
        #input()
    end_time = time.time()
    print('Total execution time: ', end_time - start_time)
    return files_ar

