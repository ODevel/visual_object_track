import cv2
import numpy as np
from PIL import Image
import os


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
import keras_tuner as kt

test = 'blurbody'

def model_builder(hp):
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
    #cnn.add(Dropout(0.5))
    cnn.add(Dense(64, activation="relu"))
    cnn.add(Dense(4, activation="relu"))
    
    
    # Part 3 - Training the CNN
    
    # Compiling the CNN
    #cnn.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam") #optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-1, 1.0])
    cnn.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"], optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)) #optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return cnn
    
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=40,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)



fp = open(test + '/' + test + '/pos.txt')
line = fp.readline()
line_arr = []
while line:
    line_arr.append(line.split(' '))
    line = fp.readline()

# Prepare data
i = 0
for lin in line_arr:
    path = test + '/' + test+'/p/' +lin[0][2:]
    if(not os.path.exists(path)):
        print(path, ': image not found')
        continue
    print('Image found')
    img = cv2.imread(path)
    l = img.shape[0]
    b = img.shape[1]
    img = cv2.resize(img, (145,145))
    x_0 = int(int(lin[2]) * 145/b)
    x_1 = int(int(lin[3]) * 145/l)
    y_0 = int(int(lin[4]) * 145/b)
    y_1 = int(int(lin[5]) * 145/l)

    #cv2.rectangle(img,(x_0,x_1), (y_0+x_0,y_1+x_1), (0,255,0), 2)
    ##cv2.rectangle(img,(x_0,x_0), (x_1,y_1), (0,255,0), 2)
    #cv2.imwrite('cam.jpg', img)
    #input()

    point = np.array([[[x_0, x_1, x_0 + y_0, x_1 + y_1]]])
    if(i == 0):
        X = np.array([img])
        y = point
        i += 1
    else:
        X = np.concatenate((X , np.array([img])))
        y = np.concatenate((y, point))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
test_generator = ImageDataGenerator(rescale=1/255)

train_generator = train_generator.flow(np.array(X_train), y_train, shuffle=False)
test_generator = test_generator.flow(np.array(X_test), y_test, shuffle=False)
#train_set = np.concatenate((np.reshape(X_train, (len(X_train),1)), np.reshape(y_train, (len(y_train),1))), axis=1)
#test_set = np.concatenate((np.reshape(X_test, (len(X_test),1)), np.reshape(y_test, (len(y_test),1))), axis=1)



tuner.search(X_train,tf.expand_dims(y_train,1), epochs=40, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=10)[0]

cnn = tuner.hypermodel.build(best_hps)
#history = cnn.fit(train_generator, epochs=40, validation_data=test_generator, shuffle=True, validation_steps=len(y))

print('--Traing is done --\n')
   
