import cv2
import numpy as np
from PIL import Image
import os
import sys

# Path for face image database
if(len(sys.argv) != 2):
    print('''
    Usage:
    python train_obj_opencv.py <test_name>
    ''')
    
test = sys.argv[1]
path = test + '/' + test + '/' + '/p/'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(test + '/' + test + '/cascade/' + 'cascade.xml');

# function to get the images and label data
def getImagesAndLabels(path, id):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        print(imagePath)
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        #id = int(os.path.split(imagePath)[-1].split("_")[0])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path,0)
path = './n/'
faces1,ids1 = getImagesAndLabels(path,1)
for f in faces1:
    faces.append(f)
for i in ids1:
    ids.append(i)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
if(not os.path.exists(test+ '/' + test + '/trainer')):
    os.mkdir(test + '/' + test + '/trainer')

recognizer.write(test+'/' + test + '/trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
