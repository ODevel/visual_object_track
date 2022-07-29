# Visual Object Tracking
Welcome to the github repository for the project "Visual Object Tracking". This readme explains how to use the python files in order to train and execute sample tests on the Neural Net models.
## 3 different approaches
As mentioned in the report, we have python scripts for three different approaches. The functionality written in these files can be accessed from the top level `cli.py` or can be accessed individually.
Following is a brief summary about these files.
###### train_app_a.py | run_app_a.py
Use these files to train and run neural network based upon Approach A methodology. Use train and run API to do so. Provide test case name wherever necessary. After training, `simple_detection.h5` checkpoint file should be generated in `checkpoints` folder.
###### train_app_b.py 
Use this file to train the neural network for Approach B. There will be four checkpoint files that should be generated, ending by the * binary * index of the position annotation. These checkpoints will again be generated in the `checkpoints` folder.
###### train_app_b_tuning.py
This file can be used to train the model with extra hyper-tuning enabled for various attributes and hyper-parameters mentioned inside the model. The checkpoint files that this model generates will have a similar name as that of `train_app_b.py`. So you can use only once at a time.
###### run_app_b.py | run_app_b_vid.py
Use these files to try the Approach B that got trained either for a single file or for whole set of images respectively. The result of this would be output images with object detected in the form of rectangular annotation around the object.
###### train_app_c.py
Use this file to train `trainer.yml` required for OpenCV based cascade recognition. You will need to provide the test case name for which to train the yml file.
###### run_app_c.py
This file will pick the `trainer.yml` and `cascade.xml` to detect and track the object again using a rectangular box around the object, similar to Approach B.
## Directory structure
The directory structure of the repository is divided as follows:
* <test-dataset> / <test-dataset> / *
                              * /p/  <images> *  --> Data images
                              * /cascade/ *   --> cascade.xml (OpenCV)
* checkpoints/ *    --> Folder containing checkpoint files
* n/ *    --> negative data images
* Python files are all present at top level directory structure itself. *

*** Notes: ***
1) For each approach, the model has to be trained first before makeing use of it.
2) Approach A, is meant for object detection only, although Approach B still uses it for the detection part.
3) All the above python scripts are written in the form of API's that can be accessed either through Python interpreter OR a top level `cli.py` file provided.
