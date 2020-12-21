# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 20:26:33 2018

@author: SUDHIR
"""
import sys
import cv2
import os
import glob
import pandas as pd
import numpy as np
# Get user supplied values

cascade = cv2.CascadeClassifier(r'C:\Users\SUDHIR\Anaconda3\pkgs\libopencv-3.4.1-h875b8b8_3\Library\etc\haarcascades\haarcascade_frontalface_default.xml')

# Import & crop Formal Shirt images
cropImages1=[]
img_dir = r"E:\Old D\IIT B\SEM 3\CS725(ML)\Project\dataset\Formal_Shirt" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1,0) 
    data.append(img)
    
for f2 in data:

    faces = cascade.detectMultiScale(
        f2,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE )
        
        
    for (x, y, w, h) in faces:
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        f2=f2[int(y+0.8*h):int(y+2*h), int(x-0.2*w):int(x+1.5*w)]
        
    cropImages1.append(f2)

# Import & crop round neck images

cropImages2=[]
img_dir = r"E:\Old D\IIT B\SEM 3\CS725(ML)\Project\dataset\round_neck" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1,0) 
    data.append(img)
    
for f2 in data:

    faces = cascade.detectMultiScale(
        f2,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE )
        
        
    for (x, y, w, h) in faces:
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        f2=f2[int(y+0.8*h):int(y+2*h), int(x-0.2*w):int(x+1.5*w)]
        
    cropImages2.append(f2)

# Import & crop round V_neck images

cropImages3=[]
img_dir = r"E:\Old D\IIT B\SEM 3\CS725(ML)\Project\dataset\V_neck" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1,0) 
    data.append(img)
    
for f2 in data:

    faces = cascade.detectMultiScale(
        f2,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE )
        
        
    for (x, y, w, h) in faces:
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        f2=f2[int(y+0.8*h):int(y+2*h), int(x-0.2*w):int(x+1.5*w)]
        
    cropImages3.append(f2)

#def image_to_feature_vector(image,size):
    
    
	# resize the image to a fixed size, then flatten the image into a list of raw pixel intensities

    
Final_data=np.zeros((len(cropImages1),1024))
i=0
for f3 in cropImages1:
    
    f3 = cv2.resize(f3,(32,32)).flatten()
    f3=f3.reshape(1,-1)
    Final_data[i,:]=f3
    i=i+1

Final_data1=np.zeros((len(cropImages2),1024))
i=0
for f3 in cropImages2:
    
    f3 = cv2.resize(f3,(32,32)).flatten()
    f3=f3.reshape(1,-1)
    Final_data1[i,:]=f3
    i=i+1
    
Final_data2=np.zeros((len(cropImages3),1024))
i=0
for f3 in cropImages3:
    
    f3 = cv2.resize(f3,(32,32)).flatten()
    f3=f3.reshape(1,-1)
    Final_data2[i,:]=f3
    i=i+1    
 


# Add Labels to images

zero = np.zeros(len(cropImages1))
data1 =np.column_stack((zero,Final_data))
ones = np.ones(len(cropImages2))
data2 =np.column_stack((ones,Final_data1))
two = np.full(len(cropImages3),2)
data3 =np.column_stack((two,Final_data2))


data = np.concatenate((data1, data2,data3))

X_train = data[:,1:1025]
Y_train = data[:,0]  


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=12)
model = knn.fit(X_train,Y_train) 
Y_predict = model.predict(X_train)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix

confusion_matrix = confusion_matrix(Y_train, Y_predict)

print(f1_score(Y_train, Y_predict, average="macro"))
print(precision_score(Y_train, Y_predict, average="macro"))
print(recall_score(Y_train, Y_predict, average="macro"))

# Test 

cropImages_1=[]
img_dir = r"E:\Old D\IIT B\SEM 3\CS725(ML)\Project\testing\Collar" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1,0) 
    data.append(img)
    
for f2 in data:

    faces = cascade.detectMultiScale(
        f2,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE )
        
        
    for (x, y, w, h) in faces:
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        f2=f2[int(y+0.8*h):int(y+2*h), int(x-0.2*w):int(x+1.5*w)]
        
    cropImages_1.append(f2)

test_data_1=np.zeros((len(cropImages_1),1024))
i=0
for f3 in cropImages_1:
    f3 = cv2.resize(f3,(32,32)).flatten()
    f3=f3.reshape(1,-1)
    test_data_1[i,:]=f3
    i=i+1

cropImages_2=[]
img_dir = r"E:\Old D\IIT B\SEM 3\CS725(ML)\Project\testing\Round" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1,0) 
    data.append(img)
    
for f2 in data:

    faces = cascade.detectMultiScale(
        f2,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE )
        
        
    for (x, y, w, h) in faces:
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        f2=f2[int(y+0.8*h):int(y+2*h), int(x-0.2*w):int(x+1.5*w)]
        
    cropImages_2.append(f2)

test_data_2=np.zeros((len(cropImages_2),1024))
i=0
for f3 in cropImages_2:
    f3 = cv2.resize(f3,(32,32)).flatten()
    f3=f3.reshape(1,-1)
    test_data_2[i,:]=f3
    i=i+1
    
cropImages_3=[]
img_dir = r"E:\Old D\IIT B\SEM 3\CS725(ML)\Project\testing\Vneck" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1,0) 
    data.append(img)
    
for f2 in data:

    faces = cascade.detectMultiScale(
        f2,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE )
        
        
    for (x, y, w, h) in faces:
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        f2=f2[int(y+0.8*h):int(y+2*h), int(x-0.2*w):int(x+1.5*w)]
        
    cropImages_3.append(f2)

test_data_3=np.zeros((len(cropImages_3),1024))
i=0
for f3 in cropImages_3:
    f3 = cv2.resize(f3,(32,32)).flatten()
    f3=f3.reshape(1,-1)
    test_data_3[i,:]=f3
    i=i+1   

X_test = np.concatenate((test_data_1, test_data_2,test_data_3))
zero = np.zeros(len(cropImages_1))
ones = np.ones(len(cropImages_2))
two = np.full(len(cropImages_3),2)

Y_actual = np.concatenate((zero, ones,two))

Y_test = model.predict(X_test)

print(f1_score(Y_actual, Y_test, average="macro"))
print(precision_score(Y_actual, Y_test, average="macro"))
print(recall_score(Y_actual, Y_test, average="macro"))


# Show the image
   
#cv2.imshow("Faces found",cropImages[168])
#cv2.waitKey(0) 
#
#cv2.imshow("Faces found",data[215])
#cv2.waitKey(0)

#  Labelling of Images