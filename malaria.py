
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


DATADIR = "/Users/indradumna/desktop/Malaria-cells-ML-code/Sample_images"

Categories = ["P1","P2"]
for i, a in enumerate (Categories):
    print (i, a)

for i in Categories:
    path = os.path.join (DATADIR, i) # path to directory
    for img in os.listdir(path):
        image_array= cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow (image_array,cmap="gray")
        plt.show()
        break
        break
        
  
print (len(image_array))

print (image_array)

print (image_array.shape)


IMG_SIZE =60
new1_array = cv2.resize(image_array,(IMG_SIZE,IMG_SIZE))
plt.imshow (new1_array,cmap='gray')
plt.show()
print (len(new1_array))

from tqdm import tqdm
training_data = []
def create_training_data():
    for i in Categories:
        path = os.path.join (DATADIR,i) # path to directory
        Class_num = Categories.index (i)
        for img in tqdm(os.listdir(path)):
            try:
                image_array= cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array= cv2.resize(image_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,Class_num])
            except Exception as e:
                pass
                
create_training_data()


print (len(training_data))       

print (len(training_data))


import random
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])
    
x= []
y= []

for features,label in training_data:
    x.append(features)
    y.append(label)


print(x[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

import pickle

pickle_out = open("x.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

import pickle

pickle_in = open("x.pickle","rb")
x = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

x = x/255.0

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(16))
model.add(Activation('sigmoid'))

model.add(Dense(1))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit (x,y,epochs=5,batch_size=1,validation_split=0.2)

from keras.callbacks import TensorBoard

import time 

NAME = "testing".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.fit(x, y,
                      batch_size=32,
                      epochs=10,
                      validation_split=0.3,
                      callbacks=[tensorboard])


model.save ('testing.model')


#Open Macterminal to view tensorboard data on generated link for macbook
#Type in terminal: tensorboard --logdir=/Users/indradumna/logs/Mal-vs-Healthy-CNN
# Open http://Indradumnas-MacBook-Pro.local:6006




filepath = "/Users/indradumna/Desktop/cell_images/testing"
CATEGORIES = ["P1", "P2"]


def prepare(filepath):
    IMG_SIZE = 60  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

plt.imshow (new_array,cmap='gray')

model = keras.models.load_model("testing.model")




#give the full path of the image
prediction = model.predict([prepare('/Users/indradumna/Desktop/cell_images/testing/1.png')])



print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])


print(CATEGORIES[int(np.argmax(prediction[0]))])


