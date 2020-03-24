from __future__ import print_function

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from keras import applications
import cv2 
from os import listdir
import numpy as np
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, AveragePooling2D
from keras.models import Model

root_directory = "./data_two_class"

train_data=[]
test_data=[]
train_label=[]
test_label=[]

img_row,img_col = 160,160

nb_classes = 2

data_type =["train","test"]
folder_names=["covid","normal"]
label_mapping={"covid":0,"normal":1}


def process_img(dataset,folder,filename): 
    file_path = root_directory+"/"+dataset+"/"+folder+"/"+filename
    img = cv2.imread(file_path)
    image = cv2.resize(img,(img_row,img_col))
    return image 

for dataset in data_type:
    for folder in folder_names:
        for filename in listdir(root_directory+"/"+dataset+"/"+folder+"/"):
            if dataset is "train":
                train_data.append(process_img(dataset,folder,filename))
                train_label.append(label_mapping[folder])
            else:
                test_data.append(process_img(dataset,folder,filename))
                test_label.append(label_mapping[folder])

train_data = np.asarray(train_data)
train_label = np.asarray(train_label)

test_data = np.asarray(test_data)
test_label = np.asarray(test_label)


x_train = train_data.reshape(train_data.shape[0], img_row, img_col, 3)
x_test = test_data.reshape(test_data.shape[0], img_row, img_col, 3)

y_train = np_utils.to_categorical(train_label, nb_classes)
y_test = np_utils.to_categorical(test_label, nb_classes)



resnet_model = applications.resnet50.ResNet50(weights='imagenet',
                               include_top=False,
                               input_shape=(160, 160, 3))


x = resnet_model.output

x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(nb_classes, activation='softmax')(x)

custom_model = Model(input=resnet_model.input, output=x)

for layer in custom_model.layers[:7]:
    layer.trainable = False

custom_model.compile(loss='binary_crossentropy',
                     optimizer='Adam',
                     metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1./255,
 rotation_range=50,
 horizontal_flip = True ,
 vertical_flip = True ,
 validation_split = 0.2)

datagen.fit(x_train)


custom_model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, verbose=1,epochs=25,validation_data=(x_test,y_test))
