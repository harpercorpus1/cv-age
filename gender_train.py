import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
from numpy import asarray

import tensorflow as tf
import pandas as pd
import random
import time

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Add, Activation, RandomFlip, Input, RandomRotation, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from tensorflow.keras.regularizers import l2

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def data_aug(x):
    x = RandomFlip("horizontal_and_vertical")(x)
    x = RandomRotation(0.2)(x)
    return x 

def conv_pool(x, filters):
    x = Conv2D(filters, (3,3), padding='same', strides=(2,2))(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization(axis=1)(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    
    return Activation('relu')(x)

def conv(x, filters):
    # x_skip = x
    x = Conv2D(filters, (3,3), padding='same', strides=(2,2))(x)
    x = Dropout(0.25)(x)

    x = BatchNormalization(axis=1)(x)
    # x_skip = Conv2D(filters, (1,1), strides=(2,2))(x_skip)

    # x = Add()([x, x_skip])    
    return x #Activation('relu')(x)

def sigmoid_model():
    input = Input((200,200,3))
    
    x = data_aug(input)

    x = conv(x, 64)
    x = conv_pool(x, 128)
    x = conv(x, 256)
    x = conv_pool(x, 512)


    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    x = Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs = input, outputs = x, name="sigmoid-model")

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    
    return model

def sig(noise=0, inputSize=(200,200,3)):
    model = Sequential([
        Input(inputSize),

        GaussianNoise(noise),
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        
        Conv2D(64, (3,3), activation='relu'),
        MaxPool2D(pool_size=(2,2)),
        BatchNormalization(),
        Dropout(0.25),

        Conv2D(128, (3,3), activation='relu'),
        MaxPool2D(pool_size=(2,2)),
        BatchNormalization(),
        Dropout(0.25),

        Conv2D(256, (3,3), activation='relu'),
        MaxPool2D(pool_size=(2,2)),
        BatchNormalization(),
        Dropout(0.25),

        Conv2D(512, (3,3), activation='relu'),
        MaxPool2D(pool_size=(2,2)),
        BatchNormalization(),
        Dropout(0.25),

        Conv2D(1024, (3,3), activation='relu'),
        BatchNormalization(),
        Dropout(0.25),

        Flatten(),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(1, activation='sigmoid')                           
    ])
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    return model

def get_subset(set_n, directory='UTKFace'):
    if directory == 'UTKFace_bw':
        inputSize=(200,200,1)
    else:
        inputSize=(200,200,3)

    found_ind = 0
    df_subset = np.zeros((len(sets[set_n])* inputSize[0] * inputSize[1] * inputSize[2]), dtype='float32').reshape(-1,inputSize[0],inputSize[1],inputSize[2])
    df_output = np.zeros((len(sets[set_n])), dtype='int8').reshape(len(sets[set_n]), 1)
    for count, filename in enumerate(os.listdir(directory)):
        if count == sets[set_n][found_ind]:
            df_subset[found_ind] = asarray(Image.open(f'{directory}/{filename}')).reshape(inputSize)
            df_output[found_ind][0] = int(filename.split('_')[1])
            found_ind += 1
            if found_ind == len(sets[set_n]):
                break
    return df_subset, df_output

from resnet34 import *

random.seed(time.time())
train_times = 15
numSets = 20

from resnet34 import *


directory = 'UTKFace_bw'
ds_len = 0 
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        ds_len += 1
    else:
        print(filename)
        
        continue

for train in range(train_times):
    for noise in range(15, 0, -1):
        allNums = random.sample(range(ds_len), ds_len)
        sets = [0]*numSets
        for i in range(numSets):
            sets[i] = sorted(allNums[((ds_len//numSets) * i) : ((ds_len//numSets) * (i+1))])

        # genderModel = get_sigmoid_resnet34_model(noise, (200,200,1))
        # genderModel.load_weights('./checkpoints/gender_resnet_bw')
        genderModel = sig(noise=noise, inputSize=(200,200,1))
        genderModel.load_weights('./checkpoints/gender_cnn_sig')
        for i in range(int(numSets)):
            X, y = get_subset(i, 'UTKFace_bw')
            trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)
            hist = genderModel.fit(
                trainX, 
                trainY, 
                epochs=30, 
                batch_size=32, 
                validation_data=(testX, testY), 
                verbose=0
            )
            print(f'iteration-{train} noise-{noise} set-{i} accuracy (train/validation)-({round(hist.history["accuracy"][-1], 3)}/{round(hist.history["val_accuracy"][-1], 3)})')
            # print('iteration-{} noise-{} set-{}: accuracy-{}'.format(train, noise, i, hist.history['accuracy'][-1]))
            genderModel.save_weights('./checkpoints/gender_cnn_sig')

# allNums = random.sample(range(ds_len), ds_len)
# sets = [0]*numSets
# for i in range(numSets):
#     sets[i] = sorted(allNums[((ds_len//numSets) * i) : ((ds_len//numSets) * (i+1))])

# X, y = get_subset(0)
# trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)

# for i in range(5):
#     print(X[i].shape)
#     print(y[i])

