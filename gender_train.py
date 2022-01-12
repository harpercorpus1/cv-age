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

directory = 'UTKFace'
ds_len = 0 
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        ds_len += 1
    else:
        print(filename)
        
        continue

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

def sig():
    model = Sequential([
        Input((200,200,3)),

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

def get_subset(set_n):
    found_ind = 0
    df_subset = np.zeros((len(sets[set_n])* 200*200*3), dtype='float32').reshape(-1,200,200,3)
    df_output = np.zeros((len(sets[set_n])), dtype='int8').reshape(len(sets[set_n]), 1)
    for count, filename in enumerate(os.listdir(directory)):
        if count == sets[set_n][found_ind]:
            df_subset[found_ind] = asarray(Image.open(f'UTKFace/{filename}'))
            df_output[found_ind][0] = int(filename.split('_')[1])
            found_ind += 1
            if found_ind == len(sets[set_n]):
                break
    return df_subset, df_output

genderMode = sig()

# genderMode.load_weights('./checkpoints/gender_check_Jan_11_1_12')

random.seed(time.time())
train_times = 1
numSets = 50

for train in range(train_times):
    allNums = random.sample(range(ds_len), ds_len)
    sets = [0]*numSets
    for i in range(numSets):
        sets[i] = sorted(allNums[((ds_len//numSets) * i) : ((ds_len//numSets) * (i+1))])

    for i in range(int(numSets*0.6)):
        X, y = get_subset(i)
        trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)
        hist = genderMode.fit(trainX, trainY, epochs=50, batch_size=64, validation_data=(testX, testY), verbose=0)
        print('model accuracy {}-{}: {}'.format(train, i, hist.history['accuracy'][-1]))
        genderMode.save_weights('./checkpoints/gender_check_Jan_11_9_24')

# allNums = random.sample(range(ds_len), ds_len)
# sets = [0]*numSets
# for i in range(numSets):
#     sets[i] = sorted(allNums[((ds_len//numSets) * i) : ((ds_len//numSets) * (i+1))])

# X, y = get_subset(0)
# trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)

# for i in range(5):
#     print(X[i].shape)
#     print(y[i])

