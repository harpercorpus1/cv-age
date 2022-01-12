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
from tensorflow.keras.layers import ZeroPadding2D, Add,  Activation, RandomFlip, Input, RandomRotation, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense

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


def identity(x, num_filter=64):
    x_skip = x

    x = Conv2D(num_filter, (3,3), padding='same')(x)
    x = Dropout(0.2)(x)

    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filter, (3,3), padding='same')(x)
    x = Dropout(0.2)(x)

    x = BatchNormalization(axis=3)(x)

    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x

def conv(x, num_filter):
    x_skip = x

    x = Conv2D(num_filter, (3,3), padding='same', strides=(2,2))(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filter, (3,3), padding='same')(x)    
    x = Dropout(0.2)(x)

    x = BatchNormalization(axis=3)(x)

    x_skip = Conv2D(num_filter, (1,1), strides=(2,2))(x_skip)

    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x



def resnet_model():
    input = Input((200,200,3))

    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(input)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    block_layers = [3,4,6,3]
    filter_size=64
    for i in range(4):
        if i == 0:
            for j in range(block_layers[i]):
                x = identity(x, filter_size)
        else:
            filter_size=filter_size*2
            x = conv(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity(x, filter_size)
    
    x = tf.keras.layers.AveragePooling2D((2,2), padding='same')(x)
    x = Flatten()(x)
    
    # x = Dense(512, activation='relu')(x)
    
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs = input, outputs = x, name="Residual-Network")

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

genderMode = resnet_model()
# print(genderMode.summary())

# genderMode.load_weights('./checkpoints/gender_check_Jan_10_12_11')

random.seed(time.time())
train_times = 1
numSets = 50

for train in range(train_times):
    allNums = random.sample(range(ds_len), ds_len)
    sets = [0]*numSets
    for i in range(numSets):
        sets[i] = sorted(allNums[((ds_len//numSets) * i) : ((ds_len//numSets) * (i+1))])

    for i in range(numSets):
        X, y = get_subset(i)
        trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)
        hist = genderMode.fit(trainX, trainY, epochs=50, batch_size=32, validation_data=(testX, testY), verbose=0)
        print('model accuracy {}-{}: {}'.format(train, i, hist.history['accuracy'][-1]))
        genderMode.save_weights('./checkpoints/gender_check_Jan_10_12_11')

