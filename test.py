import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
from numpy import asarray

import tensorflow as tf
import os
import pandas as pd
import random
import time

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, RandomFlip, Input, RandomRotation, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense

from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2

# directory = 'UTKFace'
# ds_len = 0 
# for filename in os.listdir(directory):
#     if filename.endswith(".jpg"):
#         ds_len += 1
#     else:
#         print(filename)
        
#         continue

# random.seed(time.time())
# allNums = random.sample(range(ds_len), ds_len)
# sets = [0]*10
# for i in range(10):
#     sets[i] = sorted(allNums[((ds_len//10) * i) : ((ds_len//10) * (i+1))])

def online_model():
    model = Sequential([
        Input((200,200,3)),

        Conv2D(32, kernel_size=(3,3), padding='same', strides=(1,1), 
        activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.25),
        Activation('relu'),

        MaxPool2D(pool_size=(2,2)),

        Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), 
        activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.25),
        Activation('relu'),

        MaxPool2D(pool_size=(2,2)),

        Conv2D(128, kernel_size=(3,3), padding='same', strides=(1,1), 
        activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.25),
        Activation('relu'),

        MaxPool2D(pool_size=(2,2)),

        Conv2D(256, kernel_size=(3,3), padding='same', strides=(1,1), 
        activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.25),
        Activation('relu'),

        MaxPool2D(pool_size=(2,2)),

        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')

    ])

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
    return model


def linear_model():
    model = Sequential([
        Input((200,200,3)),

        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),

        MaxPool2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),

        Dropout(0.25),

        Flatten(),

        Dense(256, activation='relu'),
        BatchNormalization(),

        Dropout(0.25),

        Dense(1, activation='linear')                           
    ])
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])
    return model

def get_subset(set_n):
    found_ind = 0
    df_subset = np.zeros((len(sets[set_n])* 200*200*3), dtype='float32').reshape(-1,200,200,3)
    df_output = np.zeros((len(sets[set_n])), dtype='int8').reshape(len(sets[set_n]), 1)
    for count, filename in enumerate(os.listdir(directory)):
        if count == sets[set_n][found_ind]:
            df_subset[found_ind] = asarray(Image.open(f'UTKFace/{filename}'))
            df_output[found_ind][0] = int(filename.split('_')[0])
            found_ind += 1
            if found_ind == len(sets[set_n]):
                break
    return df_subset, df_output

ageModel = linear_model()
ageModel.load_weights('./checkpoints/age_check_Jan_6_12_44')

for count, filename in enumerate(os.listdir('UTKFace')):
    if count > 4: 
        break
    if filename.endswith(".jpg"):
        img = asarray(Image.open(f'UTKFace/{filename}')).reshape(1,200,200,3)
        plt.imshow(Image.open(f'UTKFace/{filename}'))
        plt.savefig(f'fig_{count}.jpg')
        pred = ageModel(img)
        print(f'{count}_{filename}_{pred}')


# def PreProcessData(df):
#     retdf = np.zeros(len(df)*200*200*1, dtype='float32').reshape(len(df),200,200,1)
#     for ind, img in enumerate(df):
#         print(f'{ind}/{len(df)}')
#         for i in range(200):
#             for j in range(200):
#                 retdf[ind][i][j] = np.sum(img[i][j]) / 768.0
    

# for i in range(10):
#     X, y = get_subset(i)
#     trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)
#     # trainX = PreProcessData(trainX); testX = PreProcessData(testX)
#     print(len(trainX))
#     print(len(testX))
#     ageModel.fit(trainX, trainY, epochs=10, batch_size=128, validation_data=(testX, testY))

# ageModel.save_weights('./checkpoints/age_check')

# X, y = get_subset(i)
# trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)
# ageModel.fit(trainX, trainY, epochs=5, batch_size=128, validation_data=(testX, testY))