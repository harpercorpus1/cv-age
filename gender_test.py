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
from tensorflow.keras.layers import Add, Activation, RandomFlip, Input, RandomRotation, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense

from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2

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

def sigmoid_model():
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
        MaxPool2D(pool_size=(2,2)),
        BatchNormalization(),
        Dropout(0.25),

        Flatten(),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(1, activation='sigmoid')                           
    ])
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    return model

def identity(x, num_filter=64):
    x_skip = x

    x = Conv2D(num_filter, (3,3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filter, (3,3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)

    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x

def conv(x, num_filter):
    x_skip = x

    x = Conv2D(num_filter, (3,3), padding='same', strides=(2,2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filter, (3,3), padding='same')(x)    
    x = BatchNormalization(axis=3)(x)

    x_skip = Conv2D(num_filter, (1,1), strides=(2,2))(x_skip)

    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x



def resnet_model():
    input = Input((200,200,3))

    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(input)
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
    x = Dense(256, activation='relu')(x)
    
    x = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs = input, outputs = x, name="Residual-Network")

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    return model

def get_subset(set_n):
    directory = 'UTKFace'
    ds_len = 0 
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            ds_len += 1
        else:
            print(filename)
            continue
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

ageModel = sig()
ageModel.load_weights('./checkpoints/gender_check_Jan_11_1_12')


correct = 0
total = 0

ageGroups = [[0,0] for _ in range(12)]
for count, filename in enumerate(os.listdir('UTKFace')):
    if filename.endswith(".jpg"):
        img = asarray(Image.open(f'UTKFace/{filename}')).reshape(1,200,200,3)
        # plt.imshow(Image.open(f'UTKFace/{filename}'))
        # plt.savefig(f'fig_{count}.jpg')
        pred = round(ageModel.predict(img)[0][0])
        # print(f'{count}_{filename}_{pred}')

        age, gender, _, _ = filename.split('_')
        age = int(age)
        gender = int(gender)
        if pred == gender:
            ageGroups[age // 10][0] += 1
            correct += 1
        ageGroups[age // 10][1] += 1
        total += 1

for count, group in enumerate(ageGroups):
    print(f'{10*count}-{10*(count+1)}: {group[0]}/{group[1]}: {group[0]/group[1]}')


print(f"final accuracy: {correct}/{total} = {correct/total}")