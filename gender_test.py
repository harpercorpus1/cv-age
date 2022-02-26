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
from tensorflow.keras.layers import GaussianNoise, Add, Activation, RandomFlip, Input, RandomRotation, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense

from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2

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


# from resnet34 import *
# genderModel = get_sigmoid_resnet34_model()
# genderModel.load_weights('./checkpoints/gender_check_2_4')

genderModel = sig(noise=0, inputSize=(200,200,1))
print(genderModel.summary())

genderModel.load_weights('./checkpoints/gender_cnn_sig')

correct = 0
total = 0
gender = [[0,0], [0,0]]

directory = 'UTKFace_bw'
for count, filename in enumerate(os.listdir(directory)):
    if filename.endswith(".jpg"):
        img = asarray(Image.open(f'{directory}/{filename}')).reshape(1,200,200,1)
        # plt.imshow(Image.open(f'UTKFace/{filename}'))
        # plt.savefig(f'fig_{count}.jpg')
        pred = round(genderModel.predict(img)[0][0])
        # print(f'{count}_{filename}_{pred}')

        y = int(filename.split('_')[1])

        gender[y][1] += 1
        if pred == y:
            gender[y][0] += 1
            correct += 1

        total += 1


        

print(f'Women: {gender[1][0]}/{gender[1][1]}')
print(f'Men: {gender[0][0]}/{gender[0][1]}')


print(f"final accuracy: {correct}/{total} = {correct/total}")