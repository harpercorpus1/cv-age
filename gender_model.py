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
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2

def conv_and_pool(x, num_filters=64):
    output = Conv2D(num_filters, (3,3), activation='relu')(x)
    output = MaxPool2D(pool_size=(2,2))(output)
    output = BatchNormalization()(output)
    return Dropout(0.25)(output)

def Dense_and_Normalize(x, output_dim=64, activation='relu'):
    output = Dense(output_dim, activation=activation)(x)
    output = BatchNormalization()(output)
    return Dropout(0.2)(output)

def gender_model():
    input = Input(200,200,3)
    output = RandomFlip("horizontal_and_vertical")(input)
    output = RandomRotation(0.2)(output)

    output = conv_and_pool(output, 64)
    output = conv_and_pool(output, 128)
    output = conv_and_pool(output, 256)
    output = conv_and_pool(output, 512)

    output = Flatten()(output)

    output = Dense_and_Normalize(output, 256)
    output = Dense_and_Normalize(output, 256)

    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=input, outputs=output)

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    
    return model



    
