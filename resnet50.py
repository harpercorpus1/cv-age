import numpy as np
from numpy import asarray, kron

import tensorflow as tf
import pandas as pd
import random
import time

from tensorflow.keras import Sequential
from tensorflow.keras.layers import ZeroPadding2D, RandomTranslation, GaussianNoise, concatenate, Add, AvgPool2D, Activation, RandomFlip, Input, RandomRotation, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Nadam;

from tensorflow.keras.regularizers import l2

def Dense_Normal(input_data, dimension=64, activ='relu', drop_rate=0.25):
    output_data = Dense(dimension, activation=activ)(input_data)
    output_data = BatchNormalization()(output_data)
    return Dropout(drop_rate)(output_data)


def ConvBN(inputs, filters=64, kernel_size=(1,1), strides=(1,1), padding='valid'):
    outputs = Conv2D(filters=filters, kernel_size=kernel_size, 
            strides=strides, padding=padding)(inputs)

    return BatchNormalization()(outputs)

def ConvBNActivate(inputs, filters=64, kernel_size=(1,1), strides=(1,1), padding='valid'):
    outputs = Conv2D(filters=filters, kernel_size=kernel_size, 
            strides=strides, padding=padding)(inputs)

    outputs = BatchNormalization()(outputs)
    return Activation('relu')(outputs)

def identity(input, filters):
    filter1, filter2 = filters

    output = ConvBNActivate(input, filters=filter1, kernel_size=(1,1), strides=(1,1), padding='valid')
    output = ConvBNActivate(output, filters=filter1, kernel_size=(3,3), strides=(1,1), padding='same')
    output = ConvBN(output, filters=filter2, kernel_size=(1,1), strides=(1,1), padding='valid')
    
    output = Add()([input, output])
    return Activation('relu')(output)

def residual(input, filters, strides):
    filter1, filter2 = filters

    output = ConvBNActivate(input, filters=filter1, kernel_size=(1,1), strides=strides, padding='valid')
    output = ConvBNActivate(output, filters=filter1, kernel_size=(3,3), strides=(1,1), padding='same')
    output = ConvBN(output, filters=filter2, kernel_size=(1,1), strides=(1,1), padding='valid')

    output_skip = ConvBN(input, filters=filter2, kernel_size=(1,1), strides=strides, padding='valid')
    
    output = Add()([output, output_skip])
    return Activation('relu')(output)

def data_augmentation(input, noise=0):
    output = GaussianNoise(noise)(input)
    output = RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='constant')(output)
    output = RandomFlip('horizontal')(output)
    return RandomRotation(0.2)(output)

def resnet34_building_block(inputSize=(200,200,3), num_classes=10, noise_size=0):
    input = Input(shape=(inputSize))

    output = tf.image.resize(images=input, size=[224, 224])
    output = GaussianNoise(noise_size)(output)
    output = RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='constant')(output)
    output = RandomFlip('horizontal')(output)
    output = RandomRotation(0.2)(output)

    output = ZeroPadding2D(padding=(3,3))(output)

    output = ConvBNActivate(output, filters=64, kernel_size=(7,7), strides=(2,2), padding='valid')
    output = MaxPool2D(pool_size=(3,3), strides=(2,2))(output)

    output = residual(output, filters=(64, 256), strides=(1,1))
    output = identity(output, filters=(64, 256))
    output = identity(output, filters=(64, 256))

    output = residual(output, filters=(128, 512), strides=(2,2))
    output = identity(output, filters=(128, 512))
    output = identity(output, filters=(128, 512))
    output = identity(output, filters=(128, 512))

    output = residual(output, filters=(256, 1024), strides=(2,2))
    output = identity(output, filters=(256, 1024))
    output = identity(output, filters=(256, 1024))
    output = identity(output, filters=(256, 1024))
    output = identity(output, filters=(256, 1024))
    output = identity(output, filters=(256, 1024))

    output = residual(output, filters=(512, 2048), strides=(2,2))
    output = identity(output, filters=(512, 2048))
    output = identity(output, filters=(512, 2048))

    output = AvgPool2D(pool_size=(2,2), padding='same')(output)

    output = Flatten()(output)

    gender_input = Input(shape=(1,))
    gender_output = Dense_Normal(gender_input, dimension=256)

    combined_output = concatenate([output, gender_output])

    combined_output = Dense_Normal(combined_output, dimension=256)

    combined_output = Dense(num_classes, activation='softmax')(combined_output)
    
    model = Model(inputs=[input, gender_input], outputs=combined_output)

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model

