import numpy as np
from numpy import asarray

import tensorflow as tf
import pandas as pd
import random
import time

from tensorflow.keras import Sequential
from tensorflow.keras.layers import concatenate, Add, AvgPool2D, Activation, RandomFlip, Input, RandomRotation, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

from tensorflow.keras.regularizers import l2

def ConvBN(inputs, filters=64, kernel_size=(1,1), strides=(1,1), padding='valid'):
    outputs = Conv2D(filters=filters, kernel_size=kernel_size, 
            strides=strides, padding=padding)(inputs)

    return BatchNormalization()(outputs)

def ResidualBasicBlock(inputs, filters):
    outputs = ConvBN(inputs, filters = filters, strides=(1,1), padding='same')
    outputs = Activation('relu')(outputs)
    return ConvBN(outputs, filters = filters, strides=(1,1), padding='same')

def ResidualBasicBlockShortcut(inputs, filters):
    outputs = ConvBN(inputs, filters=filters, strides=(2,2), padding='same')
    outputs = Activation('relu')(outputs)
    outputs = ConvBN(outputs, filters = filters, strides=(1,1), padding='same')
    shortcuts = ConvBN(inputs, filters=1, strides=(2,2), padding='same')
    return Add()([outputs,shortcuts])

def layer_1(inputs):
    outputs = ConvBN(inputs, filters=64, kernel_size=(7,7), strides=(2,2), padding='same')
    return MaxPool2D(pool_size=(2,2), padding='same')(outputs)

def layer_2(inputs):
    outputs = ResidualBasicBlock(inputs, filters=64)
    outputs = ResidualBasicBlock(outputs, filters=64)
    return ResidualBasicBlock(outputs, filters=64)

def layer_3(inputs):
    outputs = ResidualBasicBlockShortcut(inputs, filters=128)
    outputs = ResidualBasicBlock(outputs, filters=128)
    outputs = ResidualBasicBlock(outputs, filters=128)
    return ResidualBasicBlock(outputs, filters=128)

def layer_4(inputs):
    outputs = ResidualBasicBlockShortcut(inputs, filters=256)
    outputs = ResidualBasicBlock(outputs, filters=256)
    outputs = ResidualBasicBlock(outputs, filters=256)
    outputs = ResidualBasicBlock(outputs, filters=256)
    outputs = ResidualBasicBlock(outputs, filters=256)
    return ResidualBasicBlock(outputs, filters=256)

def layer_5(inputs):
    outputs = ResidualBasicBlockShortcut(inputs, filters=512)
    outputs = ResidualBasicBlock(outputs, filters=512)
    return ResidualBasicBlock(outputs, filters=512)

def outbound_processing(inputs):
    outputs = AvgPool2D(pool_size=(7,7), padding='same')(inputs)
    return Flatten()(outputs)

def Dense_Normal(input_data, dimension=64, activ='relu', drop_rate=0.25):
    output_data = Dense(dimension, activation=activ)(input_data)
    output_data = BatchNormalization()(output_data)
    return Dropout(drop_rate)(output_data)

def get_resnet34_model():
    image_input = Input((200,200,3))
    
    image_output = layer_1(image_input)
    image_output = layer_2(image_output)
    image_output = layer_3(image_output)
    image_output = layer_4(image_output)
    image_output = layer_5(image_output)
    image_output = outbound_processing(image_output)

    gender_input = Input((1,))

    gender_output = Dense_Normal(gender_input)

    combined_input = concatenate([image_output, gender_output])
    
    combined_output = Dense_Normal(combined_input, dimension=256)
    combined_output = Dense_Normal(combined_output, dimension=256)

    combined_output = Dense(1, activation='linear')(combined_output)

    model = Model(inputs=[image_input, gender_input], outputs=combined_output)

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
    return model




# AxA conv, B /C
# A: Kernel Size
# B: Filters
# C: Strides
