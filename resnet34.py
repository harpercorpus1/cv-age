import numpy as np
from numpy import asarray

import tensorflow as tf
import pandas as pd
import random
import time

from tensorflow.keras import Sequential
from tensorflow.keras.layers import RandomTranslation, GaussianNoise, concatenate, Add, AvgPool2D, Activation, RandomFlip, Input, RandomRotation, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Nadam;

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

def ensemble_structure(image_input, gender_input):
    image_output = layer_1(image_input)
    image_output = layer_2(image_output)
    image_output = layer_3(image_output)
    image_output = layer_4(image_output)
    image_output = layer_5(image_output)
    image_output = outbound_processing(image_output)

    gender_output = Dense_Normal(gender_input)

    combined_input = concatenate([image_output, gender_output])
    
    combined_output = Dense_Normal(combined_input, dimension=256)
    combined_output = Dense_Normal(combined_output, dimension=256)

    return combined_output

def get_linear_resnet34_model():
    image_input = Input((200,200,3))
    gender_input = Input((1,))

    image_input = GaussianNoise(15)(image_input)

    combined_output = ensemble_structure(image_input, gender_input)

    combined_output = Dense(1, activation='linear')(combined_output)

    model = Model(inputs=[image_input, gender_input], outputs=combined_output)

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
    return model

def get_softmax_resnet34_model(num_categories=15, noise_size=15, inputSize=(200,200,3)):
    image_input = Input(inputSize)
    gender_input = Input((1,))

    image_output = tf.image.resize(image_input, [224,224], method='bilinear')
    image_output = GaussianNoise(noise_size)(image_output)
    image_output = RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='constant')(image_output)
    image_output = RandomFlip('horizontal')(image_output)
    image_output = RandomRotation(0.2)(image_output)

    combined_output = ensemble_structure(image_output, gender_input)

    combined_output = Dense(num_categories, activation='softmax')(combined_output)

    model = Model(inputs=[image_input, gender_input], outputs=combined_output)

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

def solo_input_structure(image_input):
    image_output = layer_1(image_input)
    image_output = layer_2(image_output)
    image_output = layer_3(image_output)
    image_output = layer_4(image_output)
    image_output = layer_5(image_output)
    image_output = outbound_processing(image_output)
    
    image_output = Dense_Normal(image_output, dimension=256)
    image_output = Dense_Normal(image_output, dimension=256)

    return image_output

def get_sigmoid_resnet34_model(noise_size=15, inputSize=(200,200,3)):
    image_input = Input(inputSize)

    image_output = GaussianNoise(noise_size)(image_input)
    image_output = RandomFlip('horizontal')(image_output)
    image_output = RandomRotation(0.2)(image_output)

    image_output = solo_input_structure(image_output)

    image_output = Dense(1, activation='sigmoid')(image_output)

    model = Model(inputs=image_input, outputs=image_output)

    opt = Nadam()

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# AxA conv, B /C
# A: Kernel Size
# B: Filters
# C: Strides
