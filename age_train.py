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
from tensorflow.keras.layers import GaussianNoise, concatenate, Add, Activation, RandomFlip, Input, RandomRotation, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

from tensorflow.keras.regularizers import l2

def gen_sig(noise=0, inputSize=(200,200,3)):
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


def conv_layer(data, filters=64):
    output_data = Conv2D(filters, (3,3), activation='relu', padding='same')(data)
    output_data = MaxPool2D(pool_size=(2,2))(output_data)
    output_data = BatchNormalization()(output_data)
    return Dropout(0.25)(output_data)

def age_image_conv_nn(images, noise=15):
    processed_im = RandomFlip("horizontal_and_vertical")(images)
    processed_im = GaussianNoise(noise)(processed_im)
    processed_im = RandomRotation(0.2)(processed_im)

    processed_im = conv_layer(processed_im, 64)
    processed_im = conv_layer(processed_im, 128)
    processed_im = conv_layer(processed_im, 256)

    return conv_layer(processed_im, 512)

def Dense_Normal(input_data, dimension=64, activ='relu', drop_rate=0.25):
    output_data = Dense(dimension, activation=activ)(input_data)
    output_data = BatchNormalization()(output_data)
    return Dropout(drop_rate)(output_data)

def age_gender_processing_ann(gen_data):
    output_data = Dense_Normal(gen_data)
    return Dense_Normal(output_data)

def age_linear_model():
    image_input = Input((200,200,3))
    image_x = age_image_conv_nn(image_input) 
    image_x = Flatten()(image_x)

    gender_input = Input((1,))
    gender_x = age_gender_processing_ann(gender_input)

    combined_input = concatenate([image_x, gender_x])
    
    combined_output = Dense_Normal(combined_input)
    combined_output = Dense_Normal(combined_output)

    combined_output = Dense(1, activation='linear')(combined_output)

    model = Model(inputs=[image_input, gender_input], outputs=combined_output)

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
    return model

def age_softmax_model(num_classes, noise=15, inputSize=(200,200,3)):
    image_input = Input(inputSize)
    image_x = age_image_conv_nn(image_input, noise)
    image_x = Flatten()(image_x)

    gender_input = Input((1,))
    gender_x = age_gender_processing_ann(gender_input)

    combined_input = concatenate([image_x,gender_x])
    
    combined_output = Dense_Normal(combined_input, 256)
    combined_output = Dense_Normal(combined_output, 256)
    combined_output = Dense_Normal(combined_output, 256)
    
    combined_output = Dense(num_classes, activation='softmax')(combined_output)

    model = Model(inputs=[image_input, gender_input], outputs=combined_output)

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    return model

def predict(ageModel, genderModel, input):
    pred = round(genderModel.predict(input)[0][0])
    return ageModel.predict([input, pred])

def produce_categorical_by_bounds(input, upper_bounds=(5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)):
    output = np.zeros((len(upper_bounds)+1) * len(input)).reshape(-1, len(upper_bounds)+1)
    for i_count, i in enumerate(input):
        for j_count, j in enumerate(upper_bounds):
            if i[0] < j:
                output[i_count][j_count] = 1
                break
            if j == upper_bounds[-1]:
                output[i_count][-1] = 1
    return output

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

directory = 'UTKFace_Edited'
ds_len = 0 
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        ds_len += 1
    else:
        print(filename)
        
        continue


def get_subset(set_n, directory='UTKFace_Edited'):
    ds_len = 0 
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            ds_len += 1
        else:
            print(filename)
    found_ind = 0
    df_subset = np.zeros((len(sets[set_n])* 200*200*3), dtype='float32').reshape(-1,200,200,3)
    df_output = np.zeros((len(sets[set_n])), dtype='int8').reshape(len(sets[set_n]), 1)
    for count, filename in enumerate(os.listdir(directory)):
        if count == sets[set_n][found_ind]:
            df_subset[found_ind] = asarray(Image.open(f'{directory}/{filename}'))
            df_output[found_ind][0] = int(filename.split('_')[0])
            found_ind += 1
            if found_ind == len(sets[set_n]):
                break
    return df_subset, df_output

def get_softmax_subset(set_n, upper_bounds, directory='UTKFace_Edited', inputSize=(200,200,3)):
    ds_len = 0 
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            ds_len += 1
        else:
            print(filename)
    found_ind = 0
    df_subset = np.zeros((len(sets[set_n])* inputSize[0]*inputSize[1]*inputSize[2]), dtype='float32').reshape(-1,inputSize[0],inputSize[1],inputSize[2])
    df_output = np.zeros((len(sets[set_n])), dtype='int8').reshape(len(sets[set_n]), 1)
    for count, filename in enumerate(os.listdir(directory)):
        if count == sets[set_n][found_ind]:
            df_subset[found_ind] = asarray(Image.open(f'{directory}/{filename}')).reshape(inputSize[0], inputSize[1], inputSize[2])
            df_output[found_ind][0] = int(filename.split('_')[0])
            found_ind += 1
            if found_ind == len(sets[set_n]):
                break
    return df_subset, produce_categorical_by_bounds(df_output, upper_bounds=upper_bounds)

def get_unique_size_softmax_subset(
                set_n, 
                upper_bounds, 
                gender_directory='UTKFace_bw', 
                gender_input_size=(200,200,1), 
                age_directory='UTKFace_Edited', 
                age_input_size=(200,200,3)
):
    ds_len = 0 
    for filename in os.listdir(gender_directory):
        if filename.endswith(".jpg"):
            ds_len += 1
        else:
            print(filename)
    found_ind = 0
    gender_subset   = np.zeros((len(sets[set_n])* gender_input_size[0]*gender_input_size[1]*gender_input_size[2]), dtype='float32').reshape(-1,gender_input_size[0],gender_input_size[1],gender_input_size[2])
    age_subset      = np.zeros((len(sets[set_n])* age_input_size[0]*age_input_size[1]*age_input_size[2]), dtype='float32').reshape(-1,age_input_size[0],age_input_size[1],age_input_size[2])
    df_output = np.zeros((len(sets[set_n])), dtype='int8').reshape(len(sets[set_n]), 1)
    for count, filename in enumerate(os.listdir(directory)):
        if count == sets[set_n][found_ind]:
            gender_subset[found_ind]    = asarray(Image.open(f'{gender_directory}/{filename}')).reshape(gender_input_size[0], gender_input_size[1], gender_input_size[2])
            age_subset[found_ind]       = asarray(Image.open(f'{age_directory}/{filename}')).reshape(age_input_size[0], age_input_size[1], age_input_size[2])
            
            df_output[found_ind][0] = int(filename.split('_')[0])
            found_ind += 1
            if found_ind == len(sets[set_n]):
                break
    return gender_subset, age_subset, produce_categorical_by_bounds(df_output, upper_bounds=upper_bounds)

random.seed(time.time())
train_times = 15
numSets = 30

bounds = (10, 20, 30, 40, 50, 60)
num_classes = len(bounds) + 1

genderModel = gen_sig(inputSize=(200,200,1))
genderModel.load_weights('./checkpoints/gender_cnn_sig')

from resnet34 import *


gender_directory='UTKFace_bw'
age_directory='UTKFace_Edited'
gender_dimension = (200,200,1)
age_dimension = (200,200,3)

from resnet50 import *

for train in range(train_times):
    for noise in range(3, 0, -1):
        ageModel = resnet34_building_block(inputSize=age_dimension, num_classes=num_classes, noise_size=noise)
        ageModel.load_weights('./checkpoints/age_check_resnet50')

        # ageModel = get_softmax_resnet34_model(num_classes, noise, inputSize=(age_dimension))
        # ageModel.load_weights('./checkpoints/age_check_resnet34')

        # ageModel = age_softmax_model(num_classes=num_classes, noise=noise, inputSize=(200,200,1))
        #ageModel.load_weights('./checkpoints/age_check_cnn_softmax_13')

        allNums = random.sample(range(ds_len), ds_len)
        sets = [0]*numSets
        for i in range(numSets):
            sets[i] = sorted(allNums[((ds_len//numSets) * i) : ((ds_len//numSets) * (i+1))])

        for i in range(int(numSets)):
            genderX, ageX, y = get_unique_size_softmax_subset(i, upper_bounds=bounds, gender_directory=gender_directory, gender_input_size=gender_dimension, age_directory=age_directory, age_input_size=age_dimension)
            gender_train_X, gender_test_X, gender_train_y, gender_test_y = train_test_split(genderX, y, test_size=0.2)
            age_train_X, age_test_X, age_train_y, age_test_y = train_test_split(ageX, y, test_size=0.2)

            train_gender_predictions = np.array([genderModel.predict(i.reshape(-1,gender_dimension[0],gender_dimension[1],gender_dimension[2])) for i in gender_train_X]).reshape(len(gender_train_X),1)
            test_gender_predictions = np.array([genderModel.predict(i.reshape(-1,gender_dimension[0],gender_dimension[1],gender_dimension[2])) for i in gender_test_X]).reshape(len(gender_test_X),1)

            hist = ageModel.fit(
                x=[age_train_X, train_gender_predictions], 
                y = age_train_y, 
                epochs=25, 
                batch_size=16, 
                validation_data=([age_test_X, test_gender_predictions], age_test_y),
                verbose=0
            )
            print(f'iteration-{train} noise-{noise} set-{i} accuracy (train/validation)-({round(hist.history["accuracy"][-1], 3)}/{round(hist.history["val_accuracy"][-1], 3)})')
            #print('iteration-{} noise-{} set-{}: accuracy-{}'.format(train, noise, i, hist.history['accuracy'][-1]))
        ageModel.save_weights('./checkpoints/age_check_resnet50')

# allNums = random.sample(range(ds_len), ds_len)
# sets = [0]*numSets
# for i in range(numSets):
#     sets[i] = sorted(allNums[((ds_len//numSets) * i) : ((ds_len//numSets) * (i+1))])

# X, y = get_subset(0)
# trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)

# for i in trainX:
#     print(i.shape)
#     break