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
from tensorflow.keras.layers import concatenate, Add, Activation, RandomFlip, Input, RandomRotation, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

from tensorflow.keras.regularizers import l2

def gen_sig():
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

def conv_layer(data, filters=64):
    output_data = Conv2D(filters, (3,3), activation='relu', padding='same')(data)
    output_data = MaxPool2D(pool_size=(2,2))(output_data)
    output_data = BatchNormalization()(output_data)
    return Dropout(0.25)(output_data)

def age_image_conv_nn(images):
    processed_im = RandomFlip("horizontal_and_vertical")(images)
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

def age_softmax_model(num_classes):
    image_input = Input((200,200,3))
    image_x = age_image_conv_nn(image_input)
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

def produce_categorical_by_bounds(input, num_classes):
    ageMax = 116 ; ageMin = 0
    bound_len = ((ageMax-ageMin)/(num_classes))+0.01
    output = np.zeros(num_classes*len(input)).reshape(len(input), num_classes)
    for count, i in enumerate(input):
        output[count][int(i[0]/bound_len)] = 1
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

directory = 'UTKFace'
ds_len = 0 
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        ds_len += 1
    else:
        print(filename)
        
        continue

genderModel = gen_sig()
genderModel.load_weights('./checkpoints/gender_check_Jan_11_1_12')

def get_subset(set_n):
    directory = 'UTKFace'
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
            df_subset[found_ind] = asarray(Image.open(f'UTKFace/{filename}'))
            df_output[found_ind][0] = int(filename.split('_')[0])
            found_ind += 1
            if found_ind == len(sets[set_n]):
                break
    return df_subset, df_output

def get_softmax_subset(set_n, num_classes):
    directory = 'UTKFace'
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
            df_subset[found_ind] = asarray(Image.open(f'UTKFace/{filename}'))
            df_output[found_ind][0] = int(filename.split('_')[0])
            found_ind += 1
            if found_ind == len(sets[set_n]):
                break
    return df_subset, produce_categorical_by_bounds(df_output, num_classes=num_classes)


random.seed(time.time())
train_times = 10
numSets = 50

from resnet34 import *

# ageModel = get_resnet34_model(5)

# ageModel = age_softmax_model(5)
# ageModel.load_weights('./checkpoints/age_check_softmax_1_25')

for train in range(train_times):
    ageModel = get_resnet34_model()
    ageModel.load_weights('./checkpoints/age_check_resnet_34_256')
    allNums = random.sample(range(ds_len), ds_len)
    sets = [0]*numSets
    for i in range(numSets):
        sets[i] = sorted(allNums[((ds_len//numSets) * i) : ((ds_len//numSets) * (i+1))])

    for i in range(int(numSets*0.6)):
        X, y = get_subset(i)
        trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)
        train_gender_predictions = np.array([genderModel.predict(i.reshape(-1,200,200,3)) for i in trainX]).reshape(len(trainX),1)
        test_gender_predictions = np.array([genderModel.predict(i.reshape(-1,200,200,3)) for i in testX]).reshape(len(testX),1)
        # complete_train_X = tf.data.Dataset.zip((trainX, train_gender_predictions))
        # complete_test_X = tf.data.Dataset.zip((testX, test_gender_predictions))
        hist = ageModel.fit(
            x=[trainX, train_gender_predictions], 
            y = trainY, 
            epochs=100, 
            batch_size=128, 
            validation_data=([testX, test_gender_predictions], testY) #,
            # verbose=0
        )
        print('model mse {}-{}: {}'.format(train, i, hist.history['mse'][-1]))
        ageModel.save_weights('./checkpoints/age_check_resnet_34_256')

# allNums = random.sample(range(ds_len), ds_len)
# sets = [0]*numSets
# for i in range(numSets):
#     sets[i] = sorted(allNums[((ds_len//numSets) * i) : ((ds_len//numSets) * (i+1))])

# X, y = get_subset(0)
# trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)

# for i in trainX:
#     print(i.shape)
#     break