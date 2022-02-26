import os
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
    output_data = Conv2D(filters, (3,3), activation='relu')(data)
    output_data = MaxPool2D(pool_size=(2,2))(output_data)
    output_data = BatchNormalization()(output_data)
    return Dropout(0.25)(output_data)

def age_image_conv_nn(images):
    processed_im = RandomFlip("horizontal_and_vertical")(images)
    processed_im = RandomRotation(0.2)(processed_im)

    processed_im = conv_layer(processed_im, 64)
    processed_im = conv_layer(processed_im, 64)
    processed_im = conv_layer(processed_im, 64)

    return conv_layer(processed_im, 64)

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

def predict(ageModel, genderModel, input):
    pred = round(genderModel.predict(input)[0][0])
    return ageModel.predict([input, pred])


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

def produce_categorical_by_bounds(input, num_classes):
    ageMax = 116 ; ageMin = 0
    bound_len = ((ageMax-ageMin)/(num_classes))+0.01
    output = np.zeros(num_classes*len(input)).reshape(len(input), num_classes)
    for count, i in enumerate(input):
        output[count][int(i[0]/bound_len)] = 1
    return output

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



# ageGroups = [[0,0] for _ in range(12)]
# for count, filename in enumerate(os.listdir('UTKFace')):
#     if filename.endswith(".jpg"):
#         img = asarray(Image.open(f'UTKFace/{filename}')).reshape(1,200,200,3)
#         # plt.imshow(Image.open(f'UTKFace/{filename}'))
#         plt.savefig(f'fig_{count}.jpg')
#         print(predict(ageModel, genderModel, img))
#         # print(f'{count}_{filename}_{pred}')


# for count, group in enumerate(ageGroups):
#     print(f'{10*count}-{10*(count+1)}: {group[0]}/{group[1]}: {group[0]/group[1]}')


# print(f"final accuracy: {correct}/{total} = {correct/total}")


genderModel = gen_sig()
genderModel.load_weights('./checkpoints/gender_check_Jan_11_1_12')

# ageModel = age_linear_model()
# ageModel.load_weights('./checkpoints/age_check_Jan_13_10_47_second')
from resnet34 import *
ageModel = get_softmax_resnet34_model(20)
ageModel.load_weights('./checkpoints/age_check_resnet_softmax')

random.seed(time.time())
train_times = 1
numSets = 50

directory = 'UTKFace_Edited'
ds_len = 0 
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        ds_len += 1
    else:
        print(filename)
        
        continue

allNums = random.sample(range(ds_len), ds_len)
sets = [0]*numSets
for i in range(numSets):
    sets[i] = sorted(allNums[((ds_len//numSets) * i) : ((ds_len//numSets) * (i+1))])


df_input, df_output = get_softmax_subset(0, 20)

# print(df_input.shape)

# for i in df_output:
#     print(i)
#     break

for count, img in enumerate(df_input):
    if count > 5: 
        break
    plt.imshow(img.astype(np.uint8))
    plt.savefig(f'fig_{count}.jpg')
    image_resized = img.reshape(-1,200,200,3)
    pred = np.around(genderModel.predict(image_resized)[0])
    print(pred)
    age_pred = ageModel.predict([image_resized, pred])
    print(age_pred)
    print(age_pred.argmax())
    print(f'actual: {df_output[count].argmax()}')


# count = 0; tot_error = 0
# for img in df_input:
#     image_resized = img.reshape(-1,200,200,3)
#     pred = np.around(genderModel.predict(image_resized)[0])
#     tot_error += abs(pred - df_output[count])
#     count += 1

# print(f'total error: {tot_error}')
# print(f'total error/count: {tot_error/count}')
# print(f'count: {count}')