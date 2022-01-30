""" imports to run OpenCV """

import cv2
import numpy as np
import dlib 

from FPS import *

""" imports to predict from tensorflow model"""

# import matplotlib.pyplot as plt
# from PIL import Image

# from numpy import asarray

# import tensorflow as tf
# import os
# import pandas as pd
# import random
# import time

# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Add, Activation, RandomFlip, Input, RandomRotation, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense

# from sklearn.model_selection import train_test_split
# from tensorflow.keras.regularizers import l2

from gender_model import *

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

fps = FPS_s().start()

ret, frame = cap.read()

black_background = np.zeros(frame.shape, dtype='uint8')

genModel = gender_model()

while True:
    fps.update()
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    num_faces = len(faces)
    if num_faces != 0:
        cropped_faces = []
        for face in faces:
            x, y, x1, y1 = face.left(), face.top(), face.right(), face.bottom()

            cropped_face = frame[y:y1, x:x1]
            cropped_face = cv2.resize(cropped_face, ((700)//num_faces,(700)//num_faces), interpolation=cv2.INTER_AREA)

            cropped_faces.append(cropped_face)



        Horizontal_Concat = np.concatenate(cropped_faces, axis=1)

        x_offset = (frame.shape[1] - Horizontal_Concat.shape[1]) // 2
        y_offset = (frame.shape[0] - Horizontal_Concat.shape[0]) // 2

        black_background[y_offset: y_offset + Horizontal_Concat.shape[0], 
                        x_offset: x_offset + Horizontal_Concat.shape[1]] = Horizontal_Concat

        cv2.imshow('FaceTracking', black_background)

        black_background[y_offset: y_offset + Horizontal_Concat.shape[0], 
                        x_offset: x_offset + Horizontal_Concat.shape[1]][:] = 0
    else:
        cv2.putText(frame, 'No Faces Found', (0,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.imshow('FaceTracking', frame)

    # i = 0
    # for face in faces:
    #     x, y = face.left(), face.top()
    #     x1, y1 = face.right(), face.bottom()
    #     cv2.rectangle(frame, (x,y), (x1,y1), (0,255,0), 2)

    #     i +=1 
    #     cv2.putText(frame, 'face num' + str(i), 
    #         (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)



    # cv2.imshow('frame', frame)
    # cv2.imwrite('frame.jpg', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        fps.stop()
        break

print('Average Frames per Second: ',fps.fps())

cap.release()
cv2.destroyAllWindows()