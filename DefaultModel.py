#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')
from keras_diagram import ascii

def CNN(Bands,Size,Filepath):
    
    model = Sequential()
        
    model.add(Convolution2D(30, (15, 15), padding='valid', input_shape=(int(Bands), int(Size[0]), int(Size[1])), activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros', name='convolution_1'))
    model.add(Convolution2D(30, (15, 15), activation='relu'))
    model.add(Convolution2D(15, (5, 5), activation='relu'))
    model.add(Convolution2D(15, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(400, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    model.summary()
    Map = ascii(model)
    print('')
    print(Map)
    file = open('Output/'+Filepath+'/Model_Layout.txt','w')
    file.write(Map)
    file.close()
    
    return model
    
