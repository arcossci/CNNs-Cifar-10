#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:05:50 2017

@author: jefferson
CNN for CIFAR10 dataset
tensorflow with CPU mode
"""

#import dataset and tools
import numpy as np
import pandas as pd
from keras.datasets import cifar10   
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from matplotlib import pyplot as plt

# fix seed for reproducibility
seed = 1
np.random.seed(seed)

#load data
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data() 

# normalize from 0 to 1 values
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / (255.0)
X_test = X_test / (255.0)

#define constants
n_train, depth, height, width = X_train.shape     
n_classes = np.unique(Y_train).shape[0]     # 10 classes
k_s = 3    # kernel size 3 x 3
p_s = 2    # pool size 2 x 2
d_p = 0.2     # drop out probability
f_c_2 = 1024    # 1024 fully connected layers
f_c_1 = 512     # 512 fully connected layers
conv_1 = 32     # 1 convolution 32 feature maps
conv_2 = 64     # 2 convolution 64 feature maps
conv_3 = 128    # 3 convolution 128 feature maps
epochs = 1      # number times to iterate
lrate = 0.01
decay = lrate/epochs 
b_s = 64    # 64 example training for iteration

# One hot encode labels
Y_train = np_utils.to_categorical(Y_train, n_classes)
Y_test = np_utils.to_categorical(Y_test, n_classes) 

# Secuential model definition
model = Sequential()
model.add(Convolution2D(conv_1, k_s, k_s, input_shape=(depth, height, width), activation='relu', border_mode='same'))
model.add(Dropout( d_p ))
model.add(Convolution2D( conv_1, k_s, k_s, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=( p_s, p_s )))
# 2 convolution
model.add(Convolution2D( conv_2, k_s, k_s, activation='relu', border_mode='same'))
model.add(Dropout( d_p ))
model.add(Convolution2D( conv_2, k_s, k_s, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=( p_s, p_s )))

# 3 convolution
model.add(Convolution2D( conv_3, k_s, k_s, activation='relu', border_mode='same'))
model.add(Dropout( d_p ))
model.add(Convolution2D(conv_3, k_s, k_s, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=( p_s, p_s )))
model.add(Flatten())
model.add(Dropout( d_p ))
# 1024 fully conectted layers
model.add(Dense( f_c_2, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout( d_p ))

# 512 fully conectted layers
model.add(Dense( f_c_1, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout( d_p ))
model.add(Dense( n_classes, activation='softmax' ))

# compile model

sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)   # optimizer
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

 print(model.summary())

# fit and evaluate model
hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=b_s, nb_epoch=epochs)
score = model.evaluate(X_test, Y_test, verbose=0)
# accuracy
print("Accuracy: %.2f%%" % (score[1]*100))

# save model
model.save('my_model.h5')

# architecture net
print(model.summary())
print(model.layers[0].get_weights()[0].shape)
print(model.layers[14].get_weights()[0].shape) #layers


# confision matrix
(x_tr, y_tr), (x_te, y_te) = cifar10.load_data() 
y_hat = model.predict_classes(X_test)
b2 = y_te.ravel()   #convert data to 1dD
pd.crosstab(y_hat, b2)

# visualization
#print(history.history.keys())
# summarize history for accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




















