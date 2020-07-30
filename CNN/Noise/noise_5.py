# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 22:49:08 2020

@author: SOYOUNG
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import keras

from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt

#%% read data, split train & test data
noise_5 = pd.read_csv('C:/Users/SOYOUNG/Desktop/final_5by5_200721.csv')
noise_5.head()
noise_5.shape

x = np.array(noise_5.iloc[:, 2:])
x.shape[1] / 13

y = noise_5.iloc[:, 1][:, np.newaxis]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, shuffle = True, random_state = 123)
x_train.shape, y_train.shape

# x_train.columns[52]
# a = np.array(x_train.loc[:, 'Population_Density.4':'Green.4'])
# a = tf.reshape(a, shape = [a.shape[0], 1, 1, 13])

#%% tensor
# a = np.array(x)
# b = tf.reshape(a, shape = [a.shape[0], 3, 3, 13])

train_tensor = tf.reshape(x_train, shape = [x_train.shape[0], 5, 5, 13])
train_tensor.shape

test_tensor = tf.reshape(x_test, shape = [x_test.shape[0], 5, 5, 13])
test_tensor.shape

#%% CNN
# model_5 = models.Sequential()
# model_5.add(layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same', input_shape = (5, 5, 13)))
# model_5.add(layers.Dropout(0.3))
# model_5.add(layers.Conv2D(32, (2, 2), activation = 'relu', padding = "valid", ))
# model_5.add(layers.Dropout(0.3))
# model_5.add(layers.Conv2D(16, (2, 2), activation = 'relu'))
# model_5.add(layers.MaxPooling2D((2, 2)))
# model_5.add(layers.Flatten())
# model_5.add(layers.Dense(128, activation = "relu"))
# model_5.add(layers.Dropout(0.3))
# model_5.add(layers.Dense(32, activation = "relu"))
# model_5.add(layers.Dropout(0.3))
# model_5.add(layers.Dense(1))


# model_5.summary()

# adam = Adam(lr = 0.001)
# model_5.compile(optimizer = adam, loss = 'mae', metrics = ['mse'])
# model_5.fit(train_tensor, y_train, epochs = 1000, batch_size = 100,  validation_data = (test_tensor, y_test))

y_hat_train = model_5.predict(train_tensor).flatten()


plt.scatter(y_hat_train, y_train, alpha = 0.1)
plt.xlabel(r'$\hat{y}$')
plt.ylabel('y')

# model_5.save(r"C:\Users\SOYOUNG\Desktop\CNN models\cnn_model_5.h5")
# model_5= tf.keras.models.load_model(r"C:\Users\SOYOUNG\Desktop\CNN models\cnn_model_5.h5")

#%%
pred_y = model_5.predict(test_tensor).flatten()
plt.scatter(y_test, pred_y, alpha = 0.1)

#%%
train_noise = pd.DataFrame({'Noise_Level' : y_train.flatten(), 'pred_noise' : y_hat_train})
test_noise = pd.DataFrame({'Noise_Level' : y_test.flatten(), 'pred_noise' : pred_y})
noise = pd.concat([train_noise, test_noise])

pd.merge(noise_5, noise, on = 'Noise_Level').to_csv('C:/Users/SOYOUNG/Desktop/noise_5by5.csv', header = True, index = False, encoding = 'utf-8')


#%%
# test_model = models.Sequential()
# test_model.add(layers.Conv2D(16, (3, 3), activation = 'relu', padding = 'same', input_shape = (5, 5, 13)))
# test_model.add(layers.Dropout(0.3))
# test_model.add(layers.Conv2D(4, (3, 3), activation = 'relu'))
# test_model.add(layers.Flatten())
# test_model.add(layers.Dense(1))

# test_model.summary()

# adam = Adam(lr = 0.00001)
# test_model.compile(optimizer = adam, loss = 'mae', metrics = ['mse'])
# test_model.fit(train_tensor, y_train, epochs = 500, batch_size = 100, validation_data = (test_tensor, y_test))

# pred_y = test_model.predict(train_tensor).flatten()

# plt.scatter(y_train, pred_y, alpha = 0.2)
