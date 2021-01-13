import cv2
import os
import glob
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.linear_model import LinearRegression

#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
from tensorflow.python.client import device_lib
print('=========================================')
print(device_lib.list_local_devices())
tf.debugging.set_log_device_placement(True)

#%%
road_dir = '/home/jeon/Desktop/cho/noise_map/100/road_pool'
road_path = os.path.join(road_dir, '*g')
road_files = sorted(glob.glob(road_path))
road_names = [road_files[i].split('/')[-1][:-4] for i in range(len(road_files))]
noise_val = np.array([road_names[i][:2] for i in range(len(road_files))], dtype = 'float32')


road = np.array(pd.read_csv('/home/jeon/Desktop/cho/noise_map/100/road_prop.csv', header = None, index_col = None), 
                dtype = 'float32')
build= np.array(pd.read_csv('/home/jeon/Desktop/cho/noise_map/100/build_prop.csv', header = None, index_col = None), 
                dtype = 'float32')
wall = np.array(pd.read_csv('/home/jeon/Desktop/cho/noise_map/100/wall_prop.csv', header = None, index_col = None), 
                dtype = 'float32')
rb = np.concatenate((road, build), axis = 1)


train_idx = sorted(random.sample(range(road.shape[0]), road.shape[0] - 3000))
test_idx = sorted(list(set(range(road.shape[0])) - set(train_idx)))

train_noise_val = noise_val[train_idx]; test_noise_val = noise_val[test_idx]
road_train = road[train_idx, ]; road_test = road[test_idx, ]
rb_train = rb[train_idx, ]; rb_test = rb[test_idx, ]

#%%
lin_reg1 = LinearRegression()
mse = K.losses.MeanSquaredError()

lin_reg1.fit(road_train, train_noise_val)
lin_reg1.intercept_
lin_reg1.coef_

road_pred = lin_reg1.predict(road_test)
mse(test_noise_val, road_pred).numpy()

#%%
# plt.hist(lin_reg.predict(road_train) - train_noise_val, bins = 50)
pd.Series(lin_reg1.predict(road_train) - train_noise_val).hist(bins=50)
pd.Series(road_pred - test_noise_val).hist(bins=50)

t = np.linspace(min(train_noise_val), max(train_noise_val), 100)
plt.figure(figsize=(8, 8))
plt.scatter(lin_reg1.predict(road_train), train_noise_val, alpha=0.5)
plt.plot(t, t, color='darkorange', linewidth=3)
plt.xlabel('prediction', fontsize=15)
plt.ylabel('true data', fontsize=15)

t = np.linspace(min(test_noise_val), max(test_noise_val), 100)
plt.figure(figsize=(8, 8))
plt.scatter(road_pred, test_noise_val, alpha=0.5)
plt.plot(t, t, color='darkorange', linewidth=3)
plt.xlabel('prediction', fontsize=15)
plt.ylabel('true data', fontsize=15)

#%%
lin_reg2 = LinearRegression()

lin_reg2.fit(rb_train, train_noise_val)
lin_reg2.intercept_
lin_reg2.coef_

rb_pred = lin_reg2.predict(rb_test)
mse(test_noise_val, rb_pred).numpy()

#%%
# plt.hist(lin_reg.predict(road_train) - train_noise_val, bins = 50)
pd.Series(lin_reg2.predict(rb_train) - train_noise_val).hist(bins=50)
pd.Series(rb_pred - test_noise_val).hist(bins=50)

t = np.linspace(min(train_noise_val), max(train_noise_val), 100)
plt.figure(figsize=(8, 8))
plt.scatter(lin_reg2.predict(rb_train), train_noise_val, alpha=0.5)
plt.plot(t, t, color='darkorange', linewidth=3)
plt.xlabel('prediction', fontsize=15)
plt.ylabel('true data', fontsize=15)

t = np.linspace(min(test_noise_val), max(test_noise_val), 100)
plt.figure(figsize=(8, 8))
plt.scatter(rb_pred, test_noise_val, alpha=0.5)
plt.plot(t, t, color='darkorange', linewidth=3)
plt.xlabel('prediction', fontsize=15)
plt.ylabel('true data', fontsize=15)
