import cv2
import os
import glob
import random 
import time

from tqdm import tqdm

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
# !pip install sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
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
road_names = [road_files[i].split('/')[-1] for i in range(len(road_files))]

all_idx = [i for i in range(len(road_files)) if int(road_names[i][:2]) >= 0]
# over50_idx = [i for i in range(len(road_files)) if int(road_names[i][:2]) >= 50]

# 줄여보기
start = 45
width = 10
# start2 = 40
# width2 = 20

u = []
def cell_prop1(idx, start, width):
    img = np.array([cv2.imread(road_files[i])[:, :, 0] for i in idx])
    rcode = [np.unique(x)[:-1] for x in img]
    n = width * width
    u.append(rcode)
    p = {idx[j] : {rcode[j][i] : round((np.sum(img[j, start:(start+width), start:(start+width)] == rcode[j][i]) / n), 5)
                   for i in range(len(rcode[j]))} 
         for j in range(img.shape[0])}
    
    return p

prop10 = cell_prop1(all_idx, start, width)
rcode_unique = np.unique(np.hstack(np.hstack(u)))
len(rcode_unique)

def cell_prop2(idx, start1, width1, start2, width2):
    img = np.array([cv2.imread(road_files[i])[:, :, 0] for i in idx])
    rcode = [np.unique(x)[:-1] for x in img]
    n = width1*width1 - width2*width2
    p = {idx[j] : {rcode[j][i] : (round((np.sum(img[j, start1:(start1+width1), start1:(start1+width1)] == rcode[j][i]) - 
                                    np.sum(img[j, start2:(start2+width2), start2:(start2+width2)] == rcode[j][i])) / n, 5)) 
                    for i in range(len(rcode[j]))} 
          for j in range(img.shape[0])}
    
    return p

prop20 = cell_prop2(all_idx, 40, 20, start, width)
prop30 = cell_prop2(all_idx, 35, 30, 40, 20)
prop40 = cell_prop2(all_idx, 30, 40, 35, 30)
prop50 = cell_prop2(all_idx, 25, 50, 30, 40)
prop60 = cell_prop2(all_idx, 20, 60, 25, 50)
prop70 = cell_prop2(all_idx, 15, 70, 20, 60)
prop80 = cell_prop2(all_idx, 10, 80, 15, 70)
prop90 = cell_prop2(all_idx, 5, 90, 10, 80)
prop100 = cell_prop2(all_idx, 0, 100, 5, 90)
 
# prop_idx = [x for x in over50_idx if (len(prop[x].values()) > 0) if (max(prop[x].values()) >= 0.1)]
# len(prop_idx)

#%%
train_idx = random.sample(all_idx, len(all_idx) - 3000)
test_idx = sorted(set(all_idx) - set(train_idx))

y_train = np.array([road_names[i][:2] for i in train_idx], dtype = 'float32')
y_test = np.array([road_names[i][:2] for i in test_idx], dtype = 'float32')
y_int = np.array([int(road_names[i][:2]) for i in range(len(road_files))], dtype = 'int8')

dictvectorizer = DictVectorizer(sparse = False)
x10 = dictvectorizer.fit_transform([prop10[i] for i in all_idx])
x20 = dictvectorizer.fit_transform(prop20[i] for i in all_idx)
x30 = dictvectorizer.fit_transform(prop30[i] for i in all_idx)
x40 = dictvectorizer.fit_transform(prop40[i] for i in all_idx)
x50 = dictvectorizer.fit_transform(prop50[i] for i in all_idx)
x60 = dictvectorizer.fit_transform(prop60[i] for i in all_idx)
x70 = dictvectorizer.fit_transform(prop70[i] for i in all_idx)
x80 = dictvectorizer.fit_transform(prop80[i] for i in all_idx)
x90 = dictvectorizer.fit_transform(prop90[i] for i in all_idx)
x100 = dictvectorizer.fit_transform(prop100[i] for i in all_idx)
# x_etc = dictvectorizer.fit_transform(prop_etc[i] for i in all_idx)

dictvectorizer.get_feature_names()
len(dictvectorizer.get_feature_names())
all(rcode_unique == dictvectorizer.get_feature_names())

x10_train = x10[train_idx, ]
x10_test = x10[test_idx, ]

x20_train = x20[train_idx,]
x20_test = x20[test_idx, ]

x30_train = x30[train_idx,]
x30_test = x30[test_idx, ]

x40_train = x40[train_idx,]
x40_test = x40[test_idx, ]

x50_train = x50[train_idx,]
x50_test = x50[test_idx, ]

x60_train = x60[train_idx,]
x60_test = x60[test_idx, ]

x70_train = x70[train_idx,]
x70_test = x70[test_idx, ]

x80_train = x80[train_idx,]
x80_test = x80[test_idx, ]

x90_train = x90[train_idx,]
x90_test = x90[test_idx, ]

x100_train = x100[train_idx,]
x100_test = x100[test_idx, ]

# x = np.concatenate((x10, x20, x30, x40, x50, x60, x70, x80, x90, x100), axis = 1)
# pd.DataFrame(x).to_csv('/home/jeon/Desktop/cho/noise_map/100/road_prop.csv')

x_train = np.concatenate((x10_train, x20_train, x30_train, x40_train, x50_train, 
                          x60_train, x70_train, x80_train, x90_train, x100_train), axis = 1)
x_test = np.concatenate((x10_test, x20_test, x30_test, x40_test, x50_test, 
                         x60_test, x70_test, x80_test, x90_test, x100_test), axis = 1)


#%%
mse = K.losses.MeanSquaredError()
mae = K.losses.MeanAbsoluteError()

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

beta0 = lin_reg.intercept_
beta = lin_reg.coef_

y_pred = lin_reg.predict(x_test)

# dir(lin_reg)
mse(y_test, y_pred).numpy()

#%%
# from keras import models, layers

# input1 = tf.keras.Input((x_train.shape[1], ))
# def build_model(x_train):
#     model = K.models.Sequential()
#     model.add(layers.Dense(128, input_shape=(x_train.shape[1], )))
#     model.add(layers.Dense(64))
#     model.add(layers.Dense(32))
#     model.add(layers.Dense(16))
#     model.add(layers.Dense(1))
#     model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
#     return model

# model = build_model(input1)
# model.fit(x_train, y_train, epochs = 100, batch_size = x_train.shape[0])

# y_pred = model.predict(x_test)
# y_pred = model.predict(x_train)

# mae(y_test, y_pred).numpy()
#%%
# 예측값과 실제값 차이의 histogram
pd.Series(y_train - lin_reg.predict(x_train)).hist(bins=50)
pd.Series(y_test - y_pred).hist(bins=50)
#%%
# train
t = np.linspace(np.min(y_train), np.max(y_train), 100)
plt.figure(figsize=(8, 8))
plt.scatter(y_pred, y_train, alpha=0.5)
plt.plot(t, t, color='darkorange', linewidth=3)
plt.xlabel('prediction', fontsize=15)
plt.ylabel('true data', fontsize=15)

#%%
# test
t = np.linspace(np.min(y_test), np.max(y_test), 100)
plt.figure(figsize=(8, 8))
plt.scatter(y_pred, y_test, alpha=0.5)
plt.plot(t, t, color='darkorange', linewidth=3)
plt.xlabel('prediction', fontsize=15)
plt.ylabel('true data', fontsize=15)

