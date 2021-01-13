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
road = np.array(pd.read_csv('/home/jeon/Desktop/cho/noise_map/100/road_prop.csv', header = None, index_col = None), 
                dtype = 'float32')

build_dir = '/home/jeon/Desktop/cho/noise_map/100/building'
build_path = os.path.join(build_dir, '*g')
build_files = sorted(glob.glob(build_path))
build_names = [build_files[i].split('/')[-1] for i in range(len(build_files))]
noise_val = np.array([build_names[i][:2] for i in range(len(build_files))], dtype = 'float32')

wall_dir = '/home/jeon/Desktop/cho/noise_map/100/wall_pool'
wall_path = os.path.join(wall_dir, '*g')
wall_files = sorted(glob.glob(wall_path))

build_dict = {(177+i*3) : (i+1) for i in range(11)}
build_dict[255] = 0

wall_dict = {(147+i*3) : (i+1) for i in range(5)}
wall_dict[0] = 0

#%%
size = 100
train_idx = sorted(random.sample(range(road.shape[0]), road.shape[0] - 3000))
test_idx = sorted(list(set(range(road.shape[0])) - set(train_idx)))

train_noise_val = noise_val[train_idx]; test_noise_val = noise_val[test_idx]
road_train = road[train_idx, ]; road_test = road[test_idx]

temp_build = pd.DataFrame({'v1' : list(build_dict.keys()), 
                           'v2' : list(build_dict.values())}) 
def build_batch(idx, batch_size, build_files):
    img = np.array([cv2.imread(build_files[x])[:, :, 0] for x in idx])
    temp2 = pd.DataFrame({'v1' : img.reshape(-1, )})
    img = np.array(temp2.merge(temp_build, on = ['v1'], how = 'left', sort = False).v2).reshape(batch_size, size, size, 1)
    
    assert img.shape == (batch_size, size, size, 1)
    
    yield np.array(img / max(build_dict.values()), np.float32)


temp_wall = pd.DataFrame({'v1' : list(wall_dict.keys()), 
                           'v2' : list(wall_dict.values())}) 
def wall_batch(idx, batch_size, wall_files):
    img = np.array([cv2.imread(wall_files[x])[:, :, 0] for x in idx])
    temp2 = pd.DataFrame({'v1' : img.reshape(-1, )})
    img = np.array(temp2.merge(temp_wall, on = ['v1'], how = 'left', sort = False).v2).reshape(batch_size, size, size, 1)
    
    assert img.shape == (batch_size, size, size, 1)
    
    yield np.array(img / max(wall_dict.values()), np.float32)
    
#%%
test_build_imgs = []
test_wall_imgs = []

for i in tqdm(test_idx):
    img2 = cv2.imread(build_files[i])[:, :, 0]
    
    temp2 = pd.DataFrame({'v1' : img2.reshape(-1, )})
    img2 = np.array(temp2.merge(temp_build, on = ['v1'], how = 'left', sort = False).v2).reshape(size, size, 1)
    
    assert img2.shape == (size, size, 1)
    
    test_build_imgs.append(img2 / max(build_dict.values()))


for i in tqdm(test_idx):
    img2 = cv2.imread(wall_files[i])[:, :, 0]
    temp2 = pd.DataFrame({'v1' : img2.reshape(-1, )})
    img2 = np.array(temp2.merge(temp_wall, on = ['v1'], how = 'left', sort = False).v2).reshape(size, size, 1)
    
    assert img2.shape == (size, size, 1)
    
    test_wall_imgs.append(img2 / max(wall_dict.values()))

test_build_imgs = np.array(test_build_imgs, np.float32)
test_wall_imgs = np.array(test_wall_imgs, np.float32)

assert test_build_imgs.shape == (3000, size, size, 1)
assert test_wall_imgs.shape == (3000, size, size, 1)

#%%
lin_reg = LinearRegression()
mse = K.losses.MeanSquaredError()

lin_reg.fit(road_train, train_noise_val)
lin_reg.intercept_
lin_reg.coef_

road_pred = lin_reg.predict(road_test)
mse(test_noise_val, road_pred).numpy()

all_res = noise_val - lin_reg.predict(road)
train_res = train_noise_val - lin_reg.predict(road_train)
test_res = test_noise_val - road_pred

#%%
# plt.hist(lin_reg.predict(road_train) - train_noise_val, bins = 50)
pd.Series(lin_reg.predict(road_train) - train_noise_val).hist(bins=50)
pd.Series(road_pred - test_noise_val).hist(bins=50)

plt.hist(all_res, bins = 50)
pd.DataFrame(all_res).describe()

t = np.linspace(min(train_noise_val), max(train_noise_val), 100)
plt.figure(figsize=(8, 8))
plt.scatter(lin_reg.predict(road_train), train_noise_val, alpha=0.5)
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
# import statsmodels.api as sm
# est = sm.OLS(y, X)
# est2 = est.fit()
# est2.summary()

#%%
input1 = layers.Input((size, size, 1))
input2 = layers.Input((size, size, 1))

max1 = layers.MaxPooling2D((2, 2))
# max2 = layers.MaxPooling2D((2, 2))

conv11 = layers.Conv2D(32, (5, 5))
maxpool11 = layers.MaxPooling2D((3, 3), strides=(2, 2))
conv12 = layers.Conv2D(64, (5, 5))
maxpool12 = layers.MaxPooling2D((3, 3))
z1 = maxpool12(conv12(maxpool11(conv11(max1(input1)))))

conv21 = layers.Conv2D(32, (5, 5))
maxpool21 = layers.MaxPooling2D((3, 3), strides=(2, 2))
conv22 = layers.Conv2D(64, (5, 5))
maxpool22 = layers.MaxPooling2D((3, 3))
z2 = maxpool22(conv22(maxpool21(conv21(max1(input2)))))

dense1_1 = layers.Dense(10)
dense1_2 = layers.Dense(10)
dense2 = layers.Dense(1)

flat = layers.Flatten()
#batch_norm = tf.keras.layers.BatchNormalization()

yhat = dense2(tf.concat((dense1_1(flat(z1)), dense1_2(flat(z2))), axis=-1)) 
#yhat = dense2(batch_norm(tf.concat((dense1_1(flat(z1)), dense1_2(flat(z2))), axis=-1)))

model = K.models.Model([input1, input2], yhat)
model.summary()

#%%
batch_size = 3000
optimizer = K.optimizers.Adam(0.1)
mae = K.losses.MeanAbsoluteError()
# huber = K.losses.Huber()
# can_idx = train_idx

for epoch in range(0, 30):
    start = time.time()
    idx = random.sample(can_idx, batch_size)
    can_idx = list(set(can_idx) - set(idx))
    
    b = tf.cast(next(iter(build_batch(idx, batch_size, build_files))), tf.float32)
    w = tf.cast(next(iter(wall_batch(idx, batch_size, wall_files))), tf.float32)
    res = all_res[idx]
    
    with tf.GradientTape(persistent = True) as tape:
        pred = model([b, w])
        loss = mae(res, pred)
        # loss = mse(res, pred)
        # loss = huber(res, pred)
    
    grad = tape.gradient(loss, model.weights)
    optimizer.apply_gradients(zip(grad, model.weights))
    
    print('\nEpoch:', epoch+1, ', TRAIN loss:', loss.numpy(), 'time:', time.time() - start)
    
    if len(can_idx) < batch_size:
        can_idx = train_idx

#%%
res_pred = model.predict([test_build_imgs, test_wall_imgs])
mse(test_res, res_pred).numpy()
# huber(test_res, res_pred).numpy()

#%%
t = np.linspace(np.min(test_res), np.max(test_res), 100)
plt.figure(figsize=(8, 8))
plt.scatter(res_pred, test_res, alpha=0.5)
plt.plot(t, t, color='darkorange', linewidth=3)
plt.xlabel('prediction', fontsize=15)
plt.ylabel('true data', fontsize=15)














