# !pip install opencv-python
import cv2
import os
import glob

import numpy as np
# !pip install pandas
import pandas as pd

# !pip install matplotlib
from matplotlib import pyplot as plt
# !pip install tqdm
from tqdm import tqdm
import random
import time

#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
# print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('=========================================')
print(device_lib.list_local_devices())
tf.debugging.set_log_device_placement(True)

#%%
road_dir = '/home/jeon/Desktop/cho/noise_map/100/road_pool'
road_path = os.path.join(road_dir, '*g')
road_files = sorted(glob.glob(road_path))
road_name = [road_files[i].split('/')[-1][:-4] for i in range(len(road_files))]

build_dir = '/home/jeon/Desktop/cho/noise_map/100/building'
build_path = os.path.join(build_dir, '*g')
build_files = sorted(glob.glob(build_path))

wall_dir = '/home/jeon/Desktop/cho/noise_map/100/wall_pool'
wall_path = os.path.join(wall_dir, '*g')
wall_files = sorted(glob.glob(wall_path))

#%% 
# dictionary 만들기
# 교통량 dictionary
traffic_cat = {1:[0, 33, 66, 99]}
for i in range(1, 11):
    traffic_cat[i+1] = [x+3 for x in traffic_cat[i]]
traffic_dict = dict((v, k) for k in traffic_cat for v in traffic_cat[k])
traffic_dict[255] = 0

# 속력 dictionary
s1 = [x*3 for x in range(11)]
speed_cat = {1:s1}
for i in range(1, 4):
    speed_cat[i+1] = [x+33 for x in speed_cat[i]]
speed_dict = dict((v, k) for k in speed_cat for v in speed_cat[k])
speed_dict[255] = 0

build_dict = {(177+i*3) : (i+1) for i in range(11)}
build_dict[255] = 0

wall_dict = {(147+i*3) : (i+1) for i in range(5)}
wall_dict[0] = 0

#%% 
# train data는 batch를 사용
size = 100
train_idx = random.sample(range(len(road_files)), len(road_files) - 3000)
test_idx = sorted(list(set(range(len(road_files))) - set(train_idx)))
# test_idx = [x for x in range(len(road_files)) if x not in train_idx]
can_idx = train_idx

# 도로 이미지의 교통량에 대한 값을 dictionary값(1~11)으로 변환
temp_traffic = pd.DataFrame({'v1' : list(traffic_dict.keys()), 
                             'v2' : list(traffic_dict.values())}) 
def traffic_batch(idx, batch_size, road_files):
    img = np.array([cv2.imread(road_files[x])[:, :, 0] for x in idx])
    # img2 = np.array(list(map(lambda x:traffic_dict.get(x, 0), 
    #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    
    temp2 = pd.DataFrame({'v1' : img.reshape(-1, )})
    img = np.array(temp2.merge(temp_traffic, on = ['v1'], how = 'left', sort = False).v2).reshape(batch_size, size, size, 1)
    
    assert img.shape == (batch_size, size, size, 1)
    
    yield np.array(img / max(traffic_dict.values()), np.float32)


# 도로 이미지의 속도에 대한 값을 dictionary값(1~4)으로 변환
temp_speed = pd.DataFrame({'v1' : list(speed_dict.keys()), 
                           'v2' : list(speed_dict.values())}) 
def speed_batch(idx, batch_size, road_files):
    img = np.array([cv2.imread(road_files[x])[:, :, 0] for x in idx])
    # img2 = np.array(list(map(lambda x:wall_dict.get(x, 0), 
    #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    
    temp2 = pd.DataFrame({'v1' : img.reshape(-1, )})
    img = np.array(temp2.merge(temp_speed, on = ['v1'], how = 'left', sort = False).v2).reshape(batch_size, size, size, 1)
    
    assert img.shape == (batch_size, size, size, 1)
    
    yield np.array(img / max(speed_dict.values()), np.float32)


# 건물 이미지의 값을 dictionary값(1~11)으로 변환
temp_build = pd.DataFrame({'v1' : list(build_dict.keys()), 
                           'v2' : list(build_dict.values())}) 
def build_batch(idx, batch_size, build_files):
    img = np.array([cv2.imread(build_files[x])[:, :, 0] for x in idx])
    # img2 = np.array(list(map(lambda x:wall_dict.get(x, 0), 
    #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    
    temp2 = pd.DataFrame({'v1' : img.reshape(-1, )})
    img = np.array(temp2.merge(temp_build, on = ['v1'], how = 'left', sort = False).v2).reshape(batch_size, size, size, 1)
    
    assert img.shape == (batch_size, size, size, 1)
    
    yield np.array(img / max(build_dict.values()), np.float32)


# 방음벽 이미지의 값을 dictionary값(1~5)으로 변환
temp_wall = pd.DataFrame({'v1' : list(wall_dict.keys()), 
                           'v2' : list(wall_dict.values())}) 
def wall_batch(idx, batch_size, wall_files):
    img = np.array([cv2.imread(wall_files[x])[:, :, 0] for x in idx])
    # img2 = np.array(list(map(lambda x:wall_dict.get(x, 0), 
    #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    
    temp2 = pd.DataFrame({'v1' : img.reshape(-1, )})
    img = np.array(temp2.merge(temp_wall, on = ['v1'], how = 'left', sort = False).v2).reshape(batch_size, size, size, 1)
    
    assert img.shape == (batch_size, size, size, 1)
    
    yield np.array(img / max(wall_dict.values()), np.float32)

#%% 
# test data는 업로드해서 사용
# test data도 train data와 마찬가지로 이미지의 값을 dictionary에 있는 값으로 대체해서 저장
test_traffic_imgs = []
test_speed_imgs = []
test_build_imgs = []
test_wall_imgs = []

for i in tqdm(test_idx):
    img2 = cv2.imread(road_files[i])[:, :, 0]
    # img2 = np.array(list(map(lambda x:traffic_dict.get(x, 0), 
    #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    
    temp2 = pd.DataFrame({'v1' : img2.reshape(-1, )})
    img2 = np.array(temp2.merge(temp_traffic, on = ['v1'], how = 'left', sort = False).v2).reshape(size, size, 1)
    
    assert img2.shape == (size, size, 1)
    
    test_traffic_imgs.append(img2/ max(traffic_dict.values()))
    

for i in tqdm(test_idx):
    img2 = cv2.imread(road_files[i])[:, :, 0]
    
    temp2 = pd.DataFrame({'v1' : img2.reshape(-1, )})
    img2 = np.array(temp2.merge(temp_speed, on = ['v1'], how = 'left', sort = False).v2).reshape(size, size, 1)
    
    assert img2.shape == (size, size, 1)
    
    test_speed_imgs.append(img2 / max(speed_dict.values()))


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

test_traffic_imgs = np.array(test_traffic_imgs, np.float32)
test_speed_imgs = np.array(test_speed_imgs, np.float32)
test_build_imgs = np.array(test_build_imgs, np.float32)
test_wall_imgs = np.array(test_wall_imgs, np.float32)

assert test_traffic_imgs.shape == (len(test_idx), size, size, 1)
assert test_speed_imgs.shape == (len(test_idx), size, size, 1)
assert test_build_imgs.shape == (len(test_idx), size, size, 1)
assert test_wall_imgs.shape == (len(test_idx), size, size, 1)

test_noise_val = np.array([float(road_name[i][:2]) for i in test_idx])

# build_files[test_idx[10]]
# test_noise_val[10]
# plt.imshow(cv2.imread(build_files[test_idx[i]])[:,:,0])
# plt.imshow(test_build_imgs[i, ...])

# a = cv2.imread(build_files[test_idx[10]])[:,:,0]
# plt.imshow(test_build_imgs[10, ...])

#%% 
batch_size = 3000
distance = tf.cast(np.concatenate((np.tile(np.linspace(-size/2, size/2, size), (size, 1))[..., np.newaxis], 
                                   np.tile(np.linspace(size/2, -size/2, size)[:, np.newaxis], (1, size))[..., np.newaxis]), axis=-1),
                   tf.float32)[tf.newaxis, ...] / 50

distance = tf.tile(distance, (batch_size, 1, 1, 1))
#%%
input1 = layers.Input((size, size, 1))
input2 = layers.Input((size, size, 1))
input3 = layers.Input((size, size, 1))
input4 = layers.Input((size, size, 1))

d1 = tf.multiply(tf.tile(tf.cast(tf.cast(input1, tf.bool), tf.float32), (1, 1, 1, 2)), distance)
d1_ = tf.multiply(tf.tile(tf.cast(tf.cast(input2, tf.bool), tf.float32), (1, 1, 1, 2)), distance)
conv1 = layers.Conv2D(1, (10, 10), activation='sigmoid')
maxpool1 = layers.MaxPooling2D((6, 6))
dense1 = layers.Dense(50, activation='relu')
h1 = maxpool1(conv1(tf.concat((input1, d1, input2, d1_), axis=-1)))
h1 = dense1(layers.Flatten()(h1))

d2 = tf.multiply(tf.tile(tf.cast(tf.cast(input3, tf.bool), tf.float32), (1, 1, 1, 2)), distance)
conv2 = layers.Conv2D(1, (10, 10), activation='sigmoid')
maxpool2 = layers.MaxPooling2D((6, 6))
dense2 = layers.Dense(50, activation='relu')
h2 = maxpool2(conv2(tf.concat((input3, d2), axis=-1)))
h2 = dense2(layers.Flatten()(h2))

d3 = tf.multiply(tf.tile(tf.cast(tf.cast(input4, tf.bool), tf.float32), (1, 1, 1, 2)), distance)
conv3 = layers.Conv2D(1, (10, 10), activation='sigmoid')
maxpool3 = layers.MaxPooling2D((6, 6))
dense3 = layers.Dense(50, activation='relu')
h3 = maxpool3(conv3(tf.concat((input4, d3), axis=-1)))
h3 = dense3(layers.Flatten()(h3))

h = tf.concat((h1, h2, h3), axis=-1)
dense = layers.Dense(1)

yhat = dense(h)

model = K.models.Model([input1, input2, input3, input4], yhat)
model.summary()
#%%
optimizer = K.optimizers.Adam(0.01)
mse = K.losses.MeanSquaredError()
# mae = K.losses.MeanAbsoluteError()
# msle = K.losses.MeanSquaredLogarithmicError()
# epochs = (len(train_idx) // batch_size)

for epoch in range(0, 30):
    # if epoch % 10 == 1:
    #     start = time.time()
    
    start = time.time()
    idx = random.sample(can_idx, batch_size)
    can_idx = list(set(can_idx) - set(idx))
    
    t = tf.cast(next(iter(traffic_batch(idx, batch_size, road_files))), tf.float32)
    s = tf.cast(next(iter(speed_batch(idx, batch_size, road_files))), tf.float32)
    b = tf.cast(next(iter(build_batch(idx, batch_size, build_files))), tf.float32)
    w = tf.cast(next(iter(wall_batch(idx, batch_size, wall_files))), tf.float32)
    noise_val = np.array([float(road_name[i][:2]) for i in idx])
    # print(time.time() - start)
    # with tf.device('/GPU:0'):
    # for i in range(10):
    with tf.GradientTape(persistent=True) as tape:
        pred = model([t, s, b, w])
        loss = mse(noise_val, pred)
    
    grad = tape.gradient(loss, model.weights)
    optimizer.apply_gradients(zip(grad, model.weights))

    # if epoch % 10 == 0:
    print("\nEpoch:", epoch+1, ", TRAIN loss:", loss.numpy(), 'time:', time.time() - start)

    if len(can_idx) < batch_size:
        can_idx = train_idx

 #%%
# test data에 대해 예측값 확인
# y_pred = model.predict([np.tile(test_traffic_imgs, (30, 1, 1, 1)), 
#                         np.tile(test_speed_imgs, (30, 1, 1, 1)), 
#                         np.tile(test_build_imgs, (30, 1, 1, 1)), 
#                         np.tile(test_wall_imgs, (30, 1, 1, 1))])
y_pred = model([test_traffic_imgs, 
                test_speed_imgs, 
                test_build_imgs,  
                test_wall_imgs])

mse(test_noise_val, y_pred).numpy()

#%%
tt = np.linspace(np.min(test_noise_val), np.max(test_noise_val), 100)
plt.figure(figsize=(8, 8))
plt.scatter(y_pred, test_noise_val, alpha=0.5)
plt.plot(tt, tt, color='darkorange', linewidth=3)
plt.xlabel('prediction', fontsize=15)
plt.ylabel('true data', fontsize=15)
#%%
'''weight 확인'''
input1 = test_traffic_imgs
input2 = test_speed_imgs
input3 = test_build_imgs
input4 = test_wall_imgs

d1 = tf.multiply(tf.tile(tf.cast(tf.cast(input1, tf.bool), tf.float32), (1, 1, 1, 2)), distance)
d1_ = tf.multiply(tf.tile(tf.cast(tf.cast(input2, tf.bool), tf.float32), (1, 1, 1, 2)), distance)
h1 = maxpool1(conv1(tf.concat((input1, d1, input2, d1_), axis=-1)))
h1 = dense1(layers.Flatten()(h1))

d2 = tf.multiply(tf.tile(tf.cast(tf.cast(input3, tf.bool), tf.float32), (1, 1, 1, 2)), distance)
h2 = maxpool2(conv2(tf.concat((input3, d2), axis=-1)))
h2 = dense2(layers.Flatten()(h2))

d3 = tf.multiply(tf.tile(tf.cast(tf.cast(input4, tf.bool), tf.float32), (1, 1, 1, 2)), distance)
h3 = maxpool3(conv3(tf.concat((input4, d3), axis=-1)))
h3 = dense3(layers.Flatten()(h3))

h = tf.concat((h1, h2, h3), axis=-1)
y_pred = dense(h)


