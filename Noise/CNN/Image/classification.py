
import cv2
import os
import glob

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import random
import time

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
road_name = [] # list
for i in range(len(road_files)):
    road_name.append(road_files[i].split('/')[-1][:-4])
    # road_name.append(road_files[i].split('\\')[-1][:-4])

build_dir = '/home/jeon/Desktop/cho/noise_map/100/building'
build_path = os.path.join(build_dir, '*g')
build_files = sorted(glob.glob(build_path))
# build_files[0]

wall_dir = '/home/jeon/Desktop/cho/noise_map/100/wall_pool'
wall_path = os.path.join(wall_dir, '*g')
wall_files = sorted(glob.glob(wall_path))
# wall_files[0]

#%%
# def cell_prop(idx, start, width):
#     img = np.array([cv2.imread(road_files[i])[:, :, 0] for i in idx])
#     rcode = [np.unique(x) for x in img]
#     n = width * width
#     p = {x : {rcode[j][i] : (np.sum(img[j, start:(start+width), start:(start+width)] == rcode[j][i]) / n) for i in range(1, len(rcode[j]))} for j,x in enumerate(idx)}
    
#     return p

# road_val = np.array([road_names[i][:2] for i in range(len(road_names))], dtype = 'uint8')
# over50_idx, = np.where(road_val >= 50)

# a = cell_prop(over50_idx, 25, 50)
# prop_idx = [x for x in over50_idx if (len(a[x].values()) > 0) if (max(a[x].values()) >= 0.1)]
# len(prop_idx)
# len(over50_idx)


#%% 
# 교통량 dictionary
# traffic_cat = {1:[0, 33, 66, 99]}
# for i in range(1, 11):
#     traffic_cat[i+1] = [x+3 for x in traffic_cat[i]]

# traffic_dict = dict((v, k) for k in traffic_cat for v in traffic_cat[k])
# traffic_dict[0] = 0

# 속력 dictionary
s1 = [x*3 for x in range(11)]
speed_cat = {1:s1}
for i in range(1, 4):
    speed_cat[i+1] = [x+33 for x in speed_cat[i]]

speed_dict = dict((v, k) for k in speed_cat for v in speed_cat[k])
speed_dict[0] = 0

# 건물 dictionary
# build_dict = {255:0}
build_dict = {255:0}
for i in range(11):
    build_dict[177+i*3] = i+1

# 방음벽 dictionary
# wall_dict = {255:0}
wall_dict = {0:0}
for i in range(5):
    wall_dict[147+i*3] = i+1

#%% 
size = 100
train_idx = random.sample(range(len(road_files)), len(road_files) - 100)
test_idx = [x for x in range(len(road_files)) if x not in train_idx]
# train_idx = random.sample(list(over50_idx), len(over50_idx) - 100)
# test_idx = [x for x in over50_idx if x not in train_idx]
# train_idx = random.sample(prop_idx, len(prop_idx) - 100)
# test_idx = [x for x in prop_idx if x not in train_idx]
can_idx = train_idx

# 도로 이미지의 교통량에 대한 값을 dictionary값(1~11)으로 변환
# temp_traffic = pd.DataFrame({'v1' : list(traffic_dict.keys()), 
#                               'v2' : list(traffic_dict.values())}) 
# def traffic_batch(idx, batch_size, road_files):
#     img = np.array([cv2.imread(road_files[x])[:, :, 0] for x in idx])
#     # img2 = np.array(list(map(lambda x:traffic_dict.get(x, 0), 
#     #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    
#     temp2 = pd.DataFrame({'v1' : img.reshape(-1, )})
#     img = np.array(temp2.merge(temp_traffic, on = ['v1'], how = 'left', sort = False).v2).reshape(batch_size, size, size, 1)
    
#     assert img.shape == (batch_size, size, size, 1)
    
#     yield np.array(img / max(traffic_dict.values()), np.float32)


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
# test_traffic_imgs = []
test_speed_imgs = []
test_build_imgs = []
test_wall_imgs = []

# for i in tqdm(test_idx):
#     img2 = cv2.imread(road_files[i])[:, :, 0]
#     # img2 = np.array(list(map(lambda x:traffic_dict.get(x, 0), 
#     #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    
#     temp2 = pd.DataFrame({'v1' : img2.reshape(-1, )})
#     img2 = np.array(temp2.merge(temp_traffic, on = ['v1'], how = 'left', sort = False).v2).reshape(size, size, 1)
    
#     assert img2.shape == (size, size, 1)
    
#     test_traffic_imgs.append(img2/ max(traffic_dict.values()))
    

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

# test_traffic_imgs = np.array(test_traffic_imgs, np.float32)
test_speed_imgs = np.array(test_speed_imgs, np.float32)
test_build_imgs = np.array(test_build_imgs, np.float32)
test_wall_imgs = np.array(test_wall_imgs, np.float32)

# assert test_traffic_imgs.shape == (100, size, size, 1)
assert test_speed_imgs.shape == (100, size, size, 1)
assert test_build_imgs.shape == (100, size, size, 1)
assert test_wall_imgs.shape == (100, size, size, 1)

test_noise_val = np.array([float(road_name[i][:2]) for i in test_idx]).astype(int)


#%%
noise_unique = np.unique([road_names[i][:2] for i in range(len(road_names))]).astype(int)
# noise_unique = np.unique([road_names[i][:2] for i in over50_idx]).astype(int)
# noise_unique = np.unique([road_names[i][:2] for i in prop_idx]).astype(int)

input1 = layers.Input((size, size, 1))
input2 = layers.Input((size, size, 1))
input3 = layers.Input((size, size, 1))
# input4 = layers.Input((size, size, 1))

conv11 = layers.Conv2D(10, (5, 5), activation='tanh')
maxpool11 = layers.MaxPooling2D((3, 3))
conv12 = layers.Conv2D(10, (5, 5), activation='tanh')
maxpool12 = layers.MaxPooling2D((3, 3))
z1 = maxpool12(conv12(maxpool11(conv11(input1))))

conv21 = layers.Conv2D(10, (5, 5), activation='tanh')
maxpool21 = layers.MaxPooling2D((3, 3))
conv22 = layers.Conv2D(10, (5, 5), activation='tanh')
maxpool22 = layers.MaxPooling2D((3, 3))
z2 = maxpool22(conv22(maxpool21(conv21(input2))))

conv31 = layers.Conv2D(10, (5, 5), activation='tanh')
maxpool31 = layers.MaxPooling2D((3, 3))
conv32 = layers.Conv2D(10, (5, 5), activation='tanh')
maxpool32 = layers.MaxPooling2D((3, 3))
z3 = maxpool32(conv32(maxpool31(conv31(input3))))


dense1 = layers.Dense(5, activation = 'elu')
dense2 = layers.Dense(max(noise_unique) - min(noise_unique) + 1, activation='softmax')

yhat = dense2(dense1(tf.concat((tf.reshape(z1, (-1, tf.math.reduce_prod(z1.shape[1:]).numpy())), 
                                tf.reshape(z2, (-1, tf.math.reduce_prod(z1.shape[1:]).numpy())), 
                                tf.reshape(z3, (-1, tf.math.reduce_prod(z1.shape[1:]).numpy()))), axis=-1)))

# model = K.models.Model([input1, input2, input3, input4], yhat)
model = K.models.Model([input1, input2, input3], yhat)
model.summary()


#%%
# max1 = layers.MaxPooling2D((3, 3))

# conv21 = layers.Conv2D(10, (5, 5), activation='relu')
# maxpool21 = layers.MaxPooling2D((3, 3), strides=(2, 2))
# conv22 = layers.Conv2D(10, (5, 5), activation='relu')
# maxpool22 = layers.MaxPooling2D((3, 3), strides = (1, 1))
# z2 = maxpool22(conv22(maxpool21(conv21(max1(input2)))))

# conv31 = layers.Conv2D(10, (5, 5), activation='relu')
# maxpool31 = layers.MaxPooling2D((3, 3), strides=(2, 2))
# conv32 = layers.Conv2D(10, (5, 5), activation='relu')
# maxpool32 = layers.MaxPooling2D((3, 3), strides=(1, 1))
# z3 = maxpool32(conv32(maxpool31(conv31(max1(input3)))))

# conv41 = layers.Conv2D(10, (5, 5), activation='relu')
# maxpool41 = layers.MaxPooling2D((3, 3), strides=(2, 2))
# conv42 = layers.Conv2D(10, (5, 5), activation='relu')
# maxpool42 = layers.MaxPooling2D((3, 3), strides = (1, 1))
# z4 = maxpool42(conv42(maxpool41(conv41(max1(input4)))))

# dense1_1 = layers.Dense(5, activation='elu')
# dense1_2 = layers.Dense(5, activation='elu')
# dense1_3 = layers.Dense(5, activation='elu')
# dense3 = 2

# yhat = dense3(tf.concat((dense1_1(tf.reshape(z2, (-1, tf.math.reduce_prod(z2.shape[1:]).numpy()))), 
#                          dense1_2(tf.reshape(z3, (-1, tf.math.reduce_prod(z3.shape[1:]).numpy()))), 
#                          dense1_3(tf.reshape(z4, (-1, tf.math.reduce_prod(z4.shape[1:]).numpy())))), axis=-1))

# model = K.models.Model([input2, input3, input4], yhat)
# model.summary()
#%%
batch_size = 4000
optimizer = K.optimizers.Adam(0.005)
scc = K.losses.SparseCategoricalCrossentropy()
# over50 = np.unique(road_val[over50_idx]) - 50

for epoch in range(0, 50):
    # if epoch % 10 == 1:
    #     start = time.time()
    
    start = time.time()
    idx = random.sample(can_idx, batch_size)
    can_idx = list(set(can_idx) - set(idx))
    # t = tf.cast(next(iter(traffic_batch(idx, batch_size, road_files))), tf.float32)
    s = tf.cast(next(iter(speed_batch(idx, batch_size, road_files))), tf.float32)
    b = tf.cast(next(iter(build_batch(idx, batch_size, build_files))), tf.float32)
    w = tf.cast(next(iter(wall_batch(idx, batch_size, wall_files))), tf.float32)
    noise_val = np.array([float(road_name[i][:2]) for i in idx]).astype(int)
    
    # print(time.time() - start)
    # with tf.device('/GPU:0'):
    # for i in range(10):
    
    with tf.GradientTape(persistent=True) as tape:
        # pred = model([t, s, b, w])
        pred = model([s, b, w])
        loss = scc((noise_val-min(noise_unique)).tolist(), pred)
        # loss = scc((noise_val-50).tolist(), pred)

    grad = tape.gradient(loss, model.weights)
    optimizer.apply_gradients(zip(grad, model.weights))
    
    #np.unique(np.argmax(pred, -1), return_counts=True)
    # if epoch % 10 == 0:
    
    print("\nEpoch:", epoch+1, ", TRAIN loss:", loss.numpy(), 'time:', time.time() - start)

    if len(can_idx) < batch_size:
        can_idx = train_idx

# classification = K.models.clone_model(model)
# classification.set_weights(model.get_weights())
#%%
# y_pred = model.predict([test_speed_imgs, test_build_imgs, test_wall_imgs])
y_pred = classification.predict([test_speed_imgs, test_build_imgs, test_wall_imgs])
scc((test_noise_val-min(noise_unique)).tolist(), tf.constant(y_pred)).numpy()

#%%
# test
t = noise_unique - min(noise_unique)
plt.figure(figsize=(8, 8))
plt.scatter(np.argmax(y_pred, -1), test_noise_val- min(noise_unique), alpha=0.5)
plt.plot(t, t, color='darkorange', linewidth=3)
plt.xlabel('prediction', fontsize=15)
plt.ylabel('true data', fontsize=15)

#%%
# train
t = noise_unique - min(noise_unique)
plt.figure(figsize=(8, 8))
plt.scatter(np.argmax(pred, -1), noise_val-min(noise_unique), alpha=0.5)
plt.plot(t, t, color='darkorange', linewidth=3)
plt.xlabel('prediction', fontsize=15)
plt.ylabel('true data', fontsize=15)

