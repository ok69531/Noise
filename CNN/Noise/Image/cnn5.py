### batch사용 안하고 데이터 메모리에 올려서,,, >>> 터짐,,,

#%%
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
# road_dir = r''
road_dir = ''
road_path = os.path.join(road_dir, '*g')
road_files = glob.glob(road_path)
road_name = [] # list
for i in range(len(road_files)):
    road_name.append(road_files[i].split('/')[-1][:-4])
    # road_name.append(road_files[i].split('\\')[-1][:-4])
y = np.array([x[:2] for x in road_name])


# build_dir = r''
build_dir = ''
build_path = os.path.join(build_dir, '*g')
build_files = glob.glob(build_path)


# wall_dir = r''
wall_dir = ''
wall_path = os.path.join(wall_dir, '*g')
wall_files = glob.glob(wall_path)

    
#%% dictionary 만들기
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

# 건물 dict
build_dict = {255:0}
for i in range(11):
    build_dict[177+i*3] = i+1


# 방음벽 dict
wall_dict = {255:0}
for i in range(5):
    wall_dict[147+i*3] = i+1

#%%
# ordinal embedding을 위한 onehot vec 만들기
traffic_hot_dict = {}
for i in range(11):
    z = np.zeros(11)
    z[:i+1] = 1
    traffic_hot_dict[i+1] = z

speed_hot_dict = {}
for i in range(4):
    z = np.zeros(4)
    z[:i+1] = 1
    speed_hot_dict[i+1] = z

wall_hot_dict = {}
for i in range(5):
    z = np.zeros(5)
    z[:i+1] = 1
    wall_hot_dict[i+1] = z

#%%
train_idx = random.sample(range(len(road_files)), 5000)
test_idx = [x for x in range(len(road_files)) if x not in train_idx]
    
traffic_imgs = []
speed_imgs = []
build_imgs = []
wall_imgs = []

temp_traffic = pd.DataFrame({'v1' : list(traffic_dict.keys()), 
                             'v2' : list(traffic_dict.values())}) # 밖으로 빼기
for i in tqdm(range(len(train_idx))):
    img = cv2.imread(road_files[i])[:, :, 0]
    # img2 = np.array(list(map(lambda x:traffic_dict.get(x, 0), 
    #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    
    temp2 = pd.DataFrame({'v1' : img.reshape(-1, )})
    img = np.array(temp2.merge(temp_traffic, on = ['v1'], how = 'left', sort = False).v2).reshape(500, 500, 1)
    
    assert img.shape == (500, 500, 1)
    
    traffic_imgs.append(np.array(img / max(traffic_dict.values()), np.float32))
    
    del img
    del temp2
    
temp_speed = pd.DataFrame({'v1' : list(speed_dict.keys()), 
                           'v2' : list(speed_dict.values())}) # 밖으로 빼기
for i in tqdm(range(len(train_idx))):
    img = cv2.imread(road_files[i])[:, :, 0]
    # img2 = np.array(list(map(lambda x:wall_dict.get(x, 0), 
    #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    
    temp2 = pd.DataFrame({'v1' : img.reshape(-1, )})
    img2 = np.array(temp2.merge(temp_speed, on = ['v1'], how = 'left', sort = False).v2).reshape(500, 500, 1)
    
    assert img2.shape == (500, 500, 1)
    
    speed_imgs.append(img2 / max(speed_dict.values()))
    
    del img
    del temp2
    
    speed_imgs.append(img2 / max(speed_dict.values()))

temp_build = pd.DataFrame({'v1' : list(build_dict.keys()), 
                           'v2' : list(build_dict.values())}) # 밖으로 빼기
for i in tqdm(range(len(train_idx))):
    img = cv2.imread(road_files[i])[:, :, 0]
    # img2 = np.array(list(map(lambda x:wall_dict.get(x, 0), 
    #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    
    temp2 = pd.DataFrame({'v1' : img.reshape(-1, )})
    img2 = np.array(temp2.merge(temp_build, on = ['v1'], how = 'left', sort = False).v2).reshape(500, 500, 1)
    
    assert img2.shape == (500, 500, 1)
    
    build_imgs.append(img2 / max(build_dict.values()))
    
    del img
    del temp2

temp_wall = pd.DataFrame({'v1' : list(wall_dict.keys()), 
                           'v2' : list(wall_dict.values())}) # 밖으로 빼기
for i in tqdm(range(len(train_idx))):
    img = cv2.imread(wall_files[i])[:, :, 0]
    # img2 = np.array(list(map(lambda x:wall_dict.get(x, 0), 
    #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    
    temp2 = pd.DataFrame({'v1' : img.reshape(-1, )})
    img2 = np.array(temp2.merge(temp_wall, on = ['v1'], how = 'left', sort = False).v2).reshape(500, 500, 1)
    
    assert img2.shape == (500, 500, 1)
    
    wall_imgs.append(img2 / max(wall_dict.values()))
    
    del img
    del temp2

traffic_imgs = np.array(traffic_imgs)
speed_imgs = np.array(speed_imgs)
build_imgs = np.array(build_imgs)
wall_imgs = np.array(wall_imgs)

assert traffic_imgs == (5000, 500, 500, 1)
assert speed_imgs == (5000, 500, 500, 1)
assert build_imgs == (5000, 500, 500, 1)
assert wall_imgs == (5000, 500, 500, 1)

noise_val = np.array([float(road_name[i][:2]) for i in train_idx])
    
#%%
test_traffic_imgs = []
test_speed_imgs = []
test_build_imgs = []
test_wall_imgs = []

for i in tqdm(range(len(test_idx))):
    img = cv2.imread(road_files[i])[:, :, 0]
    # img2 = np.array(list(map(lambda x:traffic_dict.get(x, 0), 
    #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    
    temp2 = pd.DataFrame({'v1' : img.reshape(-1, )})
    img2 = np.array(temp2.merge(temp_traffic, on = ['v1'], how = 'left', sort = False).v2).reshape(500, 500, 1)
    
    assert img2.shape == (500, 500, 1)
    
    test_traffic_imgs.append(img2 / max(traffic_dict.values()))
    
    del img
    del temp2
    
for i in tqdm(range(len(test_idx))):
    img = cv2.imread(road_files[i])[:, :, 0]
    # img2 = np.array(list(map(lambda x:wall_dict.get(x, 0), 
    #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    
    temp2 = pd.DataFrame({'v1' : img.reshape(-1, )})
    img2 = np.array(temp2.merge(temp_speed, on = ['v1'], how = 'left', sort = False).v2).reshape(500, 500, 1)
    
    assert img2.shape == (500, 500, 1)
    
    test_speed_imgs.append(img2 / max(speed_dict.values()))
    
    del img
    del temp2
    
for i in tqdm(range(len(test_idx))):
    img = cv2.imread(road_files[i])[:, :, 0]
    # img2 = np.array(list(map(lambda x:wall_dict.get(x, 0), 
    #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    
    temp2 = pd.DataFrame({'v1' : img.reshape(-1, )})
    img2 = np.array(temp2.merge(temp_build, on = ['v1'], how = 'left', sort = False).v2).reshape(500, 500, 1)
    
    assert img2.shape == (500, 500, 1)
    
    test_build_imgs.append(img2 / max(build_dict.values()))
    
    del img
    del temp2

for i in tqdm(range(len(test_idx))):
    img = cv2.imread(wall_files[i])[:, :, 0]
    # img2 = np.array(list(map(lambda x:wall_dict.get(x, 0), 
    #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    
    temp2 = pd.DataFrame({'v1' : img.reshape(-1, )})
    img2 = np.array(temp2.merge(temp_wall, on = ['v1'], how = 'left', sort = False).v2).reshape(500, 500, 1)
    
    assert img2.shape == (500, 500, 1)
    
    test_wall_imgs.append(img2 / max(wall_dict.values()))
    
    del img
    del temp2

test_traffic_imgs = np.array(test_traffic_imgs)
test_speed_imgs = np.array(test_speed_imgs)
test_build_imgs = np.array(test_build_imgs)
test_wall_imgs = np.array(test_wall_imgs)

assert test_traffic_imgs == (100, 500, 500, 1)
assert test_speed_imgs == (100, 500, 500, 1)
assert test_build_imgs == (100, 500, 500, 1)
assert test_wall_imgs == (100, 500, 500, 1)

test_noise_val = np.array([float(road_name[i][:2]) for i in test_idx])
#%%
input1 = layers.Input((500, 500, 1))
input2 = layers.Input((500, 500, 1))
input3 = layers.Input((500, 500, 1))
input4 = layers.Input((500, 500, 1))

# filter 갯수
conv11 = layers.Conv2D(10, (5, 5), activation='tanh')
maxpool11 = layers.MaxPooling2D((10, 10))
conv12 = layers.Conv2D(10, (5, 5), activation='tanh')
maxpool12 = layers.MaxPooling2D((10, 10))
z1 = maxpool12(conv12(maxpool11(conv11(tf.concat((input1, input2), axis=-1)))))

conv21 = layers.Conv2D(10, (5, 5), activation='tanh')
maxpool21 = layers.MaxPooling2D((10, 10))
conv22 = layers.Conv2D(10, (5, 5), activation='tanh')
maxpool22 = layers.MaxPooling2D((10, 10))
z2 = maxpool22(conv22(maxpool21(conv21(input3))))

conv31 = layers.Conv2D(10, (5, 5), activation='tanh')
maxpool31 = layers.MaxPooling2D((10, 10))
conv32 = layers.Conv2D(10, (5, 5), activation='tanh')
maxpool32 = layers.MaxPooling2D((10, 10))
z3 = maxpool32(conv32(maxpool31(conv31(input4))))

dense = layers.Dense(1)
yhat = dense(tf.concat((tf.reshape(z1, (-1, tf.math.reduce_prod(z1.shape[1:]).numpy())), 
                        tf.reshape(z2, (-1, tf.math.reduce_prod(z1.shape[1:]).numpy())), 
                        tf.reshape(z3, (-1, tf.math.reduce_prod(z1.shape[1:]).numpy()))), axis=-1))

model = K.models.Model([input1, input2, input3, input4], yhat)
model.summary()

#%%
model.compile(optimizer=K.optimizers.Adam(0.005),
              loss='mse',
              metrics=['mse'])

model.fit([traffic_imgs, speed_imgs, build_imgs, wall_imgs], noise_val,
          validation_split=0.2,
          batch_size=200,
          epochs=10)
#%%
# t = np.linspace(np.min(noise_val), np.max(noise_val), 100)
# plt.figure(figsize=(8, 8))
# plt.scatter(pred, noise_val, alpha=0.5)
# plt.plot(t, t, color='darkorange', linewidth=3)
# plt.xlabel('prediction', fontsize=15)
# plt.ylabel('true data', fontsize=15)
#%%















