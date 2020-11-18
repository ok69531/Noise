### cumulative one hot vec을 이용한 ordinal embedding 
### batch 안에서 같은 temp1을 사용하는데 계속 반복되는 문제

#%%
# !pip install opencv-python
import cv2
import os
import glob

import numpy as np
# !pip install pandas
import pandas as pd

!pip install matplotlib
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


# batch_size = 10
# idx = random.sample(range(len(road_files)), batch_size)
# size = 500

# 교통량
def traffic_batch(idx, batch_size, road_files):
    img = np.array([cv2.imread(road_files[x])[:,:,0] for x in idx])
    
    # start1 = time.time()
    # img2 = np.array(list(map(lambda x:traffic_dict.get(x, 0), 
    #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    # end1 = time.time()
    # end1 - start1
    
    # start2 = time.time()
    temp1 = pd.DataFrame({"v1" : list(traffic_dict.keys()), "v2" : list(traffic_dict.values())})
    temp2 = pd.DataFrame({"v1" : img.reshape(-1, )})
    img2 = np.array(temp2.merge(temp1, on = ["v1"], how = "left", sort = False).v2).reshape(batch_size, size, size)
    # end2 = time.time()
    # end2 - start2
    
    # np.sum(np.abs(img2 - img22))
    
    img3 = np.zeros((batch_size, size, size, 11))
    for i in list(np.unique(img2)):
        if i != 0:
            h = np.where(img2 == i)[0]
            r = np.where(img2 == i)[1]
            c = np.where(img2 == i)[2]
            img3[h, r, c, :] = traffic_hot_dict.get(i)
    
    assert img3.shape == (batch_size, size, size, 11)
    
    yield img3

# np.unique(cv2.imread(road_files[idx[0]])[:,:,0]) # 35, 255 (36 -> 2)
# np.unique(img22[0,:,:])
# plt.imshow(img3[0,:,:,1])    

# 속력
def speed_batch(idx, batch_size, road_files):
    img = np.array([cv2.imread(road_files[x])[:, :, 0] for x in idx])
    # img2 = np.array(list(map(lambda x:speed_dict.get(x, 0), 
    #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    
    temp1 = pd.DataFrame({'v1' : list(speed_dict.keys()), 'v2' : list(speed_dict.values())})
    temp2 = pd.DataFrame({'v1' : img.reshape(-1, )})
    img2 = np.array(temp2.merge(temp1, on = ['v1'], how = 'left', sort = False).v2).reshape(batch_size, size, size)
    
    img3 = np.zeros((batch_size, size, size, 4))
    for i in list(np.unique(img2)):
        if i != 0:
            h = np.where(img2 == i)[0]
            r = np.where(img2 == i)[1]
            c = np.where(img2 == i)[2]
            img3[h, r, c, :] = speed_hot_dict.get(i)
    
    assert img3.shape == (batch_size, size, size, 4)
    yield img3

# 건물
def build_batch(idx, batch_size, build_files):
    img = np.array([cv2.imread(build_files[x])[:, :, 0] for x in idx])
    # img2 = np.array(list(map(lambda x:build_dict.get(x, 0), 
    #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    
    temp1 = pd.DataFrame({'v1' : list(build_dict.keys()), 'v2' : list(build_dict.values())})
    temp2 = pd.DataFrame({'v1' : img.reshape(-1, )})
    img2 = np.array(temp2.merge(temp1, on = ['v1'], how = 'left', sort = False).v2).reshape(batch_size, size, size)
    
    img3 = np.zeros((batch_size, size, size, 11))
    for i in list(np.unique(img2)):
        if i != 0:
            h = np.where(img2 == i)[0]
            r = np.where(img2 == i)[1]
            c = np.where(img2 == i)[2]
            img3[h, r, c, :] = traffic_hot_dict.get(i)
    
    assert img3.shape == (batch_size, 500, 500, 11)
    yield img3


# 방음벽
def wall_batch(idx, batch_size, wall_files):
    img = np.array([cv2.imread(wall_files[x])[:, :, 0] for x in idx])
    # img2 = np.array(list(map(lambda x:wall_dict.get(x, 0), 
    #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    
    temp1 = pd.DataFrame({'v1' : list(wall_dict.keys()), 'v2' : list(wall_dict.values())})
    temp2 = pd.DataFrame({'v1' : img.reshape(-1, )})
    img2 = np.array(temp2.merge(temp1, on = ['v1'], how = 'left', sort = False).v2).reshape(batch_size, size, size)
    
    img3 = np.zeros((batch_size, size, size, 5))
    for i in list(np.unique(img2)):
        if i != 0:
            h = np.where(img2 == i)[0]
            r = np.where(img2 == i)[1]
            c = np.where(img2 == i)[2]
            img3[h, r, c, :] = wall_hot_dict.get(i)
    
    assert img3.shape == (batch_size, size, size, 5)
    yield img3    


#%%
input1 = layers.Input((size, size, 11))
input2 = layers.Input((size, size, 4))
input3 = layers.Input((size, size, 11))
input4 = layers.Input((size, size, 5))

emb1 = layers.Dense(1, use_bias=False)
h1 = tf.reshape(tf.matmul(tf.reshape(input1, (-1, 11)), 
                             tf.exp(emb1(tf.eye(11)))), (-1, size, size, 1))

emb2 = layers.Dense(1, use_bias=False)
h2 = tf.reshape(tf.matmul(tf.reshape(input2, (-1, 4)), 
                             tf.exp(emb2(tf.eye(4)))), (-1, size, size, 1))

emb3 = layers.Dense(1, use_bias=False)
h3 = tf.reshape(tf.matmul(tf.reshape(input3, (-1, 11)), 
                             tf.exp(emb3(tf.eye(11)))), (-1, size, size, 1))

emb4 = layers.Dense(1, use_bias=False)
h4 = tf.reshape(tf.matmul(tf.reshape(input4, (-1, 5)), 
                             tf.exp(emb4(tf.eye(5)))), (-1, size, size, 1))

# filter 갯수
conv11 = layers.Conv2D(1, (5, 5), activation='relu')
maxpool11 = layers.MaxPooling2D((10, 10))
conv12 = layers.Conv2D(1, (5, 5), activation='relu')
maxpool12 = layers.MaxPooling2D((10, 10))
z1 = maxpool12(conv12(maxpool11(conv11(tf.concat((h1, h2), axis=-1)))))

conv21 = layers.Conv2D(1, (5, 5), activation='relu')
maxpool21 = layers.MaxPooling2D((10, 10))
conv22 = layers.Conv2D(1, (5, 5), activation='relu')
maxpool22 = layers.MaxPooling2D((10, 10))
z2 = maxpool22(conv22(maxpool21(conv21(h3))))

conv31 = layers.Conv2D(1, (5, 5), activation='relu')
maxpool31 = layers.MaxPooling2D((10, 10))
conv32 = layers.Conv2D(1, (5, 5), activation='relu')
maxpool32 = layers.MaxPooling2D((10, 10))
z3 = maxpool32(conv32(maxpool31(conv31(h4))))

dense = layers.Dense(1)
yhat = dense(tf.concat((tf.reshape(z1, (-1, tf.math.reduce_prod(z1.shape[1:]).numpy())), 
                        tf.reshape(z2, (-1, tf.math.reduce_prod(z1.shape[1:]).numpy())), 
                        tf.reshape(z3, (-1, tf.math.reduce_prod(z1.shape[1:]).numpy()))), axis=-1))

model = K.models.Model([input1, input2, input3, input4], yhat)
model.summary()

#%%
batch_size = 10
optimizer = K.optimizers.Adam(0.005)
mse = K.losses.MeanSquaredError()
epochs = 10 * (len(road_files) // batch_size + 1)

# from tensorflow.python.client import device_lib 
# device_lib.list_local_devices()
for epoch in range(epochs):
    # if epoch % 10 == 1:
    #     start = time.time()
    
    start = time.time()
    idx = random.sample(range(len(road_files)), batch_size)
    
    t = tf.cast(next(iter(traffic_batch(idx, batch_size, road_files))), tf.float32)
    s = tf.cast(next(iter(speed_batch(idx, batch_size, road_files))), tf.float32)
    b = tf.cast(next(iter(build_batch(idx, batch_size, build_files))), tf.float32)
    w = tf.cast(next(iter(wall_batch(idx, batch_size, wall_files))), tf.float32)
    noise_val = np.array([float(road_name[i][:2]) for i in idx])
    print(time.time() - start)
    
    # with tf.device('/GPU:0'):
    with tf.GradientTape(persistent=True) as tape:
        pred = model([t, s, b, w])
        loss = mse(noise_val, pred)
        
    grad = tape.gradient(loss, model.weights)
    optimizer.apply_gradients(zip(grad, model.weights))
    
    # if epoch % 10 == 0:
    print("\nEpoch:", epoch, ", TRAIN loss:", loss.numpy(), 'time:', time.time() - start)

#%%
