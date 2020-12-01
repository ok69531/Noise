### train : batch
### test : memory

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
# road_dir = r'D:\noise\noise_map\100\road'
road_dir = '/home/jeon/Desktop/cho/noise_map/100//road'
road_path = os.path.join(road_dir, '*g')
road_files = sorted(glob.glob(road_path))
road_name = [] # list
for i in range(len(road_files)):
    road_name.append(road_files[i].split('/')[-1][:-4])
    # road_name.append(road_files[i].split('\\')[-1][:-4])

# build_dir = r'D:\noise\noise_map\100\building'
build_dir = '/home/jeon/Desktop/cho/noise_map/100/building'
build_path = os.path.join(build_dir, '*g')
build_files = sorted(glob.glob(build_path))
# build_files[0]

# wall_dir = r'D:\noise\noise_map\100\wall'
wall_dir = '/home/jeon/Desktop/cho/noise_map/100/wall'
wall_path = os.path.join(wall_dir, '*g')
wall_files = sorted(glob.glob(wall_path))
# wall_files[0]

# road_files[5].split('/')[-1][:10]
# build_files[5].split('/')[-1][:10]
# wall_files[5].split('/')[-1][:10]

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

# 건물 dictionary
build_dict = {255:0}
for i in range(11):
    build_dict[177+i*3] = i+1

# 방음벽 dictionary
wall_dict = {255:0}
for i in range(5):
    wall_dict[147+i*3] = i+1

#%% 
# train data는 batch를 사용
size = 100
train_idx = random.sample(range(len(road_files)), len(road_files) - 100)
test_idx = [x for x in range(len(road_files)) if x not in train_idx]
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

for i in tqdm(range(len(test_idx))):
    img2 = cv2.imread(road_files[i])[:, :, 0]
    # img2 = np.array(list(map(lambda x:traffic_dict.get(x, 0), 
    #                           list(img.reshape(-1, ))))).reshape(batch_size, 500, 500)
    
    temp2 = pd.DataFrame({'v1' : img2.reshape(-1, )})
    img2 = np.array(temp2.merge(temp_traffic, on = ['v1'], how = 'left', sort = False).v2).reshape(size, size, 1)
    
    assert img2.shape == (size, size, 1)
    
    test_traffic_imgs.append(img2/ max(traffic_dict.values()))
    

for i in tqdm(range(len(test_idx))):
    img2 = cv2.imread(road_files[i])[:, :, 0]
    
    temp2 = pd.DataFrame({'v1' : img2.reshape(-1, )})
    img2 = np.array(temp2.merge(temp_speed, on = ['v1'], how = 'left', sort = False).v2).reshape(size, size, 1)
    
    assert img2.shape == (size, size, 1)
    
    test_speed_imgs.append(img2 / max(speed_dict.values()))


for i in tqdm(range(len(test_idx))):
    img2 = cv2.imread(build_files[i])[:, :, 0]
    
    temp2 = pd.DataFrame({'v1' : img2.reshape(-1, )})
    img2 = np.array(temp2.merge(temp_build, on = ['v1'], how = 'left', sort = False).v2).reshape(size, size, 1)
    
    assert img2.shape == (size, size, 1)
    
    test_build_imgs.append(img2 / max(build_dict.values()))


for i in tqdm(range(len(test_idx))):
    img2 = cv2.imread(wall_files[i])[:, :, 0]
    
    temp2 = pd.DataFrame({'v1' : img2.reshape(-1, )})
    img2 = np.array(temp2.merge(temp_wall, on = ['v1'], how = 'left', sort = False).v2).reshape(size, size, 1)
    
    assert img2.shape == (size, size, 1)
    
    test_wall_imgs.append(img2 / max(wall_dict.values()))

test_traffic_imgs = np.array(test_traffic_imgs, np.float32)
test_speed_imgs = np.array(test_speed_imgs, np.float32)
test_build_imgs = np.array(test_build_imgs, np.float32)
test_wall_imgs = np.array(test_wall_imgs, np.float32)

assert test_traffic_imgs.shape == (100, size, size, 1)
assert test_speed_imgs.shape == (100, size, size, 1)
assert test_build_imgs.shape == (100, size, size, 1)
assert test_wall_imgs.shape == (100, size, size, 1)

test_noise_val = np.array([float(road_name[i][:2]) for i in test_idx])

#%% 
# model_1
# 도로, 건물, 방음벽에대해 각각 cnn을 돌리고 마지막에 concat한 모델 사용
# 도르는 교통량과 속도 두 가지 input이 있기때문에 도로에 대해 cnn을 수행하기 전에 concat해서 사용
input1 = layers.Input((size, size, 1))
input2 = layers.Input((size, size, 1))
input3 = layers.Input((size, size, 1))
input4 = layers.Input((size, size, 1))

conv11 = layers.Conv2D(10, (5, 5), activation='tanh')
maxpool11 = layers.MaxPooling2D((5, 5))
conv12 = layers.Conv2D(10, (5, 5), activation='tanh')
maxpool12 = layers.MaxPooling2D((10, 10))
z1 = maxpool12(conv12(maxpool11(conv11(tf.concat((input1, input2), axis=-1)))))

conv21 = layers.Conv2D(10, (5, 5), activation='tanh')
maxpool21 = layers.MaxPooling2D((5, 5))
conv22 = layers.Conv2D(10, (5, 5), activation='tanh')
maxpool22 = layers.MaxPooling2D((10, 10))
z2 = maxpool22(conv22(maxpool21(conv21(input3))))

conv31 = layers.Conv2D(10, (5, 5), activation='tanh')
maxpool31 = layers.MaxPooling2D((5, 5))
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
# model_2
# 도로의 교통량과 속도에대해 따로 cnn
# 도로와 방음벽은 데이터 값의 대부분이 0이기 때문에 pooling layer를 통과하여 0의 비율을 낮춘 후 cnn
avg1 = layers.AveragePooling2D((3, 3), strides = (1, 1))
avg2 = layers.AveragePooling2D((3, 3), strides = (2, 2))
avg3 = layers.AveragePooling2D((4, 4), strides = (2, 2))

# plt.imshow(test_build_imgs[0, ...], cmap = 'gray')
# plt.imshow(avg3(avg2(avg1(test_build_imgs))).numpy()[0], cmap = 'gray')

conv11 = layers.Conv2D(5, (5, 5), activation='relu')
maxpool11 = layers.AveragePooling2D((3, 3), strides=(2, 2))
conv12 = layers.Conv2D(10, (5, 5), activation='relu')
maxpool12 = layers.AveragePooling2D((3, 3))
z1 = maxpool12(conv12(maxpool11(conv11(avg3(avg2(avg1(input1)))))))

conv21 = layers.Conv2D(5, (5, 5), activation='relu')
maxpool21 = layers.AveragePooling2D((3, 3), strides=(2, 2))
conv22 = layers.Conv2D(10, (5, 5), activation='relu')
maxpool22 = layers.AveragePooling2D((3, 3))
z2 = maxpool22(conv22(maxpool21(conv21(avg3(avg2(avg1(input2)))))))

conv31 = layers.Conv2D(5, (5, 5), strides=(2, 2), activation='relu')
maxpool31 = layers.AveragePooling2D((3, 3), strides=(2,2))
conv32 = layers.Conv2D(10, (5, 5), activation='relu')
maxpool32 = layers.AveragePooling2D((3, 3), strides=(1,1))
z3 = maxpool32(conv32(maxpool31(conv31(input3))))

conv41 = layers.Conv2D(5, (5, 5), activation='relu')
maxpool41 = layers.AveragePooling2D((3, 3), strides=(2, 2))
conv42 = layers.Conv2D(10, (5, 5), activation='relu')
maxpool42 = layers.AveragePooling2D((3, 3))
z4 = maxpool42(conv42(maxpool41(conv41(avg3(avg2(avg1(input4)))))))

# dense1 = layers.Dense(100, activation='relu')
# dense1 = layers.Dense(10, activation='elu')
dense2 = layers.Dense(5, activation='elu')
dense3 = layers.Dense(1)
yhat = dense3(dense2(tf.concat((tf.concat((tf.reshape(z1, (-1, tf.math.reduce_prod(z1.shape[1:]).numpy())), 
                                           tf.reshape(z2, (-1, tf.math.reduce_prod(z2.shape[1:]).numpy()))), axis = -1), 
                                tf.reshape(z3, (-1, tf.math.reduce_prod(z3.shape[1:]).numpy())), 
                                tf.reshape(z4, (-1, tf.math.reduce_prod(z4.shape[1:]).numpy()))), axis=-1)))
model = K.models.Model([input1, input2, input3, input4], yhat)
model.summary()

#%%
# model_3
# input 값들을 하나로 뭉친 후 cnn
total_input = tf.concat((input1, input2, input3, input4), axis=-1)

avg1 = layers.AveragePooling2D((4, 4))
avg2 = layers.AveragePooling2D((2, 2), strides = (1, 1))

h = avg2(avg1(total_input))

conv1 = layers.Conv2D(4, (5, 5), activation='tanh')
conv2 = layers.Conv2D(2, (5, 5), activation='tanh')
conv3 = layers.Conv2D(1, (5, 5), activation='tanh')
maxpool1 = layers.MaxPool2D((2, 2), strides=(1, 1))
maxpool2 = layers.MaxPool2D((3, 3), strides=(1, 1))
maxpool3 = layers.MaxPool2D((3, 3), strides=(1, 1))

h = maxpool3(conv3(maxpool2(conv2(maxpool1(conv1(h))))))
dense = layers.Dense(1)
yhat = dense(tf.reshape(h, [-1, tf.math.reduce_prod(h.shape[1:]).numpy()]))

model = K.models.Model([input1, input2, input3, input4], yhat)
model.summary()

#%%
batch_size = 3000
optimizer = K.optimizers.Adam(0.005)
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
y_pred = model.predict([test_traffic_imgs, test_speed_imgs, test_build_imgs, test_wall_imgs])
test_noise_val

mse(test_noise_val, y_pred).numpy()
#%%
t = np.linspace(np.min(test_noise_val), np.max(test_noise_val), 100)
plt.figure(figsize=(8, 8))
plt.scatter(y_pred, test_noise_val, alpha=0.5)
plt.plot(t, t, color='darkorange', linewidth=3)
plt.xlabel('prediction', fontsize=15)
plt.ylabel('true data', fontsize=15)











