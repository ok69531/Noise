# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 13:14:40 2020

@author: SOYOUNG
"""

import cv2
import copy
import os
import glob
# import subprocess

import numpy as np
# import pandas as pd

import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm
# from PIL import Image


#%%
img = cv2.imread(r'C:\Users\SOYOUNG\Desktop\noise_image\gray\81_4949663_187710_280120.png')[:, :, 0]
img.shape
plt.imshow(img, cmap = 'Greys_r')

# img = Image.open(r'C:\Users\SOYOUNG\Desktop\noise_image\gray\81_4949663_187710_280120.png')
# img = np.array(img)
# img.shape


#%% road
road = copy.copy(img)
road = np.where(road <= 129, road, 255)
plt.imshow(road)

r_code = set()
for i in tqdm(range(len(road))):
    r_code.update(list(road[i, :]))
r_code = list(r_code)
len(r_code)

rcode = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 
         60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129]

nrcode = [x for x in r_code if x not in rcode]
nroad = np.ones(img.shape) * 255
nroad[np.isin(img, nrcode)] = img[np.isin(img, nrcode)]
plt.imshow(nroad)

#%% building
building = copy.copy(img)
buliding = np.where(building >= 177, building, 0)
plt.imshow(building, cmap = 'Greys_r') # 이상하게 그려짐, 채널 분리 전혀 안됨

# 건물의 값이 아닌 것을 모두 255로 바꿈
building = copy.copy(img)
r = np.where((building < 177) | (building > 207))[0]
c = np.where((building < 177) | (building > 207))[1]
building[r, c] = 255
# plt.imshow(building)
plt.imshow(building, cmap = 'Greys_r')
plt.imshow(building[3400:3600, 3400:3600], cmap='Greys_r')
# 도로쪽 잔상이 남음
# 건물 legend값 외에 다른 값 포함된듯

# building에 어떤 값들 있는지 확인 > 도로 legend말고 사이에 값들 더 있음
b_code = set()
for i in tqdm(range(len(building))):
    b_code.update(list(building[i, :]))
b_code = list(b_code)
len(b_code)

# 177~207, 255 사이에서 건물 legend빼고 그려보기
nbcode = [x for x in b_code if x not in [177, 180, 183, 186, 189, 192, 195, 198, 201, 204, 207, 255]]

nbuild = np.ones(img.shape) * 255
nbuild[np.isin(img, nbcode)] = img[np.isin(img, nbcode)]
plt.imshow(nbuild)

# 건물 legend만 갖고 그려보기
bcode = [177, 180, 183, 186, 189, 192, 195, 198, 201, 204, 207]

build = np.ones(img.shape) * 255
build[np.isin(img, bcode)] = img[np.isin(img, bcode)]
plt.imshow(build)
plt.imshow(build, cmap = 'Greys_r')
plt.imshow(build[3400:3600, 3400:3600], cmap = 'Greys_r')

# 건물 legend값인데 도로에 들어있는 것들이 있음,,,

#%% soundproof walls
sp = copy.copy(img)
sp = np.where((sp >= 147) & (sp <= 159), sp, 255)
plt.imshow(sp) # 이것도 도로 잔상 있음

spcode = [147, 150, 153, 156, 159]
sp = np.ones(img.shape) * 255
sp[np.isin(img, spcode)] = img[np.isin(img, spcode)]
plt.imshow(sp)
plt.imshow(sp, cmap = 'Greys_r')
plt.imshow(sp[3400:3600, 3400:3600], cmap = 'Greys_r')


#%% 이상한 값들 확인
# plt.hist(img.reshape(-1, ))

# 건물
r = np.where((img >= 208) & (img < 255))[0]
c = np.where((img >= 208) & (img < 255))[1]
sample = np.ones(img.shape) * 255
sample[r, c] = 0
plt.imshow(sample, cmap='Greys_r')
# plt.imshow(sample[3400:3600, 3400:3600], cmap='Greys_r')

# 방음벽이랑 건물 legend 사이값 > 원래 비어있어야하는데 그려짐
r = np.where((img >= 160) & (img < 177))[0]
c = np.where((img >= 160) & (img < 177))[1]
sample = np.ones(img.shape) * 255
sample[r, c] = 0
plt.imshow(sample)
# plt.imshow(sample[3400:3600, 3400:3600], cmap='Greys_r')

# 도로랑 방음벽 사이값 > 이것도 없어야 하는데 그려짐
r = np.where((img >= 130) & (img < 147))[0]
c = np.where((img >= 130) & (img < 147))[1]
sample = np.ones(img.shape) * 255
sample[r, c] = 0
plt.imshow(sample)
# plt.imshow(sample[3400:3600, 3400:3600], cmap='Greys_r')

# 도로 ~ 방음벽 ~ 건물 사이값들이 비워져있어야 함 > 윤곽선 그려짐
r = np.where(((img >= 208) & (img < 255)) | ((img >= 160) & (img < 177)) | ((img >= 130) & (img < 147)))[0]
c = np.where(((img >= 208) & (img < 255)) | ((img >= 160) & (img < 177)) | ((img >= 130) & (img < 147)))[1]
sample = np.ones(img.shape) * 255
sample[r, c] = 0
plt.imshow(sample)
# plt.imshow(sample[3400:3600, 3400:3600], cmap='Greys_r')



#%%

#%% 폴더 안의 사진 모두 불러오기
img_dir = r'C:\Users\SOYOUNG\Desktop\noise_image\gray_sample'
data_path = os.path.join(img_dir, '*g')
files = glob.glob(data_path)
data = []
for i in files:
    img = cv2.imread(i)
    data.append(img)

data[0].shape
np.array(data).shape

# 데이터 이름
data_list = os.listdir(img_dir)
data_name = [file for file in data_list if file.endswith('.png')]
data_name = np.array(data_name)
data_name.shape

noise_val = [int(i[:2]) for i in data_name]
np.array(noise_val).shape


#%% 

#%% 전처리
## road preprocessing

# 도로를 어떤걸로 해야하쥐
# 1. 이걸로하면 0~129 값 모두 나타나고
road = copy.copy(img)
road = np.where(road <= 129, road, 255)
plt.imshow(road, cmap = 'gray')

# 2. 이걸로하면 road의 legend값만 나타남
rcode = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 
         60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129]
road = np.ones(img.shape) * 255
road[np.isin(img, rcode)] = img[np.isin(img, rcode)]
plt.imshow(road, cmap = 'gray')

# 첫번째에서 사이값들은 한쪽으로 몰아서 해야할듯?
# 전처리 하려면 road legend값만 갖고 해야하니까 일단 두번째걸로...

# example: road == 9
r_code
np.sum(road == 9)
idx = np.where(road == 9)
zeromat = np.zeros_like(img)
zeromat[idx] = 1e+7

plt.imshow(zeromat, cmap = 'gray')

onehot = cv2.resize(zeromat, (500, 500))
np.where(onehot > 0)
onehot[np.where(onehot > 0)] = 9
plt.imshow(onehot, cmap = 'gray')

### road value 전부 다
onehot = np.full((len(rcode), 500, 500), 0)

for i in range(len(rcode)):
    idx = np.where(road == rcode[i])
    zeromat = np.zeros_like(img)
    zeromat[idx] = 1e+7
    
    hot_mat = cv2.resize(zeromat, (500, 500))
    hot_mat[np.where(hot_mat > 0)] = rcode[i]
    
    assert np.all(hot_mat < 256)
    
    onehot[i, :, :] = hot_mat

# plt.imshow(onehot[3, :, :], cmap = 'gray')
# plt.imshow(onehot[4, :, :], cmap = 'gray')
# plt.imshow(onehot[4, :, :][350:400, 350:400], cmap = 'gray')
# plt.imshow(onehot[31, :, :])
# np.where(onehot[31, :, :] != 0)
# plt.imshow(onehot[31, :, :][350:400, 180:220], cmap = 'gray')


# 겹치는거 제거하고 각 채널 더하기
# np.array(onehot, np.bool)
indicator1 = tf.reduce_sum(np.array(onehot, np.bool)*1, axis=[1, 2]).numpy() # 각각 road value들의 갯수, axis에는 제거할 차원
indicator2 = tf.reduce_sum(np.array(onehot, np.bool)*1, axis=0).numpy()
# onehot[:, np.where(indicator > 1)]

# onehot_sum = resized road
onehot_sum = tf.reduce_sum(onehot, axis=0).numpy()
onehot_sum[np.where(indicator2 > 1)] = rcode[np.argmax(indicator1)]
onehot_sum.shape
onehot_sum[np.where(onehot_sum == 0)] = 255
plt.hist(onehot_sum.reshape(-1, ))
plt.imshow(onehot_sum, cmap='gray')

#%% building preprocessing
road = cv2.imread(r'\\172.16.33.161\ml_gj\new_map\road\81_789698_181530_285100_road.png')[:, :, 0]
build= cv2.imread(r'\\172.16.33.161\ml_gj\new_map\building_2\81_789698_181530_285100_building.png')[:, :, 0]
wall = cv2.imread(r'\\172.16.33.161\ml_gj\new_map\wall_2\81_789698_181530_285100_wall.png')[:, :, 0]

plt.imshow(road, cmap = 'gray')
plt.imshow(build, cmap = 'gray')
plt.imshow(wall, cmap = 'gray')

m = road + build + wall
plt.imshow(m, cmap = 'gray')

road = np.array(road)
raod = road.reshape((-1))
np.unique(road)

#%%
# 폴더 안의 모든 사진 불러오기
road_dir = r'\\172.16.33.161\ml_gj\new_map\road'
road_path = os.path.join(road_dir, '*g')
files = glob.glob(road_path)
road_data = []

for i in tqdm(range(len(files))):
    img = cv2.imread(files[i])
    # 전처리 다 돌고 저장
    img.save~~~

road_data[0].shape
np.array(data).shape

# 데이터 이름
data_list = os.listdir(img_dir)
data_name = [file for file in data_list if file.endswith('.png')]
data_name = np.array(data_name)
data_name.shape

noise_val = [int(i[:2]) for i in data_name]
np.array(noise_val).shape
