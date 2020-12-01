
import cv2
import copy
import os
import glob

import numpy as np
import tensorflow as tf
# import pandas as pd

from matplotlib import pyplot as plt
from tqdm import tqdm
# from PIL import Image

# import skimage
# from skimage.measure import block_reduce
# from scipy import signal

#%%
# 폴더 안의 사진 한장씩 불러와서 전처리하고 저장하기
# 도로먼저
road_dir = r'server'
road_path = os.path.join(road_dir, '*g')
files = glob.glob(road_path)

for i in tqdm(range(len(files))):
    # img 하나만 불러와서
    img = cv2.imread(files[i])[:, :, 0]
    
    # resizing하고
    rcode = np.unique(img)
    onehot = np.full((len(rcode), 500, 500), 0)
    
    for j in range(len(rcode)):
        idx = np.where(img == rcode[j])
        zeromat = np.zeros_like(img)
        zeromat[idx] = 1e+7
        
        hot_mat = cv2.resize(zeromat, (500, 500))
        hot_mat[np.where(hot_mat > 0)] = rcode[j]
        assert np.all(hot_mat < 256)
        onehot[j, :, :] = hot_mat
        
    # 다시 합치기
    indicator1 = tf.reduce_sum(np.array(onehot, np.bool)*1, axis=[1, 2]).numpy()
    indicator2 = tf.reduce_sum(np.array(onehot, np.bool)*1, axis=0).numpy()
    
    resized = tf.reduce_sum(onehot, axis=0).numpy()
    resized[np.where(indicator2 > 1)] = rcode[np.argsort(indicator1)[-2]]
    resized[np.where(resized == 0)] = 255
    
    # assert np.all(np.unique(resized) == rcode)
    assert all(np.isin(np.unique(resized), rcode))
        
    save_dir = "D:/noise/noise_map/road/" + files[i].split('\\')[-1]
    cv2.imwrite(save_dir, resized)


# 데이터 이름
# data_list = os.listdir(road_dir)
# data_name = [file for file in data_list if file.endswith('.png')]
# data_name[0][0:2]
# noise_val = [x[0:2] for x in data_name]

######### 44_4853019_190690_284120_road : 데이터 손실 좀 있는듯..

#%% building
build_dir = r''
build_path = os.path.join(build_dir, '*g')
build_files = glob.glob(build_path)

for i in tqdm(range(len(build_files))):
    img = cv2.imread(build_files[i])[:, :, 0]
    
    bcode = np.unique(img)
    onehot = np.full((len(bcode), 500, 500), 0)
    
    for j in range(len(bcode)):
        idx = np.where(img == bcode[j])
        zeromat = np.zeros_like(img)
        zeromat[idx] = 5e+7
        
        hotmat = cv2.resize(zeromat, (500, 500))
        hotmat[np.where(hotmat > 0)] = bcode[j]
        assert np.all(hotmat < 256)
        onehot[j, :, :] = hotmat
    
    indicator1 = tf.reduce_sum(np.array(onehot, np.bool)*1, axis = [1, 2]).numpy()
    indicator2 = tf.reduce_sum(np.array(onehot, np.bool)*1, axis = 0).numpy()
    
    resized = tf.reduce_sum(onehot, axis = 0).numpy()
    resized[np.where(indicator2 > 1)] = bcode[np.argsort(indicator1)[-2]]
    resized[np.where(resized == 0)] = 255
    
    assert np.all(np.isin(np.unique(resized), bcode))
    
    save_dir = "D:/noise/noise_map/building/" + build_files[i].split('\\')[-1]
    cv2.imwrite(save_dir, resized)
    
#%% soundproof wall
wall_dir = r''
wall_path = os.path.join(wall_dir, '*g')
wall_files = glob.glob(wall_path)

for i in tqdm(range(len(wall_files))):
    img = cv2.imread(wall_files[i])[:, :, 0]
    
    wcode = np.unique(img)
    onehot = np.full((len(wcode), 500, 500), 0)
    
    for j in range(len(wcode)):
        idx = np.where(img == wcode[j])
        zeromat = np.zeros_like(img)
        zeromat[idx] = 5e+7
        
        hotmat = cv2.resize(zeromat, (500, 500))
        hotmat[np.where(hotmat > 0)] = wcode[j]
        assert np.all(hotmat < 256)
        onehot[j, :, :] = hotmat
    
    indicator1 = tf.reduce_sum(np.array(onehot, np.bool)*1, axis = [1, 2]).numpy()
    indicator2 = tf.reduce_sum(np.array(onehot, np.bool)*1, axis = 0).numpy()
    
    resized = tf.reduce_sum(onehot, axis = 0).numpy()
    resized[np.where(indicator2 > 1)] = wcode[np.argsort(indicator)[-2]]
    resized[np.where(resized == 0)] = 255
    
    assert np.all(np.isin(np.unique(resized), wcode))
    
    save_dir = "D:/noise/noise_map/wall/" + wall_files[i].split('\\')[-1]
    cv2.imwrite(save_dir, resized)
    
