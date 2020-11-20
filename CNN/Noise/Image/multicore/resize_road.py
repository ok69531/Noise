import cv2
import os
import glob
import argparse

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from matplotlib import pyplot as plt
from tensorflow.keras import layers

parser = argparse.ArgumentParser(description='idx')
parser.add_argument('--idx', type=int)
args = parser.parse_args()
file_idx = args.idx

road_dir = '/home/jeon/Desktop/cho/raw/new_map/road'
road_path = os.path.join(road_dir, '*g')
road_files = sorted(glob.glob(road_path))
# file_idx = 3

try:
    if ((file_idx>=0)&(file_idx<=12)):
        road_file_batch = road_files[(file_idx * 1000):((file_idx+1)*1000)]
    elif(file_idx == 13):
        road_file_batch = road_files[(file_idx * 1000):]
except:
    print('You got the wrong file index!')    

max1 = layers.MaxPooling2D((6, 6))
max2 = layers.MaxPooling2D((5, 5))
max3 = layers.MaxPooling2D((5, 5), strides = (2, 2))
max4 = layers.MaxPooling2D((3, 3), strides = (1, 1))

# ssh = paramiko.SSHClient()
# ssh.set_missing_host_key_policy(paramiko.WarningPolicy)
# ssh.connect("localhost", username="admin", password="pass")

for i in range(len(road_file_batch)):
    img = cv2.imread(road_file_batch[i])[:, :, 0]
    img[np.where(img == 255)] = 0
    resized = max4(max3(max2(max1(img[np.newaxis, ..., np.newaxis]))))
    
    save_dir = "/home/jeon/Desktop/cho/noise_map/100/road_pool/" + road_file_batch[i].split('/')[-1]
    cv2.imwrite(save_dir, tf.squeeze(resized).numpy())

# cd /home/jeon/~~
#  nohub python resize_pool.py --idx i &
    







