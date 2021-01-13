import cv2
import glob
import os

import numpy as np

from matplotlib import pyplot as plt
from sklearn.feature_extraction import DictVectorizer

#%%
road_dir = '/home/jeon/Desktop/cho/noise_map/100/road_pool'
road_path = os.path.join(road_dir, '*g')
road_files = sorted(glob.glob(road_path))
road_names = [road_files[i].split('/')[-1] for i in range(len(road_files))]

build_dir = '/home/jeon/Desktop/cho/noise_map/100/building'
build_path = os.path.join(build_dir, '*g')
build_files = sorted(glob.glob(build_path))
build_names = [build_files[i].split('/')[-1] for i in range(len(build_files))]

wall_dir = '/home/jeon/Desktop/cho/noise_map/100/wall_pool'
wall_path = os.path.join(wall_dir, '*g')
wall_files = sorted(glob.glob(wall_path))
wall_names = [wall_files[i].split('/')[-1] for i in range(len(wall_files))]

all_idx = range(len(wall_names))

## road
u = []
def road_prop(idx, start1, width1, start2, width2):
    img = np.array([cv2.imread(road_files[i])[:, :, 0] for i in idx])
    rcode = [np.unique(x)[:-1] for x in img]
    u.append(rcode)
    n = width1*width1 - width2*width2
    p = {idx[j] : {rcode[j][i] : ((np.sum(img[j, start1:(start1+width1), start1:(start1+width1)] == rcode[j][i]) - 
                                    np.sum(img[j, start2:(start2+width2), start2:(start2+width2)] == rcode[j][i])) / n) 
                    for i in range(len(rcode[j]))} 
          for j in range(img.shape[0])}
    
    return p

road_prop10 = road_prop(all_idx, 45, 10, 0, 0)
road_prop20 = road_prop(all_idx, 40, 20, 45, 10)
road_prop30 = road_prop(all_idx, 35, 30, 40, 20)
road_prop40 = road_prop(all_idx, 30, 40, 35, 30)
road_prop50 = road_prop(all_idx, 25, 50, 30, 40)
road_prop60 = road_prop(all_idx, 20, 60, 25, 50)
road_prop70 = road_prop(all_idx, 15, 70, 20, 60)
road_prop80 = road_prop(all_idx, 10, 80, 15, 70)
road_prop90 = road_prop(all_idx, 5, 90, 10, 80)
road_prop100 = road_prop(all_idx, 0, 100, 5, 90)
rcode_unique = np.unique(np.hstack(np.hstack(u)))

dicttomat = DictVectorizer(sparse = False)
road10 = dicttomat.fit_transform([road_prop10[i] for i in range(len(all_idx))])
road20 = dicttomat.fit_transform([road_prop20[i] for i in range(len(all_idx))])
road30 = dicttomat.fit_transform([road_prop30[i] for i in range(len(all_idx))])
road40 = dicttomat.fit_transform([road_prop40[i] for i in range(len(all_idx))])
road50 = dicttomat.fit_transform([road_prop50[i] for i in range(len(all_idx))])
road60 = dicttomat.fit_transform([road_prop60[i] for i in range(len(all_idx))])
road70 = dicttomat.fit_transform([road_prop70[i] for i in range(len(all_idx))])
road80 = dicttomat.fit_transform([road_prop80[i] for i in range(len(all_idx))])
road90 = dicttomat.fit_transform([road_prop90[i] for i in range(len(all_idx))])
road100 = dicttomat.fit_transform([road_prop100[i] for i in range(len(all_idx))])
dicttomat.get_feature_names()
all(rcode_unique == dicttomat.get_feature_names())

road = np.concatenate((road10, road20, road30, road40, road50, 
                        road60, road70, road80, road90, road100), axis = 1)
# build_train = build[train_idx, ]; build_test = build[test_idx, ]
pd.DataFrame(road).to_csv('/home/jeon/Desktop/cho/noise_map/100/road_prop.csv', index = False, header = False)


### building
u = []
def build_prop(idx, start1, width1, start2, width2):
    img = np.array([cv2.imread(build_files[i])[:, :, 0] for i in idx])
    bcode = [np.unique(x)[:-1] for x in img]
    u.append(bcode)
    n = width1*width1 - width2*width2
    p = {idx[j] : {bcode[j][i] : ((np.sum(img[j, start1:(start1+width1), start1:(start1+width1)] == bcode[j][i]) - 
                                    np.sum(img[j, start2:(start2+width2), start2:(start2+width2)] == bcode[j][i])) / n) 
                    for i in range(len(bcode[j]))} 
          for j in range(img.shape[0])}
    
    return p


build_prop10 = build_prop(all_idx, 45, 10, 0, 0)
build_prop20 = build_prop(all_idx, 40, 20, 45, 10)
build_prop30 = build_prop(all_idx, 35, 30, 40, 20)
build_prop40 = build_prop(all_idx, 30, 40, 35, 30)
build_prop50 = build_prop(all_idx, 25, 50, 30, 40)
build_prop60 = build_prop(all_idx, 20, 60, 25, 50)
build_prop70 = build_prop(all_idx, 15, 70, 20, 60)
build_prop80 = build_prop(all_idx, 10, 80, 15, 70)
build_prop90 = build_prop(all_idx, 5, 90, 10, 80)
build_prop100 = build_prop(all_idx, 0, 100, 5, 90)
bcode_unique = np.unique(np.hstack(np.hstack(u)))

dicttomat = DictVectorizer(sparse = False)
build10 = dicttomat.fit_transform([build_prop10[i] for i in range(len(all_idx))])
build20 = dicttomat.fit_transform([build_prop20[i] for i in range(len(all_idx))])
build30 = dicttomat.fit_transform([build_prop30[i] for i in range(len(all_idx))])
build40 = dicttomat.fit_transform([build_prop40[i] for i in range(len(all_idx))])
build50 = dicttomat.fit_transform([build_prop50[i] for i in range(len(all_idx))])
build60 = dicttomat.fit_transform([build_prop60[i] for i in range(len(all_idx))])
build70 = dicttomat.fit_transform([build_prop70[i] for i in range(len(all_idx))])
build80 = dicttomat.fit_transform([build_prop80[i] for i in range(len(all_idx))])
build90 = dicttomat.fit_transform([build_prop90[i] for i in range(len(all_idx))])
build100 = dicttomat.fit_transform([build_prop100[i] for i in range(len(all_idx))])
dicttomat.get_feature_names()
all(bcode_unique == dicttomat.get_feature_names())

build = np.concatenate((build10, build20, build30, build40, build50, 
                        build60, build70, build80, build90, build100), axis = 1)
pd.DataFrame(build).to_csv('/home/jeon/Desktop/cho/noise_map/100/build_prop.csv', index = False, header = False)




### wall
u = []
def wall_prop(idx, start1, width1, start2, width2):
    img = np.array([cv2.imread(wall_files[i])[:, :, 0] for i in idx])
    wcode = [np.unique(x)[1:] for x in img]
    u.append(wcode)
    n = width1*width1 - width2*width2
    p = {idx[j] : {wcode[j][i] : (round(np.sum(img[j, start1:(start1+width1), start1:(start1+width1)] == wcode[j][i]) - 
                                    np.sum(img[j, start2:(start2+width2), start2:(start2+width2)] == wcode[j][i])) / n) 
                    for i in range(len(wcode[j]))} 
          for j in range(img.shape[0])}
    
    return p


wall_prop10 = wall_prop(all_idx, 45, 10, 0, 0)
wall_prop20 = wall_prop(all_idx, 40, 20, 45, 10)
wall_prop30 = wall_prop(all_idx, 35, 30, 40, 20)
wall_prop40 = wall_prop(all_idx, 30, 40, 35, 30)
wall_prop50 = wall_prop(all_idx, 25, 50, 30, 40)
wall_prop60 = wall_prop(all_idx, 20, 60, 25, 50)
wall_prop70 = wall_prop(all_idx, 15, 70, 20, 60)
wall_prop80 = wall_prop(all_idx, 10, 80, 15, 70)
wall_prop90 = wall_prop(all_idx, 5, 90, 10, 80)
wall_prop100 = wall_prop(all_idx, 0, 100, 5, 90)
wcode_unique = np.unique(np.hstack(np.hstack(u)))

dicttomat = DictVectorizer(sparse = False)
wall10 = dicttomat.fit_transform([wall_prop10[i] for i in range(len(all_idx))])
wall20 = dicttomat.fit_transform([wall_prop20[i] for i in range(len(all_idx))])
wall30 = dicttomat.fit_transform([wall_prop30[i] for i in range(len(all_idx))])
wall40 = dicttomat.fit_transform([wall_prop40[i] for i in range(len(all_idx))])
wall50 = dicttomat.fit_transform([wall_prop50[i] for i in range(len(all_idx))])
wall60 = dicttomat.fit_transform([wall_prop60[i] for i in range(len(all_idx))])
wall70 = dicttomat.fit_transform([wall_prop70[i] for i in range(len(all_idx))])
wall80 = dicttomat.fit_transform([wall_prop80[i] for i in range(len(all_idx))])
wall90 = dicttomat.fit_transform([wall_prop90[i] for i in range(len(all_idx))])
wall100 = dicttomat.fit_transform([wall_prop100[i] for i in range(len(all_idx))])
dicttomat.get_feature_names()
all(wcode_unique == dicttomat.get_feature_names())

wall = np.concatenate((wall10, wall20, wall30, wall40, wall50, 
                        wall60, wall70, wall80, wall90, wall100), axis = 1)
pd.DataFrame(wall).to_csv('/home/jeon/Desktop/cho/noise_map/100/wall_prop.csv', index = False, header = False)

