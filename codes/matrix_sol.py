#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 01:19:05 2020

@author: mingjunliu
"""

import collections
import numpy as np
import pandas as pd
import time
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm



# dataset_raw = pd.read_csv('trip01.csv')

# dataset_raw = pd.read_csv('ratings_10000.csv')

# triplet_dataset, triplet_test = train_test_split(dataset_raw, test_size = 0.2)


# dataset_raw = pd.read_csv(filepath_or_buffer='kaggle_visible_evaluation_triplets.txt', sep='\t', header=None)

# part1, part2 = train_test_split(dataset_raw, test_size = 0.001)

# part2.columns = ['user', 'song', 'rating']
# triplet_dataset, triplet_test = train_test_split(part2, test_size = 0.2)

triplet_train = pd.read_csv(filepath_or_buffer='Test/msd_train_visible.txt', sep='\t', header=None)
triplet_train.columns = ['user', 'song', 'rating']
train1, train2 = train_test_split(triplet_train, test_size = 0.2)

triplet_dataset = pd.read_csv(filepath_or_buffer='Test/msd_test_visible.txt', sep='\t', header=None)
triplet_dataset.columns = ['user', 'song', 'rating']

# triplet_dataset = pd.concat([triplet_dataset, train2])


triplet_test = pd.read_csv(filepath_or_buffer='Test/msd_test_hidden.txt', sep='\t', header=None)
triplet_test.columns = ['user', 'song', 'rating']



row_idx = []
col_idx = []
data_ary = []
user_dict = {}
song_dict = {}
user_mean_dict = collections.defaultdict(list)
song_mean_dict = collections.defaultdict(list)
db = collections.defaultdict(dict)

for idx, row in tqdm(triplet_dataset.iterrows()):
    uid = row['user']
    sid = row['song']
    rating = int(row['rating'])
    
    if uid not in user_dict:
        user_dict[uid] = len(user_dict)
    if sid not in song_dict:
        song_dict[sid] = len(song_dict)
        
    row_idx.append(song_dict[sid])
    col_idx.append(user_dict[uid])
    
    user_mean_dict[user_dict[uid]].append(rating)
    song_mean_dict[song_dict[sid]].append(rating)
    data_ary.append(rating)
    

print('done db')
data_ary = np.asarray(data_ary)
mean_rating = np.mean(data_ary)

user_mean_ary = np.zeros(len(user_mean_dict), dtype = np.float32)
for idx, ary in user_mean_dict.items():
    user_mean_ary[idx] = np.mean(ary) - mean_rating
user_mean_ary = user_mean_ary.reshape((1, -1))
user_mean_ary = np.repeat(user_mean_ary, len(song_dict), 0)

print('done user mean')

song_mean_ary = np.zeros(len(song_mean_dict), dtype = np.float32)
for idx, ary in song_mean_dict.items():
    song_mean_ary[idx] = np.mean(ary) - mean_rating
song_mean_ary = song_mean_ary.reshape((-1, 1))
song_mean_ary = np.repeat(song_mean_ary, len(user_dict), 1)

print('done song mean')

# base_rating = user_mean_ary + song_mean_ary + mean_rating
base_rating = np.ones((len(song_dict), len(user_dict))) * mean_rating

m = np.zeros((len(song_dict), len(user_dict)), dtype = np.float32)

# for i in range(len(data_ary)):
#     m[row_idx[i]][col_idx[i]] = data_ary[i] - mean_rating - user_mean_ary[0][col_idx[i]] - song_mean_ary[row_idx[i]][0]

for idx, row in tqdm(triplet_dataset.iterrows()):
    uid = row['user']
    sid = row['song']
    rating = int(row['rating'])
    m[song_dict[sid], user_dict[uid]] = rating


print('done m')

m = m + user_mean_ary#+ song_mean_ary + user_mean_ary

X = m
u, s, vh = np.linalg.svd(X)
s = np.sqrt(s)
s_u = np.zeros(X.shape)
for i, v in enumerate(s):
    s_u[i, i] = v
s_v = np.diag(s)

r =5
A = np.matmul(u, s_u[:, :r])
B = np.matmul(s_v[:r, :], vh)
AB = np.matmul(A, B)
loss = np.mean((AB - X) ** 2)

X_re = AB + user_mean_ary + song_mean_ary + base_rating 

print('done training')


N = 14
pred = []

user_list = triplet_test['user'].unique()
for uid in tqdm(user_list):
    user_song_list = triplet_test.loc[triplet_test['user'] == uid ]['song'].tolist()
    user_train_list = triplet_dataset.loc[triplet_dataset['user'] == uid]['song'].tolist()
    user_index = user_dict.get(uid, None)
    
    song_rating = []
    for sid in song_dict:
        song_index = song_dict.get(sid, None)
        
        if user_index is not None and song_index is not None:
            rating = X_re[song_index][user_index]
        else:
            if user_index is not None:
                rating = user_mean_ary[0][user_index] + mean_rating
            elif song_index is not None:
                rating = song_mean_ary[song_index][0] + mean_rating
            else:
                    rating = mean_rating
        if rating > 0:
            song_rating.append([sid, rating])
    song_rating.sort(key = lambda x: x[1], reverse = True)
    
    p = []
    for k in range(1, N + 1):
        count = 0
        song = 0
        for i in song_rating:
            if i[0] not in user_train_list:
                song += 1
                if i[0] in user_song_list:
                    count += 1
            
            if song >= k:
                break
            
        if i[0] in user_song_list:
            p.append(count/k)
        else:
            p.append(0)
    pred.append(sum(p) / min(N, len(song_rating)))
    
    # pred.append(count/len(user_song_list))
        
    # break
        
mAP = np.mean(pred)
        
print(mAP)
#     for i in song_rating[:N]:
#         if i[0] in user_song_list:
#             count += 1
#     pred.append(count)
    
# mAP = np.mean(pred)
            



    
    
    
    
    
# test_rating = []
# test_truth = []
# for idx, row in tqdm(triplet_test.iterrows()):
#     uid = row['user']
#     sid = row['song']
    
#     user_index = user_dict.get(uid, None)
#     song_index = song_dict.get(sid, None)
    
#     if user_index is not None and song_index is not None:
#         rating = X_re[song_index][user_index]
#     else:
#         if user_index is not None:
#             rating = user_mean_ary[0][user_index] + mean_rating
#         elif song_index is not None:
#             rating = song_mean_ary[song_index][0] + mean_rating
#         else:
#                 rating = mean_rating
    
#     # rating = np.clip(rating, 0.5, None)
#     test_rating.append(rating)
#     test_truth.append(float(row['rating']))

# test_rating = np.array(test_rating)
# mse = np.mean((test_rating - test_truth) ** 2)

# print(mse)

# test_rating = {}
# test_truth = {}
# for idx, row in tqdm(triplet_test.iterrows()):
#     uid = row['user']
#     sid = row['song']
    
#     user_index = user_dict.get(uid, None)
#     song_index = song_dict.get(sid, None)
    
#     if user_index is not None and song_index is not None:
#         rating = X_re[song_index][user_index]
#     else:
#         if user_index is not None:
#             rating = user_mean_ary[0][user_index] + mean_rating
#         elif song_index is not None:
#             rating = song_mean_ary[song_index][0] + mean_rating
#         else:
#                 rating = mean_rating
    
#     # rating = np.clip(rating, 0.5, None)
#     # test_rating.append(rating)
#     # test_truth.append(float(row['rating']))
    
#     if uid not in test_rating:
#         test_rating[uid] = [[sid, rating]]
#         test_truth[uid] = [[sid, float(row['rating'])]]
#     else:
#         test_rating[uid].append([sid, rating])
#         test_truth[uid].append([sid, float(row['rating'])])
        
# pre = []
# for uid in test_rating:
#     test_rating[uid].sort(key = lambda x: x[1], reverse = True)
#     test_truth[uid].sort(key = lambda x: x[1], reverse = True)
#     count = 0
#     total = 0
#     for i in range(len(test_rating[uid])):
#         if test_truth[uid][i][1] != 1:
#             total += 1
#             if test_rating[uid][i][0] == test_truth[uid][i][0]:
#                 count += 1
#     pre.append(count / len(test_rating[uid]))

# mAP = np.mean(pre)
# print(mAP)





