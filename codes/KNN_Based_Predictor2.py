__author__ = 'Amin'

from collections import OrderedDict
import collections
import math
import pandas as pd
import numpy as np
from tqdm import tqdm



def cosine_similarity(v1, v2):
    dotp = 0
    s = 0.0

    l2 = v2.keys()

    for item in l2:
        if item in v1:
            dotp += float(v2.get(item)) * float(v1.get(item))

    for w in v1.values():
        w = float(w)
        s += w * w
    len1 = math.sqrt(s)

    s = 0.0
    for w in v2.values():
        w = float(w)
        s += w * w
    len2 = math.sqrt(s)

    return dotp / (len1 * len2)


def cal_precision(res):
    testFile = open("Test/msd_test_hidden.txt", 'r')

    total = 0
    hidden_dict = {}
    for line in testFile:
        tokens = line.split()
        userName = tokens[0]
        itemName = tokens[1]

        if not userName in hidden_dict:
            list = []
            list.append(itemName)
            hidden_dict[userName] = list
        else:
            hidden_dict[userName].append(itemName)
        total += 1


    count = 0

    for (user, recommend_songs) in res.items():
        # total += len(recommend_songs)
        for song in recommend_songs:
            if song in hidden_dict[user]:
                count += 1

    print(count / total)


userMap = {}
itemMap = collections.defaultdict(OrderedDict)
userItemMap = collections.defaultdict(dict)
userTestMap = collections.defaultdict(dict)

itemReverseMap = collections.defaultdict()
userTestReverseMap = collections.defaultdict()
trainFile = open('Test/msd_train_visible.txt', 'r')
testFile = open("Test/msd_test_visible.txt", 'r')
# fileOut = open('out_file.txt', 'a')

for line in trainFile:

    tokens = line.split()
    userName = tokens[0]
    itemName = tokens[1]
    numberOfUsers = len(userMap)
    numberOfItems = len(itemMap)

    if not userName in userMap:
        userMap[userName] = numberOfUsers + 1
        userValue = numberOfUsers + 1
    else:
        userValue = userMap.get(userName)

    if not itemName in itemMap:
        itemMap[itemName] = numberOfItems + 1
        itemValue = numberOfItems + 1
        itemReverseMap[numberOfItems+1] = itemName
    else:
        itemValue = itemMap.get(itemName)

    userItemMap[userValue][itemValue] = tokens[2]
print("Done1")


for line in testFile:
    tokens = line.split()
    userName = tokens[0]
    itemName = tokens[1]
    numberOfUsers = len(userMap)
    numberOfItems = len(itemMap)

    if not userName in userMap:
        userMap[userName] = numberOfUsers + 1
        userValue = numberOfUsers + 1
        userTestReverseMap[numberOfUsers + 1] = userName
    else:
        userValue = userMap.get(userName)

    if not itemName in itemMap:
        itemMap[itemName] = numberOfItems + 1
        itemValue = numberOfItems + 1
    else:
        itemValue = itemMap.get(itemName)

    userTestMap[userValue][itemValue] = tokens[2]
print("Done2")


cosineDistance = collections.defaultdict(dict)
for testUser in userTestMap.items():
    for trainUser in userItemMap.items():
        val = cosine_similarity(trainUser[1], testUser[1])
        cosineDistance[testUser[0]][trainUser[0]] = val
print("Done3")

lenCos = len(cosineDistance)

K = 10
nearestNeighbors = [[]]
for i in range(0, lenCos - 1, 1):
    topKUsers = sorted(list(cosineDistance.items())[i][1].items(), key=lambda t: float(t[1]), reverse=True)
    userID = [[]]
    for j in range(0, K, 1):
        userID.append(topKUsers[j][0])
    nearestNeighbors.append([list(cosineDistance.items())[i][0], userID[1:]])

N = 30
lenNearestNeighbor = len(nearestNeighbors)


res = {}
for i in range(1, lenNearestNeighbor - 1, 1):
    currTestMap = userTestMap.get(nearestNeighbors[i][0])
    newSongs = collections.defaultdict()
    for j in range(0, K, 1):
        currUserMap = userItemMap.get(nearestNeighbors[i][1][j])
        for k, v in currUserMap.items():
            if not k in currTestMap:
                if not k in newSongs:
                    newSongs[k] = v
                else:
                    val = newSongs.get(k)
                    newVal = max(val, v)
                    newSongs[k] = newVal
    d = sorted(newSongs.items(), key=lambda t: int(t[1]), reverse=True)
    numElems = min(N, len(d))
    dnew = d[:numElems]
    recommendedItems = collections.OrderedDict(dnew)


    userName = userTestReverseMap.get(nearestNeighbors[i][0])
    itemNames = []
    for key in recommendedItems.keys():
        itemName = itemReverseMap.get(key)
        itemNames.append(itemName)
    # print(fileOut, userName, " ", itemNames)
    print(userName, " ", itemNames)
    res[userName] = itemNames

#cal_precision(res)

triplet_dataset = pd.read_csv(filepath_or_buffer='Test/msd_test_visible.txt', sep='\t', header=None)
triplet_dataset.columns = ['user', 'song', 'rating']

triplet_test = pd.read_csv(filepath_or_buffer='Test/msd_test_hidden.txt', sep='\t', header=None)
triplet_test.columns = ['user', 'song', 'rating']

song_list = list(triplet_dataset['song'].unique())
song_list.sort()

user_list = list(triplet_dataset['user'].unique())
user_list.sort()

#%%
pred = []
for uid in tqdm(user_list):
    user_song_list = triplet_test.loc[triplet_test['user'] == uid]['song'].tolist()
    user_train_list = triplet_dataset.loc[triplet_dataset['user'] == uid]['song'].tolist()
    if uid in res:
        song_rating = res[uid]
        # song_rating.sort(key = lambda x : x[1], reverse = True)
        p = []
        for k in range(1, N + 1):
            count = 0
            song = 0
            for i in song_rating:
                # if i[0] not in user_train_list:
                song += 1
                if i[0] in user_song_list:
                    count += 1
                        
                if song >= k:
                    break
            # p.append( count / k)
            if i[0] in user_song_list:
                p.append(count / k)
            else:
                p.append(0)
        pred.append(sum(p) / min(N, len(song_rating)))
                    
print(np.mean(pred))

testFile.close()
trainFile.close()
# fileOut.close()