#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 01:40:50 2020

@author: mingjunliu
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 18:10:53 2020

@author: mingjunliu
"""


import pdb
import collections
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score as acs
from sklearn.neighbors import KNeighborsClassifier


def get_user_song_rating_dct(dataset, user_based = True):
    db = collections.defaultdict(dict)
    for idx, row in dataset.iterrows():
        if user_based:
            userId = row['user']
            songId = row['song']
        else:
            userId = row['song']
            songId = row['user']
            
        rating = row['rating']
        
        db[userId][songId] = rating
    return db

def get_similarity_dict(db, id_list, similarity_func):
	#sim_db map user_id to its K_NEIGHBORS, sorted by similarity
	sim_db = {}
	# You could use tqdm to make a progressbar: https://github.com/tqdm/tqdm
	for target_user in tqdm(id_list):
		sim_ary = []
		target_dict = db[target_user]
		for current_user in db:
			if current_user != target_user:
				current_dict = db[current_user]
				sim = similarity_func(target_dict, current_dict)
				sim_ary.append([current_user, sim])

		sim_ary.sort(key=lambda x: x[1], reverse=True)
		sim_db[target_user] = sim_ary

	return sim_db

# d1, d2 are two rating dictionary for a user
def jaccard_similarity(d1, d2):
	d1_k = d1.keys()
	d2_k = d2.keys()
	return len(d1_k & d2_k) / len(d1_k | d2_k)

def cosine_similarity(d1, d2):
	"""
	Compute cosine similarity of user1 and user2 using dot(user1, user2) / (norm(user1) * norm(user2))
	"""
	common_key = d1.keys() & d2.keys() #intersection
	if len(common_key) > 0:
		dot = sum([d1[k] * d2[k] for k in common_key])
		norm1 = np.linalg.norm(list(d1.values())) #by default, L2 norm
		norm2 = np.linalg.norm(list(d2.values()))
		norm_12 = norm1 * norm2
		if norm_12 == 0:
			return 0.0
		else:
			sim = dot / norm_12
			return sim
	else:
		return 0.0

def centred_cosine_similarity(d1, d2):
	"""
	Compute cosine similarity after subtracting avg of user ratings for each user
	"""
	m1 = np.mean(list(d1.values()))
	m2 = np.mean(list(d2.values()))

	d1 = {k: v - m1 for k, v in d1.items()}
	d2 = {k: v - m2 for k, v in d2.items()}

	return cosine_similarity(d1, d2)

def pearson_similarity(d1, d2):
	"""
	Similarity based on co-rated items
	"""
	common_key = d1.keys() & d2.keys()
	if len(common_key) >= 2:
		d1 = {k: v for k, v in d1.items() if k in common_key}
		d2 = {k: v for k, v in d2.items() if k in common_key}
		return centred_cosine_similarity(d1, d2)
	else:
		return 0.0

def predict_ratings(db, sim_db, song_list, user_list, test_dataset, N_TOP, K_NEIGHBORS, user_based=True):
    user_list = triplet_test['user'].unique()
    pred = []
    for uid in tqdm(user_list):

        if uid in user_list:
            user_song_list = triplet_test.loc[triplet_test['user'] == uid ]['song'].tolist()
            user_train_list = triplet_dataset.loc[triplet_dataset['user'] == uid]['song'].tolist()

            r_a = np.mean(list(db[uid].values()))
            p = []
            rec = []
            for sid in song_list:
                
                if sid not in db[uid]:
                    num_ary = []
                    den_ary = []
                    for user_id, sim in sim_db[uid]:
                        if sim > 0:
                            r_ui = db[user_id].get(sid, None)
                            if r_ui is not None:
                                r_u = np.mean([v for k, v in db[user_id].items() if k!=sid])
                                num_ary.append((r_ui-r_u) * sim)
                                den_ary.append(sim)
                                if len(den_ary)>=K_NEIGHBORS:
                                    break
                    if len(den_ary)==0:
                        p_ai = r_a
                    else:
                        p_ai = r_a + sum(num_ary)/sum(den_ary)
                        
                    if p_ai > 0:
                        p.append([sid, p_ai])
            p.sort(key = lambda x: x[1], reverse = True)
            
            for k in range(1, N_TOP  + 1):
                count = 0
                song = 0
                for i in p:
                    if i[0] not in user_train_list:
                        song += 1
                        if i[0] in user_song_list:
                            count += 1
                    if song >= k:
                        break
                if i[0] in user_song_list:
                    rec.append(count/k)
                else:
                    rec.append(0)
            pred.append(sum(rec)/min(N_TOP, len(p)))
        
    return np.mean(pred)

def read_db(file):
    data = pd.read_csv(file, sep = '\t', header = None)
    data.columns = ['user', 'song', 'rating']
    
    return data



N_TOP = 10
K_NEIGHBORS = 10
SIM_FUNCTION = cosine_similarity

# triplet_file = 'train_triplets.txt'
        
# dataset_raw = pd.read_csv('ratings_10000.csv')


# triplet_dataset, triplet_test = train_test_split(dataset_raw, test_size = 0.2)


triplet_train = pd.read_csv(filepath_or_buffer='Test/msd_train_visible.txt', sep='\t', header=None)
triplet_train.columns = ['user', 'song', 'rating']
train1, train2 = train_test_split(triplet_train, test_size = 0.1)

triplet_dataset = pd.read_csv(filepath_or_buffer='Test/msd_test_visible.txt', sep='\t', header=None)
triplet_dataset.columns = ['user', 'song', 'rating']

# triplet_dataset = pd.concat([triplet_dataset, train2])


triplet_test = pd.read_csv(filepath_or_buffer='Test/msd_test_hidden.txt', sep='\t', header=None)
triplet_test.columns = ['user', 'song', 'rating']





song_list = list(triplet_dataset['song'].unique())
song_list.sort()
# song_dict = {str(song_list[i]):str(i) for i in range(len(song_list))}
user_list = list(triplet_dataset['user'].unique())
user_list.sort()
# user_dict = {str(user_list[i]):str(i) for i in range(len(user_list))}

test_user_ids = list(triplet_test['user'].unique())
test_user_ids.sort()

db = get_user_song_rating_dct(triplet_dataset)
print('done db')

sim_db = get_similarity_dict(db, test_user_ids, SIM_FUNCTION)
print('done similarity')

mse = predict_ratings(db, sim_db, song_list, user_list, triplet_test, N_TOP, K_NEIGHBORS)



print('User based mse:', mse)




