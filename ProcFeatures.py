import numpy as np
import os
import json
import glob 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_features(kolmo_f, threshold, layer, clss, cl):
    new_dic={}
    features=[]
    for line in open(kolmo_f, 'r'):
        tw = json.loads(line)
        new_dic[list(tw.keys())[0]] = list(tw.values())[0]
    words = [k for k,v in new_dic.items() if v < threshold]

    for c in clss:
        if c in words:
            words.remove(c)

    for w in words:
        features.append(layer+w) 
    if cl:
        features.append(cl)
    print("Number of words as input features:", len(features))
    return features

def feature_vectors(json_fs, word_features):
    labels = []
    ids = []
    all_arrs = []
    for f in json_fs:  #each file is a user
        features_list = []
        for line in open(f, 'r'):
            emb = json.loads(line)
            n_id = list(emb.keys())[0]
            labels.append(emb[n_id]['label'])
            ids.append(n_id)

            for feat in word_features:
                if feat in emb[n_id].keys():
                    features_list.append(np.array(emb[n_id][feat]))

        features_arr = np.mean(np.array(features_list, dtype='object'), 0).reshape(1, -1) 
        all_arrs.append(features_arr)
    all_arrs = np.concatenate(np.array(all_arrs, dtype='object'), 0) #(200,768) 
    # print(all_arrs.shape) 

    final_arr = np.insert(all_arrs, 0, np.array(labels, dtype='object').reshape(1, -1), axis=1)
    final_arr = np.insert(final_arr, 1, np.array(ids, dtype='object').reshape(1, -1), axis=1)

    return pd.DataFrame(final_arr)

def prepare_data(df):
    X, y = df.values[:, 2:].astype('float32'), df.values[:, 0].astype(int)

    standard = MinMaxScaler() #(feature_range=(0, 1))
    X = standard.fit_transform(X)  #normalized
    return X, y



