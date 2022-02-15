from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from joblib import dump, load
from os import path
import os
from collections import Counter
import xml.etree.ElementTree as ET

def svm(C, gamma, degree, kernel):
    if kernel == 'poly':
        return SVC(kernel= kernel, C=C, gamma=gamma, degree=degree, random_state=2)
    if kernel == 'linear':
        return SVC(kernel= kernel, random_state=2)
    else:
        return SVC(kernel= kernel, C=C, gamma=gamma, random_state=2) 

def save_final_model(model, data, targets, f_save_model):
    if path.isdir("./model"):
        pass
    else:
        os.makedirs("./model") 
    model.fit(data, targets)
    dump(model, f_save_model)
    cval = cross_val_score(model, data, targets, scoring='accuracy', cv=5)
    print("Performance with 5-fold cross-validation in the training set:", cval.mean())
    print(f"MODEL SAVED AS: {f_save_model}")

def prepare_data_ids(df):
    X = df.values[:, 2:].astype('float32')
    ids = df.values[:, 1].tolist()

    standard = MinMaxScaler() #(feature_range=(0, 1))
    X = standard.fit_transform(X)  #normalized

    return X, ids


def predic_labels(json_fs, df, lang, load_model, folder_predic):
    if path.isdir("./predictions"):
        pass
    else:
        os.makedirs("./predictions")
    if path.isdir("./predictions/"+lang):
        pass
    else:
        os.makedirs("./predictions/"+lang)

    X, ids = prepare_data_ids(df)

    model = load(load_model)
    y_predic = model.predict(X)
    print(Counter(y_predic))
    dic={}
    for i in range(len(X)):
        dic[ids[i]] = str(y_predic[i])

        output_file = open(folder_predic+ids[i]+".xml", 'wb')
        author = ET.Element('author')
        author.set('id', ids[i])
        author.set('lang', lang)
        author.set('type',str(y_predic[i]))
        mydata = ET.tostring(author)
        output_file.write(mydata)
        output_file.close()
    print("Find the classifications in the folder ./predictions =)")

    return dic