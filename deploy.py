import glob 
import Embeddings
import ProcFeatures
import ModelSvm
import os
from os import path
import torch 
import ParseData
import shutil

bert_emb = input("Do you want to start from scratch and extract the bert embeddings too? \nTo answer type: yes / no\n")
print()
es_en = input("For Spanish, type: es \nFor English, type: en\n")
if bert_emb == 'yes':
    train_folder = input("Type the name of the directory where the TRAIN set is located. \nFor example: \n./media/training_set/pan21-author-profiling-training-2021-03-14\n")
    print()
    test_folder = input("Type the name of the directory where the TEST set is located. \nFor example: \n./media/training_set/pan21-author-profiling-test-without-gold\n")
    print()
else:
    train_folder = None
    test_folder = None

languages = {
	'en':{
    'train_emb':"./en_train_embed/", 
    'parsing_train': str(train_folder)+'/en/',
    'kolmogorov': 'en_kolmogorov_avg.json',
    'save_model':'./model/english_model.joblib', 
    'kernel':'rbf',
    'C':14.074941979133145, 
    'gamma':0.009556817583272984, 
    'degree':None, 
    'test_emb' : "./en_test_embed/",
    'parsing_test': str(test_folder)+'/en/',
    'predict':'./predictions/en/', 
    'threshold': 0.998, 
    'cl_token':'cls_12hidden', 
    'layer': '12th_'
    },
	'es':{
    'train_emb':"./es_train_embed/", 
    'parsing_train': str(train_folder)+'/es/',
    'save_model':'./model/spanish_model.joblib', 
    'kernel': 'poly',
    'C':7.358801222763173, 
    'gamma':0.028509010421467293, 
    'degree':1.2859653546747762,
    'test_emb' : "./es_test_embed/", 
    'parsing_test': str(test_folder)+'/es/',
    'predict':'./predictions/es/', 
    }
}

print("Language selected:", es_en)

print("<<<<START TRAINING STEP>>>>")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device", device)

if bert_emb == 'yes':  #start from scratch
    folder_embeds = languages[es_en]['train_emb']
    if path.isdir(folder_embeds):
        shutil.rmtree(folder_embeds)
    df_parsed = ParseData.parse_tweets(languages[es_en]['parsing_train'], es_en, None)
    Embeddings.get_embeddings(df_parsed, 'train', es_en, folder_embeds, device)

json_fs = glob.glob("./"+languages[es_en]['train_emb']+"/*.json")
clss = ["cls_12hidden", "cls_11hidden"] 

if es_en == "en": 
    layer = languages[es_en]['layer']
    kolmo_f = languages[es_en]['kolmogorov']
    cl_token = languages[es_en]['cl_token'] 
    threshold = languages[es_en]['threshold']
    print("threshold", threshold)
    features = ProcFeatures.get_features(kolmo_f, threshold, layer, clss, cl_token)
else:
	features = ["cls_11hidden"]

df = ProcFeatures.feature_vectors(json_fs, features)
data, targets = ProcFeatures.prepare_data(df)

C = languages[es_en]['C']
gamma = languages[es_en]['gamma']
degree = languages[es_en]['degree']
kernel = languages[es_en]['kernel']
save_model = languages[es_en]['save_model']
model = ModelSvm.svm(C, gamma, degree, kernel)
ModelSvm.save_final_model(model, data, targets, save_model)

print("<<<<START TESTING STEP>>>>")

if bert_emb == 'yes':
    df_parsed = ParseData.parse_tweets(languages[es_en]['parsing_test'], es_en, 'test')
    folder_embeds = languages[es_en]['test_emb']
    if path.isdir(folder_embeds):
        shutil.rmtree(folder_embeds)
    Embeddings.get_embeddings(df_parsed, 'test', es_en, folder_embeds, device)

json_fs = glob.glob(languages[es_en]['test_emb']+"*.json")

df = ProcFeatures.feature_vectors(json_fs, features)

y_predict = ModelSvm.predic_labels(json_fs, df, es_en, save_model, languages[es_en]['predict'])
