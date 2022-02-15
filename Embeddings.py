import numpy as np
import pandas as pd
from collections import defaultdict
from transformers import BertTokenizer, BertModel
import json
import torch
import utils
import pickle
import os
from os import path

def get_embeddings(df, label_test, es_en, folder_out, device):
  if es_en == "en":
    PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
    MAX_LEN=75
    new_dic={}
    for line in open("en_kolmogorov_avg.json", 'r'):
      tw = json.loads(line)
      new_dic[list(tw.keys())[0]] = list(tw.values())[0]
    words = [k for k,v in new_dic.items() if v < 0.998]
    
  else:
    PRE_TRAINED_MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-uncased'  
    MAX_LEN=60
    words = []

  if path.isdir(folder_out):
    pass
  else:
    os.makedirs(folder_out) 

  sentences = df.per_tw.values
  ids = df.id.values

  if label_test == 'test':
    labels = [0]*len(ids)
  else:
    labels = df.label.values

  tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
  model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, output_hidden_states=True).to(device)
  print(f"Start getting embeddings for {label_test} set")
  for i in range(len(ids)): 
    dic_emb = {}
    dic_emb[ids[i]] = defaultdict(list)
    dic_emb[ids[i]]["label"]=labels[i]
    print("USER N.:", i)
    for tw in sentences[i]: 

      encoding = tokenizer.encode_plus(
        tw,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
      )

      input_ids = encoding["input_ids"].to(device)
      attention_mask = encoding["attention_mask"].to(device)

      outputs = model(input_ids, attention_mask)

      dic_emb[ids[i]]['cls_12hidden'].append(np.array(outputs.last_hidden_state[:, 0, :].squeeze().detach().cpu()))
      dic_emb[ids[i]]['cls_11hidden'].append(np.array(outputs.hidden_states[11][:, 0, :].squeeze().detach().cpu()))
      #dic_emb[ids[i]]['cls_10hidden'].append(np.array(outputs.hidden_states[10][:, 0, :].squeeze().detach().cpu()))

      if es_en == "en":
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"].flatten())
        for idx, t in enumerate(tokens):
          if t in words:
            dic_emb[ids[i]]["12th_"+t].append(np.array(outputs.last_hidden_state[:, idx, :].detach().cpu()))                  
            #dic_emb[ids[i]]["11th_"+t].append(np.array(outputs.hidden_states[11][:, idx, :].detach().cpu()))        
            #dic_emb[ids[i]]["10th_"+t].append(np.array(outputs.hidden_states[10][:, idx, :].detach().cpu()))        
      else:
        pass
    for k, v in dic_emb[ids[i]].items():
      if k != 'label':
        dic_emb[ids[i]][k] = np.mean(v, 0).tolist()
    utils.write_as_json(folder_out+ids[i]+".json", dic_emb)