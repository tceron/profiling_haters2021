import os
import glob
from xml.dom import minidom
from collections import defaultdict
import json
import pandas as pd
import numpy as np
from collections import Counter
import xml.etree.ElementTree as ET


def parse_tweets(folder, lang, test):
  files=glob.glob(folder+"*xml")
  user_tws=defaultdict(list)
  for fi in files: 
    user_id = fi.split("/")[3].split(".")[0]

    mydoc = minidom.parse(fi)
    items = mydoc.getElementsByTagName('document')
   
    for elem in items:
      tweet = elem.firstChild.data
      tweet = tweet.replace("#HASHTAG#", "hashtag").replace("#USER#", "user").replace("#URL#", "url")
      user_tws[user_id].append(tweet)   #dictionary with user_id as key and list of tweets from each id as value

  data_dict = {'id': pd.Series(list(user_tws.keys())),
               'per_tw': pd.Series(user_tws.values())}
  df = pd.DataFrame(data_dict)

  if test == 'test':
    return df
  else:
    df_label = get_label(lang, folder)
    df = df.merge(df_label, how='left', on='id')
    return df


def get_label(lang, folder):
  if lang == "en":
    labels=folder+"truth.txt"
  else:
    labels=folder+"truth.txt"

  user_label={}  #dictionary for user_id and their respective labels
  for line in open(labels).readlines():
    user=line.strip("\n").split(":::")
    user_label[user[0]]=user[1]

  df_label = pd.DataFrame.from_dict(user_label, orient="index").reset_index()
  df_label.columns = ["id", 'label']
  return df_label