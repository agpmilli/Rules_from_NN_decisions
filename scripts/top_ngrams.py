"""
  Train LDA model using https://pypi.python.org/pypi/lda,
  and visualize in 2-D space with t-SNE.
"""

import os
import time
import lda
import random
import argparse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import bokeh.plotting as bp
from bokeh.plotting import save
from bokeh.models import HoverTool
import re

def clean_string_to_list(str_):
    wordsnoquote = str_.replace("'","").replace('"','')
    wordsnocomma = re.sub(r'(?!(([^"]*"){2})*[^"]*$),', '', wordsnoquote)
    listofwords = wordsnocomma.replace("[","").replace("]","").replace(" ", "").split(",")
    return list(filter(None, listofwords))

t0 = time.time()
# load up data
print("Loading data...")
data = []
with open('/mnt/storage01/milliet/data/ngrams/clean-ngrams-score-9500.csv', 'r') as csvfile:
  lines = csvfile.readlines()
  for line in lines:
      data.append(line.split('\sep'))


data_threshold = data
#percent_threshold = 0.99
#data_threshold = list(elem for elem in data if float(elem[-1])>percent_threshold)

print("Get ngrams, labels and scores...")
ngrams_data = list(elem[0] for elem in data_threshold)
label_data = list(elem[2] for elem in data_threshold)
score_data = list(float(elem[-1]) for elem in data_threshold)
#print("Number of N-grams with score > " + str(percent_threshold)+ " : " + str(len(score_data)))
print("Number of N-grams with score == 1.0 : " + str(score_data.count(1.0)))

label1 = "anti-trump"
label2 = "pro-trump"

top50pro = []
top50anti = []

topx = 1000

print("Get top n-grams...")
for i in range(len(ngrams_data)):
    if label_data[i]==label1:
        if len(top50pro)==topx:
            top50pro.sort(key=lambda x: x[1])
            min_pro = top50pro[0]
            if float(score_data[i])>min_pro[1]:
                top50pro.remove(min_pro)
                top50pro.append([ngrams_data[i],score_data[i]])
        else:
            top50pro.append([ngrams_data[i],score_data[i]])
    else:
        if len(top50anti)==topx:
            top50anti.sort(key=lambda x: x[1])
            min_anti = top50anti[0]
            if float(score_data[i])>min_anti[1]:
                top50anti.remove(min_anti)
                top50anti.append([ngrams_data[i],score_data[i]])
        else:
            top50anti.append([ngrams_data[i],score_data[i]])
  
  
top50pro.sort(key=lambda x: -x[1])
top50anti.sort(key=lambda x: -x[1])
#print("TOP 50 PROS:")
#for ngramscore in top50pro:
#    print(ngramscore)     
#print("TOP 50 ANTIS:")     
#for ngramscore in top50anti:
#    print(ngramscore)   

print("Write file")
with open('/mnt/storage01/milliet/data/ngrams/ngrams_top' + str(topx) + '.txt', 'w') as file:
    file.write("TOP " + str(topx) + " Pro-Trump with score :\n")
    for ngramscore in top50pro:
        ngram = ngramscore[0]
        score = ngramscore[1]
        file.write(str(ngram) + " | Score : " + str(score) + "\n")
    file.write("\n\n\nTOP " + str(topx) + " Anti-Trump with score :\n")
    for ngramscore in top50anti:
        ngram = ngramscore[0]
        score = ngramscore[1]
        file.write(str(ngram) + " | Score : " + str(score) + "\n")