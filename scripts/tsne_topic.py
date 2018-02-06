
#! /usr/bin/env python
import os
import argparse
import datetime
import torch
from torch.autograd import Variable
import torchtext.data as data
import torchtext.datasets as datasets
from torchtext.vocab import Vectors, GloVe
import model
from tensorboardX import SummaryWriter
import numpy as np
import math
import time
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


# load up data
print("Loading data...")
data = []
with open('/mnt/storage01/milliet/data/ngrams/clean-ngrams-score-9500.csv', 'r') as csvfile:
    lines = csvfile.readlines()
    for line in lines:
        data.append(line.split('\sep'))
        
ngrams_data = list(clean_string_to_list(elem[0]) for elem in data)


print("Embeddings...")
embeds = []
i=0
for elem in data:
    elem_list = re.sub(' +', '|', elem[1].replace("'","").replace('"','').replace("[","").replace("]","")).split("|")
    elem_list = filter(None, elem_list)
    list_=[]
    for elemstr in elem_list:
      list_.append(float(elemstr))
    embeds.append(list_)
        
    
# Load n-grams
print("Loading ngrams...")
with open("/mnt/storage01/milliet/data/ngrams/clean-topic-100.txt", "r") as textfile:
    #with open("/mnt/storage01/milliet/wordmover/topic50_nodes100.txt", "r") as textfile:
    lines = textfile.readlines()

ngrams = []
labels = []
for line in lines:
    split = line.split(',')
    ngrams.append(split[0])
    labels.append(split[1])

#print(ngrams)

print("Get embedding of ngrams")
embed_dict = {}
i = 0
j = 0
for ngram in ngrams_data:
    ngram_txt = " ".join(ngram)
    if ngram_txt in ngrams:
        print(j, end='\r')
        embed_dict[ngram_txt] = embeds[i]
        j+=1
    i+=1    

embed_tsne = []
for ngram in ngrams:
    embed_tsne.append(embed_dict[ngram])
 
'''i = 0
for elem in embed_tsne:
  i+=1
  if len(elem)!=100:
      print(elem)
      print(i)'''
      
#print(embed_tsne[:5])   
X = np.array(embed_tsne)
#print(X[:5])
Y = np.asarray(embed_tsne)
#print(Y[:5])
##############################################################################
# threshold and plot
#threshold = 0.2
#num_example = 10000
#n_top_words = 5
#all_top_words = 20
#num_qualified_tweet = len(processed_tweet)

#_idx = np.amax(X_topics, axis=1) > threshold  # idx of tweets that > threshold
#_topics = X_topics[_idx]

#_raw_tweet = np.array(raw_tweet)[_idx]
#_processed_tweet = np.array(processed_tweet)[_idx]
#_tweet_label = np.array(tweet_label)[_idx]

# t-SNE: 50 -> 2D
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(embed_tsne)

title = "t-SNE visualization of embedding of {} tweets in topic concerning 'Felix Sater'".format(len(ngrams))

plot_lda = bp.figure(plot_width=1920, plot_height=1080,
                     title=title,
                     tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                     x_axis_type=None, y_axis_type=None, min_border=1)

label1 = "pro-trump"
label2 = "anti-trump"

labelmap = []
for lab in labels:
  label = lab.rstrip()
  if label==label1:
    labelmap.append('#FF0000')
  else:
    labelmap.append('#0000FF')


print("=======Done=======")

#with open('/mnt/storage01/milliet/embedding/ngrams/topic.txt', 'w') as file:
#  for i in range(len(topic_ngrams)):
#    file.write("%s,%s\n" % (topic_ngrams[i],topic_labels[i]))
    
source = bp.ColumnDataSource(data=dict(
                    x=tsne_lda[:, 0],
                    y=tsne_lda[:, 1],
                    tweet=ngrams,
                    label=labels,
                    color=labelmap,
                 ))

plot_lda.scatter(x='x', y='y',
                 source=source,
                 color='color',
                 radius=0.3)

# plot crucial words
for i in range(len(ngrams)):
    plot_lda.text(tsne_lda[i, 0], tsne_lda[i, 1], [ngrams[i]])
    
hover = plot_lda.select(dict(type=HoverTool))
hover.tooltips = {
  "n-gram": "@tweet",
  "label": "@label"}

save(plot_lda, 'tsne_viz_in_topic.html'.format(
  len(ngrams)))
