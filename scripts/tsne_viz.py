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
#percent_threshold = 0.999
#data_threshold = list(elem for elem in data if float(elem[-1])>percent_threshold)

print("Get ngrams...")
ngrams_data = list(clean_string_to_list(elem[0]) for elem in data_threshold)
y_data = list(elem[2] for elem in data_threshold)

processed_tweet = [' '.join(x) for x in ngrams_data]
raw_tweet = processed_tweet # testing
tweet_label = ["pro-trump" if y=="anti-trump" else "anti-trump" for y in y_data]
'''x_data = list(re.sub("\s+", ",", elem[1][2:-2]).split(',') for elem in data_threshold)
y_data = list(elem[2] for elem in data_threshold)


j = 0
for elem in x_data:
  if elem[0]=='':
      sub = elem[1:]
  if elem[-1]=='':
      sub = sub[:-1]
  x_data[j] = [float(i) for i in sub]
  j+=1

i = 0
for elem in x_data:
  if len(elem)!=100:
      print(elem)
      print(len(elem))
      print(i)
      break
  i+=1

x_data = np.array(x_data)'''

# convert image data to float64 matrix. float64 is need for bh_sne
#x_data = np.asarray(x_data).astype('float64')
#x_data = x_data.reshape((x_data.shape[0], -1))

# For speed of computation, only run on a subset
#n = 20000
#x_data = x_data[:n]

##############################################################################
# train LDA
t1 = time.time()
# ignore terms that have a document frequency strictly lower than 5
cvectorizer = CountVectorizer(min_df=5)
cvz = cvectorizer.fit_transform(processed_tweet)

n_topics_list = [100] #[10,30,50,100]
perplexities = [100] #[5,30,50,100]
n_iter = 200

label1 = "pro-trump"
label2 = "anti-trump"

do=1
while do==1:
  for n_topics in n_topics_list:
    print("\n>>> LDA with {} topics\n".format(n_topics))
    lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter)
    X_topics = lda_model.fit_transform(cvz)
    
    t2 = time.time()
    print('\n>>> LDA training done; took {} mins\n'.format((t2-t1)/60.))
    
    '''np.save('lda_simple/lda_doc_topic_{}tweets_{}topics.npy'.format(
      X_topics.shape[0], X_topics.shape[1]), X_topics)
    np.save('lda_simple/lda_topic_word_{}tweets_{}topics.npy'.format(
      X_topics.shape[0], X_topics.shape[1]), lda_model.topic_word_)
    print '\n>>> doc_topic & topic word written to disk\n'''
    
    ##############################################################################
    # threshold and plot
    threshold = 0.2
    num_example = 10000
    n_top_words = 5
    all_top_words = 20
    num_qualified_tweet = len(processed_tweet)
    
    _idx = np.amax(X_topics, axis=1) > threshold  # idx of tweets that > threshold
    _topics = X_topics[_idx]
    
    _raw_tweet = np.array(raw_tweet)[_idx]
    _processed_tweet = np.array(processed_tweet)[_idx]
    _tweet_label = np.array(tweet_label)[_idx]
      
      
    for perplexity in perplexities:
      # t-SNE: 50 -> 2D
      tsne_model = TSNE(n_components=2, verbose=1, perplexity=perplexity, random_state=0, angle=.99,
                        init='pca')
    
      tsne_lda = tsne_model.fit_transform(_topics[:num_example])
      
      t3 = time.time()
      print('\n>>> t-SNE transformation done; took {} mins\n'.format((t3-t2)/60.))
      
      # find the most probable topic for each tweet
      _lda_keys = []
      for i, tweet in enumerate(_raw_tweet):
        _lda_keys += _topics[i].argmax(),
      
      # generate random hex color
      colormap = []
      for i in range(X_topics.shape[1]):
        r = lambda: random.randint(0, 255)
        colormap += ('#%02X%02X%02X' % (r(), r(), r())),
      colormap = np.array(colormap)
      
      labelmap = []
      
      
      # show topics and their top words
      topic_summaries = []
      topics_all_summaries = []
      topic_word = lda_model.topic_word_  # get the topic words
      vocab = cvectorizer.get_feature_names()
      for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        topic_summaries.append(' '.join(topic_words))
        all_topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(all_top_words+1):-1]
        topics_all_summaries.append(' '.join(all_topic_words))
      
      # use the coordinate of a random tweet as string topic string coordinate
      #topic_coord = np.empty((X_topics.shape[1], 2)) * np.nan
      #for topic_num in _lda_keys:
        #if not np.isnan(topic_coord).any():
        #  break
        #topic_coord[topic_num] = tsne_lda[_lda_keys.index(topic_num)] #[0, topic_num*100]
      
      # plot
      
      title = "t-SNE visualization of LDA model trained on {} tweets, {} topics, {} perplexity " \
              "thresholding at {} topic probability, {} iter ({} data points and " \
              "top {} words)".format(num_qualified_tweet, n_topics, perplexity, threshold,
                                     n_iter, num_example, n_top_words)
      
      plot_lda = bp.figure(plot_width=1920, plot_height=1080,
                           title=title,
                           tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                           x_axis_type=None, y_axis_type=None, min_border=1)
      
      for lab in _tweet_label:
        if lab==label1:
          labelmap.append('#FF0000')
        else:
          labelmap.append('#0000FF')
      
      contour = []
      for lab in _tweet_label:
          contour.append('#000000')
          
      
      topickey = None     
      for key, summary in enumerate(topics_all_summaries):
        words = summary.split()
        if ('felix' in words) and ('sater' in words) and ('muslims' in words):
          print(str(key) + " : " + str(summary))
          topickey = key
          do=0
          
      print("Get label and number of tweets for each topic")
      topic_words_list = []
      _topic_label = []
      _topic_labelscore = []
      _topic_occurences = []
      _topic_all = {}
      i = 0
      topic_ngrams = []
      topic_labels = []
      for key in _lda_keys:
        if i % 10000 == 0:
          print(str(i) + " / " + str(len(_lda_keys)), end="\r")
        if key==topickey:
          topic_ngrams.append(_raw_tweet[i])
          topic_labels.append(_tweet_label[i])
        
        topic_words_list.append(topic_summaries[key])
        if key in _topic_all:
          if _topic_all[key][0] > _topic_all[key][1]:
            _topic_label.append(label1)
            _topic_labelscore.append(_topic_all[key][0])
          else:
            _topic_label.append(label2)
            _topic_labelscore.append(_topic_all[key][1])
          _topic_occurences.append(_topic_all[key][0] + _topic_all[key][1])
        else:
          #print("Topic " + str(key))
          label1Value = 0
          label2Value = 0
          indices = [i for i, x in enumerate(_lda_keys) if x == key]
          for index in indices:
            if _tweet_label[index]==label1:
              label1Value+=1
            else:
              label2Value+=1
      
          if label1Value>label2Value:
            _topic_labelscore.append(label1Value)
            _topic_label.append(label1)
          else:
            _topic_labelscore.append(label2Value)
            _topic_label.append(label2)
      
          _topic_occurences.append(label1Value + label2Value)
          _topic_all[key]=[label1Value, label2Value]
        i+=1
      print("=======Done=======")
      
      with open('/mnt/storage01/milliet/data/ngrams/clean-topic-'+str(perplexity)+'.txt', 'w') as file:
        for i in range(len(topic_ngrams)):
          file.write("%s,%s\n" % (topic_ngrams[i],topic_labels[i]))
          
      source = bp.ColumnDataSource(data=dict(
                          x=tsne_lda[:, 0],
                          y=tsne_lda[:, 1],
                          tweet=_raw_tweet[:num_example],
                          topic_key=_lda_keys[:num_example],
                          topic_words = topic_words_list[:num_example],
                          label=_tweet_label[:num_example],
                          topic_label = _topic_label[:num_example],
                          topic_labelscore = _topic_labelscore[:num_example],
                          topic_occurences = _topic_occurences[:num_example],
                          color=labelmap[:num_example],#colormap[_lda_keys][:num_example],
                          lineColor=contour[:num_example]
                       ))
      
      plot_lda.scatter(x='x', y='y',
                       source=source,
                       color='color',
                       line_color='lineColor',
                       radius=0.3)
      
      # plot crucial words
      #for i in range(X_topics.shape[1]):
        #plot_lda.text(topic_coord[i, 0], topic_coord[i, 1], [str(i) + " : " + topic_summaries[i]])
        
      hover = plot_lda.select(dict(type=HoverTool))
      hover.tooltips = {
        "n-gram": "@tweet",
        "topic": "@topic_key : @topic_words",
        "label": "@label",
        "topic_label": "@topic_label : @topic_labelscore / @topic_occurences"}
      
      save(plot_lda, 'tsne_lda_color2_viz_num_tweets_{}_n_topics_{}_threshold_{}_n_iter_{}_n_example_{}_ntopwords_{}_perplexity_{}.html'.format(
        num_qualified_tweet, n_topics, threshold, n_iter, num_example, n_top_words, perplexity))


t4 = time.time()
print('\n>>> whole process done; took {} mins\n'.format((t4-t0)/60.))