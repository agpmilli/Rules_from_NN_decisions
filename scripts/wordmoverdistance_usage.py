
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
from random import shuffle

from wordmoverdistance import word_mover_distance_probspec
import pulp

parser = argparse.ArgumentParser(description='N-grams Score')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-momentum', type=float, default=0.9, help='momentum used for SGD [default: 0.9]')
parser.add_argument('-optim', type=str, default="adam", help='optimizer [default: adam]')
parser.add_argument('-min-freq', type=int, default=10, help='min freq of a word to be part of vocabulary [default: 0]')
parser.add_argument('-word-vectors', type=str, default="glove.twitter.27B.100d", help='use pretrained embedding [default: "glove.twitter.27B.50d"]')
parser.add_argument('-word-neighbors', type=str, default="none", help='check nearest value to word in embedding [default: "none"]')
parser.add_argument('-vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
parser.add_argument('-epochs', type=int, default=5000, help='number of epochs for train [default: 1024]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-load-dir', type=str, default='/mnt/storage01/milliet/snapshot', help='where to load the snapshot')
parser.add_argument('-save-dir', type=str, default='/mnt/storage01/milliet/data', help='where to save the snapshot')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch' )
# model
parser.add_argument('-dropout', type=float, default=0.25, help='the probability for dropout [default: 0.5]')#
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=100, help='number of embedding dimension [default: 128]')#
parser.add_argument('-kernel-num', type=int, default=1000, help='number of each kind of kernel')#
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu' )
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-state-dict', type=str, default="2018-01-15_18-54-04dataset_big_dev_data_big_kernel_num_1000_dropout_0.25_embed_dim_100_optim_adam_lr_0.001_minfreq_10_pre_embed_glove.twitter.27B.100d_bias=False", help='filename of model state-dict [default: None]')

parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-data', type=str, default="big", help='small or big [default: small]')
parser.add_argument('-dev-data', type=str, default="big", help='small or big [default: small]')
parser.add_argument('-early-stop', type=int, default=500, help='number of epochs without amelioration before stop [default: 20]')
parser.add_argument('-n3gram-file', type=str, default='/mnt/storage01/milliet/data/big/trigrams.csv', help='file containing 3-grams [default: 3-grams]')
parser.add_argument('-n4gram-file', type=str, default='/mnt/storage01/milliet/data/big/fourgrams.csv', help='file containing 4-grams [default: 4-grams]')
parser.add_argument('-n5gram-file', type=str, default='/mnt/storage01/milliet/data/big/fivegrams.csv', help='file containing 5-grams [default: 5-grams]')
parser.add_argument('-test-file', type=str, default='/mnt/storage01/milliet/data/big/test_processed.csv', help='file containing test tweets [default: test_tweets]')
parser.add_argument('-ngram-num', type=int, default=0, help='number of ngrams to be selected from file [default: 0]')
parser.add_argument('-ngram-batch-num', type=int, default=1, help='number of ngrams to be selected in each batch [default: 10]')
parser.add_argument('-threshold', type=float, default=0.9, help='threshold for selecting ngram [default: 0.9]')
args = parser.parse_args()

def dataload_trump(text_field, label_field, **kargs):
    print("Data loading")
    size = args.data
    dev_size = args.dev_data
    train_data, dev_data, test_data = data.TabularDataset.splits(
            path='/mnt/storage01/milliet/data/', train=size+'/train_processed.tsv',
            validation=dev_size+'/validate_processed.tsv', test=size+'/test_processed.tsv', format='tsv',
            fields=[('label', label_field),('text', text_field)])
    return train_data, dev_data, test_data

text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_data, dev_data, test_data = dataload_trump(text_field, label_field)

vectors=None if args.word_vectors == "none" else args.word_vectors
text_field.build_vocab(train_data, dev_data, test_data, min_freq=args.min_freq, vectors=vectors)

label_field.build_vocab(train_data)

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                            (train_data, dev_data, test_data),
                            batch_size=args.batch_size, sort_key=lambda x: len(x.text), device=args.device, repeat=False)


args.epoch_size = len(dev_iter.dataset)
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab)-1

args.cuda = (not args.no_cuda) and  torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

# model
if args.snapshot is None:
    print("Loading model")
    twitter = model.twitter_Text(args)
    if args.state_dict is not None:
        print('Loading state dict from [%s]...' % args.state_dict) 
        # load the new state dict
        twitter.load_state_dict(torch.load(args.load_dir + "/" + args.state_dict + "/model"))

if args.cuda:# Set the device to 1
    torch.cuda.set_device(args.device)
    twitter = twitter.cuda()
    
# Load n-grams
with open("/mnt/storage01/milliet/data/ngrams/clean-topic-100.txt", "r") as textfile:
    lines = textfile.readlines()

ngrams = {}
for line in lines:
    split = line.rsplit(',', 1)
    ngrams[split[0]]= split[1]

num_occ = 50
print("Getting sample (" + str(num_occ) + " of each label)")
label1="pro-trump"
label2="anti-trump"
all_nodes = []
all_labels = []
pro=0
anti=0

'''keys =  list(ngrams.keys())
shuffle(keys)
ngrams_shuffle = {}
for key in keys:
    ngrams_shuffle[key] = ngrams[key] 

for key, value in ngrams_shuffle.items():
    clean_value = value.replace("\n", "")
    if pro==num_occ and anti==num_occ:
        break
    if clean_value==label1 and pro<num_occ:
        all_nodes.append(key)
        all_labels.append(label1)
        pro+=1
    elif clean_value==label2 and anti<num_occ:
        all_nodes.append(key)
        all_labels.append(label2)
        anti+=1'''

for key, value in ngrams.items():
    clean_value = value.replace("\n", "")
    if clean_value==label1:
        all_nodes.append(key)
        all_labels.append(label1)
    elif clean_value==label2:
        all_nodes.append(key)
        all_labels.append(label2)

print(len(all_nodes))
print("Compute word mover distance between each pair")
all_distances = []
i=0
for key1 in all_nodes:
    print(str(i) + " / " + str(len(all_nodes)))
    i+=1
    j=0
    for key2 in all_nodes:
        print(str(j) + " / " + str(len(all_nodes)),end='\r')
        j+=1
        #print(key1)
        #print(key2)
        words1 = key1.split()
        words2 = key2.split()
        print(words1)
        print(words2)
        prob = word_mover_distance_probspec(words1, words2, twitter, text_field)
        all_distances.append([key1, key2, pulp.value(prob.objective)])
    
print("DONE")

with open('/mnt/storage01/milliet/data/wordmover/topic_nodes.txt', 'w') as nodefile:
    for i in range(len(all_nodes)):
        nodefile.write("%s|%s\n" % (all_nodes[i],all_labels[i]))

with open('/mnt/storage01/milliet/data/wordmover/topic_edges.txt', 'w') as edgefile:
    for i in range(len(all_distances)):
        edgefile.write("%s|%s|%s\n" % (all_distances[i][0],all_distances[i][1], all_distances[i][2]))
  

'''ngram1 = ['democracy!', '<hashtag>', 'maga!']
ngram2 = ['followbackresistance', '<hashtag>', 'fbr']
prob = word_mover_distance_probspec(ngram1, ngram1, wvmodel, text_field)
print(pulp.value(prob.objective))
for v in prob.variables():
    if v.varValue!=0:
        print(v.name, '=', v.varValue)'''

