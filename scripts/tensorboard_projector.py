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
import re

parser = argparse.ArgumentParser(description='N-grams Score')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-momentum', type=float, default=0.9, help='momentum used for SGD [default: 0.9]')
parser.add_argument('-optim', type=str, default="adam", help='optimizer [default: adam]')
parser.add_argument('-min-freq', type=int, default=0, help='min freq of a word to be part of vocabulary [default: 0]')
parser.add_argument('-word-vectors', type=str, default="glove.twitter.27B.50d", help='use pretrained embedding [default: "glove.twitter.27B.50d"]')
parser.add_argument('-word-neighbors', type=str, default="none", help='check nearest value to word in embedding [default: "none"]')
parser.add_argument('-vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
parser.add_argument('-epochs', type=int, default=1024, help='number of epochs for train [default: 1024]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-load-dir', type=str, default='/mnt/storage01/milliet/snapshot', help='where to load the snapshot')
parser.add_argument('-save-dir', type=str, default='/mnt/storage01/milliet/embedding', help='where to save the snapshot')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch' )
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')#
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=100, help='number of embedding dimension [default: 128]')#
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')#
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu' )
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-state-dict', type=str, default="2018-01-12_12-13-31dataset_big_dev_data_small_kernel_num_100_dropout_0.25_embed_dim_100_optim_adam_lr_0.001_minfreq_10_pre_embed_glove.twitter.27B.100d_bias=False", help='filename of model state-dict [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-data', type=str, default="big", help='small or big [default: small]')
parser.add_argument('-dev-data', type=str, default="small", help='small or big [default: small]')
parser.add_argument('-early-stop', type=int, default=500, help='number of epochs without amelioration before stop [default: 20]')
parser.add_argument('-n3gram-file', type=str, default='/mnt/storage01/milliet/big/trigrams.csv', help='file containing 3-grams [default: 3-grams]')
parser.add_argument('-n4gram-file', type=str, default='/mnt/storage01/milliet/big/fourgrams.csv', help='file containing 4-grams [default: 4-grams]')
parser.add_argument('-n5gram-file', type=str, default='/mnt/storage01/milliet/big/fivegrams.csv', help='file containing 5-grams [default: 5-grams]')
parser.add_argument('-ngram-num', type=int, default=0, help='number of ngrams to be selected from file [default: 0]')
parser.add_argument('-ngram-batch-num', type=int, default=1, help='number of ngrams to be selected in each batch [default: 10]')
parser.add_argument('-threshold', type=float, default=0.9, help='threshold for selecting ngram [default: 0.9]')
args = parser.parse_args()

# load up data
data = []
with open('/mnt/storage01/milliet/data/ngrams/clean-ngrams-score-9500.csv', 'r') as csvfile:
    lines = csvfile.readlines()
    for line in lines:
        data.append(line.split('\sep'))

threshold = 0.9999
bests = list(elem for elem in data if float(elem[-1])>threshold)



print("N-grams with score > " + str(threshold) + ": " + str(len(bests)))

if len(bests)>0:
    args.summary_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+"threshold_"+str(threshold)+"_model_dict_"+args.state_dict)
    writer = SummaryWriter(log_dir=args.summary_dir) 
    
    x_data = list(re.sub("\s+", ",", elem[1][2:-2]).split(',') for elem in bests)
    
    i = 0
    for elem in x_data:
        sub = elem
        if elem[0]=='':
            sub = elem[1:]
        if elem[-1]=='':
            sub = sub[:-1]
        x_data[i] = [float(i) for i in sub]
        i+=1
    
    i = 0
    for elem in x_data:
        if len(elem)!=100:
            print(elem)
            print(len(elem))
            print(i)
            break
        i+=1
    
    embeds = torch.FloatTensor(x_data)
    #embeds = torch.cat(seq)
    #embeds = embeds.data
    
    ngrams = [il[0] + "_" + il[2] for il in bests]
    
    #imgs = [torch.FloatTensor([il[3]]) for il in bests]
    #img = torch.cat(imgs)
    #img = img.data
    
    writer.add_embedding(embeds, metadata=ngrams, tag='ngrams score')