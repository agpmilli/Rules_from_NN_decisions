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
parser.add_argument('-test-file', type=str, default='/mnt/storage01/milliet/data/big/test_processed.tsv', help='file containing test tweets [default: test_tweets]')
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


def getAllTweets(tweets):
    rows = []

    write = False
    i=0
    numbertowrite=10000

    tweets_sep = [item[:-1].split(' ') for item in tweets]

    size = len(tweets_sep)
    for tweet in tweets_sep:
        print(str(i) + " / " + str(size) + " scored yet", end="\r")
        indexes = []
        for word in tweet:
            #if word in text_field.vocab.stoi:
            index_word = text_field.vocab.stoi[word]
            indexes.append(index_word)
            #else:
            #    send=False
            #    break
        #if send:
        x = torch.LongTensor([indexes])
        x = x.cuda()
        x = Variable(x)
        score, embed = twitter.tweet_forward(x)
        
        score_val = score.data[0]
        embed_str = np.array2string(embed.data.cpu().numpy())
        embed_str = embed_str.replace('\n', ' ')

        row = list()
        row.append(str(tweet))
        row.append(embed_str)
        if score_val[0]>score_val[1]:
            row.append("pro-trump")
            row.append(str(score_val[0]))
        else:
            row.append("anti-trump")
            row.append(str(score_val[1]))
            
        rows.append(row)
        
        if len(rows)==numbertowrite:
            towrite = rows
            write=True
            rows = []
        
        if write:
            with open(args.save_dir + '/tweets_representation.csv', 'a+') as f:
                for row in towrite:
                    row_str = '\sep'.join(row)
                    f.write(row_str + "\n")
            write = False
        i+=1
    
    
    with open(args.save_dir + '/tweets_representation.csv', 'a+') as f:
        for row in rows:
            row_str = '\sep'.join(row)
            f.write(row_str + "\n")
    
    print("DONE")

tweets = []

with open(args.test_file) as f:
    if args.ngram_num!=0:
        tweets.extend([next(f) for x in range(args.ngram_num)])
    else:
        tweets.extend(list(f))

print('tweets start')
getAllTweets(tweets)
print('tweets done')
            
