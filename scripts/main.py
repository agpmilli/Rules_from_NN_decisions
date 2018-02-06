#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
from torchtext.vocab import Vectors, GloVe
import model
import train
from tensorboardX import SummaryWriter

import numpy as np
import math


parser = argparse.ArgumentParser(description='Twitter text classifier')
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
parser.add_argument('-test-interval', type=int, default=1000, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir-init', type=str, default='/mnt/storage01/milliet/snapshot', help='where to save the snapshot')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch' )
# model
parser.add_argument('-dropout', type=float, default=0.25, help='the probability for dropout [default: 0.25]')#
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=100, help='number of embedding dimension [default: 100]')#
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
parser.add_argument('-data', type=str, default="big", help='small or big [default: big]')
parser.add_argument('-dev-data', type=str, default="big", help='small or big [default: small]')
parser.add_argument('-early-stop', type=int, default=20, help='number of epochs without amelioration before stop [default: 20]')
parser.add_argument('-word-score', type=str, default=None, help='find score per n-grams [default: None]')
args = parser.parse_args()

def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""
    import os, errno
    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise

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
print("Number of class: " + str(args.class_num))

args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

#Add some informations in the folder name
args.save_dir = os.path.join(args.save_dir_init, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+"dataset_" + str(args.data)+"_dev_data_"+str(args.dev_data)+"_kernel_num_"+str(args.kernel_num)+"_dropout_"+str(args.dropout)+"_embed_dim_"+str(args.embed_dim) + "_optim_"+str(args.optim)+"_lr_"+str(args.lr)+"_minfreq_"+str(args.min_freq)+"_pre_embed_"+args.word_vectors+"_bias=False")


print("Parameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
if args.snapshot is None:
    print("Loading model")
    twitter = model.twitter_Text(args)
    if args.state_dict is not None:
        print('Loading state dict from [%s]...' % args.state_dict) 
        # load the new state dict
        twitter.load_state_dict(torch.load(args.save_dir_init + "/" + args.state_dict + "/model"))
    else:
        # Initialize weights
        if vectors:
            twitter.embed.weight.data.copy_(text_field.vocab.vectors)
    
else :
    print('Loading model from [%s]...' % args.snapshot)
    try:
        twitter = torch.load(args.snapshot)
    except :
        print("Sorry, This snapshot doesn't exist."); exit()

word_neighbor=None if args.word_neighbors == "none" else args.word_neighbors
if word_neighbor:
    index_word = text_field.vocab.stoi[word_neighbor]
    embed_word = twitter.embed.weight.data[index_word].numpy()
    top10val = [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf]
    top10ind = [0,0,0,0,0,0,0,0,0,0]
    min_id = 0
    ind = 0
    for vec in twitter.embed.weight.data:
      vec_np = vec.numpy()
      for i, val in enumerate(top10val):
        if np.linalg.norm(vec_np-embed_word) < val:
            top10val[i] = np.linalg.norm(vec_np-embed_word)
            top10ind[i] = ind
            break
      ind+=1

    print("Nearer words to ", word_neighbor, " : ")
    for ind in top10ind:
        print(ind, " : " , text_field.vocab.itos[ind])
    

if args.cuda:# Set the device to 1
    torch.cuda.set_device(args.device)
    twitter = twitter.cuda()

writer = SummaryWriter(log_dir=args.save_dir) 


# train or predict
if args.predict is not None:
    label = train.predict(args.predict, twitter, text_field, label_field, args.cuda)
    print('\n[Text]  {}[Label] {}\n'.format(args.predict, label))
elif args.test :
    try:
        train.eval(test_iter, twitter, args)
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else :
    try:
        train.train(train_iter, dev_iter, twitter, writer, args)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    