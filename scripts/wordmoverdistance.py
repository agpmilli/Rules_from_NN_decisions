from itertools import product
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import euclidean
import pulp
import torch
from torch.autograd import Variable


singleindexing = lambda m, i, j: m*i+j
unpackindexing = lambda m, k: (k/m, k % m)


def tokens_to_fracdict(tokens):
    cntdict = defaultdict(lambda : 0)
    for token in tokens:
        cntdict[token] += 1
    totalcnt = sum(cntdict.values())
    return {token: float(cnt)/totalcnt for token, cnt in cntdict.items()}

def token2embed(token, model, text_field):
    index_word = text_field.vocab.stoi[token]
    x = torch.LongTensor([index_word])
    x = x.cuda()
    x = Variable(x)
    embed = model.embed(x)
    floatlist = list(embed.cpu().data.numpy())
    return floatlist

# use PuLP
def word_mover_distance_probspec(first_sent_tokens, second_sent_tokens, wvmodel, text_field, lpFile=None):
    
    '''toremove=[]
    for token in first_sent_tokens:
        if token not in text_field.vocab:
            toremove.append(token)
    print(toremove)
    for token in toremove:
        first_sent_tokens.remove(token)
    toremove=[]
    for token in second_sent_tokens:
        if token not in text_field.vocab:
            toremove.append(token)
    print(toremove)
    for token in toremove:
        second_sent_tokens.remove(token)'''
        
    all_tokens = list(set(first_sent_tokens+second_sent_tokens))
    
    #print(all_tokens)
    
    #for token in all_tokens:
    #    print(token)
    #    print(text_field.vocab.stoi[token])
    
    wordvecs = {token: token2embed(token,wvmodel, text_field) for token in all_tokens}

    first_sent_buckets = tokens_to_fracdict(first_sent_tokens)
    second_sent_buckets = tokens_to_fracdict(second_sent_tokens)

    T = pulp.LpVariable.dicts('T_matrix', list(product(all_tokens, all_tokens)), lowBound=0)

    prob = pulp.LpProblem('WMD', sense=pulp.LpMinimize)
    prob += pulp.lpSum([T[token1, token2]*euclidean(wordvecs[token1], wordvecs[token2])
                        for token1, token2 in product(all_tokens, all_tokens)])
    for token2 in second_sent_buckets:
        prob += pulp.lpSum([T[token1, token2] for token1 in first_sent_buckets])==second_sent_buckets[token2]
    for token1 in first_sent_buckets:
        prob += pulp.lpSum([T[token1, token2] for token2 in second_sent_buckets])==first_sent_buckets[token1]

    if lpFile!=None:
        prob.writeLP(lpFile)
    
    #print(prob)

    prob.solve()

    return prob


def word_mover_distance(first_sent_tokens, second_sent_tokens, wvmodel, lpFile=None):
    prob = word_mover_distance_probspec(first_sent_tokens, second_sent_tokens, wvmodel, lpFile=lpFile)
    return pulp.value(prob.objective)


# example: tokens1 = ['american', 'president']
#          tokens2 = ['chinese', 'chairman', 'king']