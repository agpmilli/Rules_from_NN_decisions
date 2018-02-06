import faiss
import re
import torch
import numpy as np

# load up data
data = []

names = ['9500'] #'5055','5560','6065','6570','7075','7580','8085','8590','9095','9500']

print("Loading data...")
for name in names:
    with open('/mnt/storage01/milliet/embedding/ngrams/ngrams-score-'+name+'.csv', 'r') as csvfile:
        lines = csvfile.readlines()
        for line in lines:
            data.append(line.split('\sep'))

print("Get embedding...")
x_data = list(re.sub("\s+", ",", elem[1][2:-2]).split(',') for elem in data)

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

x = np.array(x_data, dtype=np.float32)

ncentroids = 1024
niter = 20
verbose = True
d = x.shape[1]
print("K-Means for " + str(ncentroids) + " centroids and " + str(niter) + " iterations...")
kmeans = faiss.Kmeans(d, ncentroids, niter, verbose)
print("Training...")
kmeans.train(x)
print("Training done...")
print("Mapping...")
D, I = kmeans.index.search(x, 1)
i = 1
j=0
k=0
for indexes in I:
    if i in indexes:
        k+=1
    j+=1
        
print("Number of embedding classified : " + str(j))
print("Number of embedding in centroid "+ str(i) + " : " + str(k))