#! /usr/bin/env python
from nltk import ngrams
import csv

trigrams = []
fourgrams = []
fivegrams = []

#Ks = [3,4,5]

size = 'big'
path = '/mnt/storage01/milliet/data/'

rows = [row.rstrip('\n') for row in open(path+size+'/test_processed.tsv')]
nrows = len(rows)

i=1
print('Extract n-grams')
for row in rows:
    print(str(i)+' / '+str(nrows) + " rows treated", end='\r')
    sentence = row.split('\t')[1]
    #if "johnny manziel" in sentence:
    #    print(sentence)
    for n3gram in ngrams(sentence.split(), 3):
        #if "johnny manziel" in sentence:
        #    print(n3gram)
        #if n3gram[0]=='"us':
        #    print(sentence)
        #    print(n3gram)
        trigrams.append(n3gram)
    for n4gram in ngrams(sentence.split(), 4):
        fourgrams.append(n4gram)
    for n5gram in ngrams(sentence.split(), 5):
        fivegrams.append(n5gram)
    i+=1

print("Number of 3grams : " + str(len(trigrams)))
print("Number of 4grams : " + str(len(fourgrams)))
print("Number of 5grams : " + str(len(fivegrams)))

trigrams = list(set(trigrams))
fourgrams = list(set(fourgrams))
fivegrams = list(set(fivegrams))

print("Number of 3grams after removing duplicates: " + str(len(trigrams)))
print("Number of 4grams after removing duplicates: " + str(len(fourgrams)))
print("Number of 5grams after removing duplicates: " + str(len(fivegrams)))

print('3-grams')
print(trigrams[:10])       
print('4-grams')
print(fourgrams[:10])  
print('5-grams')
print(fivegrams[:10])


print('Write n-grams files')
with open(path+size+"/trigrams.txt", "w", newline="") as f:
    for trigram in trigrams:
        towrite = '\sep'.join(list(trigram))        
        f.write(towrite+"\n")
    
with open(path+size+"/fourgrams.txt", "w", newline="") as f:
    for fourgram in fourgrams:
        towrite = '\sep'.join(list(fourgram))
        f.write(towrite+"\n")
    
with open(path+size+"/fivegrams.txt", "w", newline="") as f:
    for fivegram in fivegrams:
        towrite = '\sep'.join(list(fivegram))
        f.write(towrite+"\n")

print("END")