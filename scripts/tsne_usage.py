import numpy as np
from matplotlib import pyplot as plt
import re
from tsne import bh_sne

# load up data
print("Loading data...")
data = []
with open('/mnt/storage01/milliet/data/ngrams/clean-ngrams-score-9500.csv', 'r') as csvfile:
    lines = csvfile.readlines()
    for line in lines:
        data.append(line.split('\sep'))

threshold = 0.999
data_threshold = list(elem for elem in data if float(elem[-1])>threshold)

print("Get embedding...")
x_data = list(re.sub("\s+", ",", elem[1][2:-2]).split(',') for elem in data_threshold)
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

x_data = np.array(x_data)

# convert image data to float64 matrix. float64 is need for bh_sne
#x_data = np.asarray(x_data).astype('float64')
#x_data = x_data.reshape((x_data.shape[0], -1))

# For speed of computation, only run on a subset
n = 20000
x_data = x_data[:n]

y_data = y_data[:n]
y = []
for elem in y_data:
    if elem=='pro-trump':
        y.append((1,0,0))
    else:
        y.append((0,0,1))

# perform t-SNE embedding
print("BH-SNE...")
vis_data = bh_sne(x_data)

# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

#print("vis_x: " + str(vis_x))
#print("vis_y: " + str(vis_y))

print("Plotting...")
plt.scatter(vis_x, vis_y, c=y, cmap=plt.cm.get_cmap("jet", 10))
#plt.colorbar(ticks=range(10))
#plt.clim(-0.5, 9.5)
plt.show()