import networkx as nx
import numpy as np
from numpy.linalg import norm
import string
import pygraphviz as pgv

num = '100'
#Get nodes
with open('graph/topic_nodes.txt', 'r') as nodefile:
    nodelines = nodefile.readlines()
i=0
nodes = {}
trumps = []
ngrams = []
for line in nodelines:
    split = line.split('|')
    ngrams.append(split[0])
    
    if split[1]=="pro-trump\n":
        trumps.append(i)
    nodes[split[0]]=i
    i+=1

print(nodes)

#Get edges
with open('graph/topic_edges.txt', 'r') as edgefile:
    edgelines = edgefile.readlines()

edge_lengths = []
for line in edgelines:
    split = line.split('|')
    edge_length = split[2][:-1]
    #print(line)
    if edge_length=="None":
        edge_length=0
    if edge_length=="0.0":
        edge_length=0
    if edge_length!="0":
        edge_length = float(edge_length)
    else:
        edge_length=0
    edge_lengths.append(edge_length)

edge_array = np.asarray(edge_lengths)
edge_norm = norm(edge_array)
edge_min = min(edge_array)
edge_max = max(edge_array)

edge_array = np.subtract(edge_array, edge_min)#rescaling 1
edge_array = np.divide(edge_array, edge_max - edge_min)#rescaling 2
#edge_array = np.divide(edge_array, edge_norm) #divide by norm
#edge_array = np.log(edge_array) #log scale 1
#edge_array = [0 if x<0 else x for x in edge_array] #log scale 2
new_edge_array = edge_array 
#print(len(new_edge_array))
    
node_edges = {}
i=0
for line in edgelines:
    split = line.split('|')
    if split[0] in node_edges:
        node_edges[split[0]].append(1-(float("{0:.2f}".format(new_edge_array[i]))))
    else:
        node_edges[split[0]] = [1-(float("{0:.2f}".format(new_edge_array[i])))]
    i+=1

#print(len(node_edges))        

j=0
list_edges = []
for i in range(len(nodes)):
    for key, value in node_edges.items():
        if value[i]==1.0:
            value[i]=0.0;
            list_edges.append(tuple(value))
            break

#print(len(list_edges))
A = np.array(list_edges)
#print(A.shape)

dt = [('len', float)]
A = A.view(dt)
#print(A.shape)
#print(list_edges)

G = nx.from_numpy_matrix(A)
#G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),string.ascii_uppercase)))    


G = nx.drawing.nx_agraph.to_agraph(G)

G.node_attr.update(color="red", style="filled")

for node in trumps:
    n = G.get_node(node)
    n.attr['color']="blue"

i=0
for string in ngrams:
    n = G.get_node(i)
    n.attr['label']=string
    i+=1
    
G.edge_attr.update(color="blue", width="0.5")

G.graph_attr.update(strict=False)
G.graph_attr.update(overlap=False)
G.graph_attr.update(splines='none')

G.write("topic_all.dot")

#print(G.edges())
#print(G)

#G.draw('out_scale_'+num+'.png', format='png', prog='neato')