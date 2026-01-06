#!/usr/bin/env python
# coding: utf-8

# In[7]:


#get_ipython().run_cell_magic('javascript', '', '\nIPython.tab_as_tab_everywhere = function(use_tabs) {\n    if (use_tabs === undefined) {\n        use_tabs = true; \n    }\n\n    // apply setting to all current CodeMirror instances\n    IPython.notebook.get_cells().map(\n        function(c) {  return c.code_mirror.options.indentWithTabs=use_tabs;  }\n    );\n    // make sure new CodeMirror instances created in the future also use this setting\n    CodeMirror.defaults.indentWithTabs=use_tabs;\n\n    };\n\nIPython.tab_as_tab_everywhere()')


# In[8]:


import numpy as np
from scipy.sparse import coo_matrix
import os
import pandas as pd

from tqdm import tqdm

#from glove.kmerge import kmerge
from utils.chrono import Chrono

import pickle
import glob
import seaborn as sns

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from sklearn.ensemble import IsolationForest #1
from sklearn.cluster import DBSCAN #2

from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
from scipy import spatial
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from collections import defaultdict
import time

def defaultPlotting():
	sns.set_context("talk")
	#sns.set(rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":20})#'figure.figsize':(11.7,8.27),
	
defaultPlotting()

do_pickle=False
do_undirected=True


# In[9]:


with open('knn_matrix.pickle', 'rb') as handle:
	knn_m = pickle.load(handle)

print("Making full K-merge matrix")

rows, cols = knn_m.nonzero()
chosen_k=100

num_cores=32
indexes=[]

num_trajs = int(len(rows)/chosen_k)
print("Total num of trajs. are: ", num_trajs)

for i in range(num_cores):
	il = int(num_trajs/num_cores)
	indexes.append([j for j in range(i*il, i*il+il)])


# In[10]:


files = glob.glob('cost_kmerge/kmerge_cost_m*')

kmerge_cost=defaultdict(tuple)
for j in tqdm(range(len(files))):
	with open(files[j]) as f:
		for line in f:
			line=line[1:-1]
			arrays = line.split('array([')
			arrays = arrays[1:]
			l=len(arrays)
			i=0
			for arr in arrays:
				if i==l-1:
					val=arr[:-3].split(",")
				else:
					val=arr[:-4].split(",")
				val[1]=val[1][1:]
				for k in range(len(val)):
					if 'e' in val[k]:
						v = val[k].split('e')
# 						if len(v[1][1:])==0:
# 							print(v, i, val[k])
						val[k]=int(float(v[0])*(10**(int(v[1][1:]))))
					else:
						val[k]=int(float(val[k]))
				kmerge_cost[(rows[indexes[j][int(i/chosen_k)]*chosen_k + i%chosen_k], cols[indexes[j][int(i/chosen_k)]*chosen_k + i%chosen_k])] = (float(val[0]), float(val[1]))
				i=i+1
				#print((rows[indexes[j][int(i/chosen_k)]*chosen_k + i%chosen_k], cols[indexes[j][int(i/chosen_k)]*chosen_k + i%chosen_k]))


# In[11]:


from itertools import chain
import networkx as nx


# In[12]:


def enumerate_digraphs():
	global num_trajs, chosen_k, kmerge_cost
	g = nx.DiGraph()
	g.add_nodes_from([i for i in range(num_trajs)])
	
	# create a list of possible edges
	empty_i=set()
	for i in tqdm(range(num_trajs)):
		edges = []
		costs={}
		for j in range(chosen_k):
			l, m = rows[i*chosen_k + j], cols[i*chosen_k + j]
			if len(kmerge_cost[(l, m)])==0:
				empty_i.add(l)
			else:
				costs[(l, m)] = kmerge_cost[(l, m)][0][0]
				costs[(m, l)] = kmerge_cost[(l, m)][0][1]
				edge1 = (l, m)
				edge2 = (m, l)
				edges.append(edge1)
				edges.append(edge2)
		#print(edges)
		nx.set_edge_attributes(g, costs, "kmerge")
		g.add_edges_from(edges)
		#break
	print("There are: ", len(empty_i), " trajectories with zero neighbours!")
	return g


# In[13]:


# digraphs = enumerate_digraphs()
# print(digraphs)


# In[14]:


if do_pickle:
	digraphs = enumerate_digraphs()
	print(digraphs)
	with open('digraphs.pickle', 'wb') as handle:
		pickle.dump(digraphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
	start = time.time()
	with open('digraphs.pickle', 'rb') as handle:
		digraphs = pickle.load(handle)
	end = time.time()
	print(end - start, "seconds")


# In[15]:


# e = nx.get_edge_attributes(digraphs, "kmerge")

# for ep in e:
# 	print(ep, digraphs.get_edge_data(ep[0], ep[1]))
# 	print(ep, digraphs.get_edge_data(ep[1], ep[0]))


# In[16]:


def enumerate_graphs():
	global num_trajs, chosen_k, kmerge_cost
	g = nx.Graph()
	g.add_nodes_from([i for i in range(num_trajs)])
	
	# create a list of possible edges
	#print(num_trajs)
	empty_i=set()
	for i in tqdm(range(num_trajs)):
		edges = []
		costs={}
		for j in range(chosen_k):
			l, m = rows[i*chosen_k + j], cols[i*chosen_k + j]
			if len(kmerge_cost[(l, m)])==0:
				empty_i.add(l)
			else:
				g.add_edge(l, m, weight=kmerge_cost[(l, m)][0])
# 				#print(kmerge_cost[(l, m)])
# 				costs[(l, m)] = kmerge_cost[(l, m)][0]
# 				edge1 = (l, m)
# 				edges.append(edge1)
# 		print(costs.items())
# 		nx.set_edge_attributes(g, costs)
# 		g.add_edges_from(edges)
		#break
	print("There are: ", len(empty_i), " trajectories with zero neighbours!")
	return g


# In[17]:


graphs = enumerate_graphs()
print(graphs)


# In[ ]:


# if do_undirected:
# # 	graphs = enumerate_digraphs()
# # 	print(graphs)
# 	with open('graphs.pickle', 'wb') as handle:
# 		pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
# else:
# 	start = time.time()
# 	with open('graphs.pickle', 'rb') as handle:
# 		graphs = pickle.load(handle)
# 	end = time.time()
# 	print(end - start, "seconds")


# In[ ]:


def find_num_disconnected_components():
	global digraphs
	
	cluster_ids=set()
	#nodes = digraphs.nodes()
	
	print("Number of strongly connected components are: ", nx.number_strongly_connected_components(digraphs))
	
	c_comp = nx.kosaraju_strongly_connected_components(digraphs)
	
	l_comp = [len(comp) for comp in c_comp]
	
# 	for comp in c_comp:
# 		print(comp)
	
	fig, ax = plt.subplots()
	
	df = pd.Series(l_comp)
	sns.boxplot(df)
	plt.xlabel("Len. of strongly connected comps.")

	plt.savefig("sconnected_comps.pdf")

#find_num_disconnected_components()


# In[ ]:


def find_num_disconnected_components_not_dir(graphs_):
	
	cluster_ids=set()
	#nodes = digraphs.nodes()
	
	print("Number of connected components are: ", nx.number_connected_components(graphs_))
	
	c_comp = nx.connected_components(graphs_)
	
	l_comp = [len(comp) for comp in c_comp]
	
# 	for comp in c_comp:
# 		print(comp)
	
	fig, ax = plt.subplots()
	
	df = pd.Series(l_comp)
	sns.boxplot(df)
	plt.xlabel("Len. of strongly connected comps.")

	plt.savefig("connected_comps.pdf")

#find_num_disconnected_components_not_dir()


# In[ ]:


def get_graph_info(g_):
	print(nx.maximal_independent_set(digraphs))

# get_graph_info()


# In[ ]:


def save_graph(graph,file_name):
	#initialze Figure
	plt.figure(num=None, figsize=(20, 20), dpi=80)
	plt.axis('off')
	fig = plt.figure(1)
	pos = nx.random_layout(graph)
	nx.draw_networkx_nodes(graph,pos)
	nx.draw_networkx_edges(graph,pos)
	nx.draw_networkx_labels(graph,pos)

	cut = 1.00
	xmax = cut * max(xx for xx, yy in pos.values())
	ymax = cut * max(yy for xx, yy in pos.values())
	plt.xlim(0, xmax)
	plt.ylim(0, ymax)

	plt.savefig(file_name,bbox_inches="tight")

#Assuming that the graph g has nodes and edges entered
#save_graph(digraphs,"knn_graph_visual.pdf")

#it can also be saved in .svg, .png. or .ps formats


# In[ ]:


with open('knn_matrix.pickle', 'rb') as handle:
	knn_m = pickle.load(handle)


# In[ ]:


print("Making full K-merge matrix")

rows, cols = knn_m.nonzero()
chosen_k=100

num_cores=32
# indexes=[]

# for i in range(num_cores):
# 	il = int(num_trajs/num_cores)
# 	indexes.append([j for j in range(i*il, i*il+il)])


# In[ ]:


from multiprocessing import Process
from multiprocessing import Pool, freeze_support
from itertools import repeat
import cProfile

# def get_kmerge_pairs_cost(indexes):
# 	global chosen_k
# 	file = open("kmerge_cost_m" + str(indexes[0]) + ".txt", "w")

# 	for i in indexes:
# 		tp=[]
# 		this_results=[]
# 		for j in range(chosen_k):
# 			traj1=np.zeros((len(tdata_full[rows[i*chosen_k + j]]), 9))
# 			traj2=np.zeros((len(tdata_full[cols[i*chosen_k + j]]), 9))

# 			idd=0
# 			for k in range(len(traj1)):
# 				traj1[k] = np.append(tdata_full[rows[i*chosen_k + j]][k], idd)

# 			for k in range(len(traj2)):
# 				traj2[k] = np.append(tdata_full[cols[i*chosen_k + j]][k], idd+1)

# 			tp.append( np.array([*traj1, *traj2]) )

# 			#tp.append( np.array([tdata_full[rows[i*chosen_k + j]], tdata_full[cols[i*chosen_k + j]]]) )
# 		for m in range(len(tp)):
# 			this_results.append( get_node_pairs_m(tp[m]) )

# 		file.write(str(this_results))
# 	file.close()

# with Pool() as pool_per_key:
# 	pool_per_key.starmap( get_kmerge_pairs_cost, zip(indexes) )


# ## Optimal Selection of k

# In[ ]:


best_selection=defaultdict(list)
anon_k=10

prev_key=-1
# for key in tqdm(kmerge_cost.keys()):
# 	if prev_key==-1:
# 		prev_key=key[0]
# 		vals=[]
# 	elif prev_key!=key[0]:
# 		s_val = sorted(vals, key=lambda x: x[1])[:anon_k]
# 		best_selection[prev_key]=s_val
# 		vals=[]
# 		vals.append( (key[1], kmerge_cost[key][1]) )
# 	else:
# 		vals.append( (key[1], kmerge_cost[key][1]) )
# 	prev_key=key[0]


# In[ ]:


def enumerate_red_graphs():
	global best_selection, anon_k, num_trajs
	
	keys = list(best_selection.keys())
	print(len(keys))
	
	g = nx.DiGraph()
	g.add_nodes_from([i for i in range(len(keys))])
	
	# create a list of possible edges
	empty_i=set()
	for i in tqdm(range(len(keys))):
		edges = []
		costs={}
		for j in range(anon_k):
			l = best_selection[keys[i]][j][0]
			if len(kmerge_cost[(i, l)])==0:
				empty_i.add(l)
			else:
				costs[(i, l)] = best_selection[keys[i]][j][1]
				edge1 = (i, l)
				edges.append(edge1)
		#print(edges)
		nx.set_edge_attributes(g, costs, "kmerge")
		g.add_edges_from(edges)
		#break
	print("There are: ", len(empty_i), " trajectories with zero neighbours!")
	return g


# In[ ]:


#redgraphs = enumerate_red_graphs()


# In[ ]:


#find_num_disconnected_components_not_dir(redgraphs)


# In[ ]:


# sns.ecdfplot(np.array(list(nx.in_degree_centrality(redgraphs).values()))*num_trajs)
# plt.xlim([0, 25])


# In[ ]:


# sns.ecdfplot(np.array(list(nx.out_degree_centrality(redgraphs).values()))*num_trajs)
# plt.xlim([0, 25])


# ## Community Detection

# In[ ]:


from community import community_louvain

partion_there=True

if not partion_there:
	#first compute the best partition
	partition = community_louvain.best_partition(graphs)#, weight='kmerge'
	print("Partioned...")

	#pickling partitions
	with open('community_partitions.pickle', 'wb') as handle:
		pickle.dump(partition, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print("Pickled...")
else:
	with open('community_partitions.pickle', 'rb') as handle:
		partition = pickle.load(handle)

#drawing
fig, ax = plt.subplots()
size = float(len(set(partition.values())))
pos = nx.spring_layout(graphs)
count = 0.
for com in set(partition.values()) :
	count = count + 1.
	list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
	nx.draw_networkx_nodes(graphs, pos, list_nodes, node_size = 20, node_color = str(count / size))

nx.draw_networkx_edges(graphs,pos, alpha=0.5)
plt.savefig("community_senegal.pdf")
#plt.show()

