#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().run_cell_magic('javascript', '', '\nIPython.tab_as_tab_everywhere = function(use_tabs) {\n    if (use_tabs === undefined) {\n        use_tabs = true; \n    }\n\n    // apply setting to all current CodeMirror instances\n    IPython.notebook.get_cells().map(\n        function(c) {  return c.code_mirror.options.indentWithTabs=use_tabs;  }\n    );\n    // make sure new CodeMirror instances created in the future also use this setting\n    CodeMirror.defaults.indentWithTabs=use_tabs;\n\n    };\n\nIPython.tab_as_tab_everywhere()')


# # Glove
# * See Gramaglia2015
# 
# * **1.** Compute the cost of merging for all couples of trajectory
# * **2.** Merge the couple $i,j$ that costs the least
# * **3.** Add the new trajectory in the database, 
# * **4.** If the new trajectory represents less than $k$ users, compute the cost of merging with each other old trajectory
# * **4.** Delete trajectories $i$ and $j$
# * **5.** Repeat from step **2.** until all trajectories are made of at least $k$ users

# In[3]:


import numpy as np
from scipy.sparse import coo_matrix

from tqdm import tqdm

from glove.kmerge import kmerge
from utils.chrono import Chrono

# from signal import signal, SIGPIPE, SIG_DFL
# signal(SIGPIPE,SIG_DFL)


# # Solver data structure

# In[4]:


def get_completeness_matrix(user_col, k):
	"""
	input : user_col : a list of ids of user from 0 to k-1, with repetition, shuffled
	return : a matrix bingo[i][j] = True iff all users are present between i and j (included) in user_col
	"""
	bingo_complete = np.zeros(shape=(len(user_col), len(user_col),k), dtype=bool)
	for i in range(len(user_col)):  # lower bound
		bingo_complete[i,i,user_col[i]] = 1
		for j in range(i+1, len(user_col)):  # upper bound 
			bingo_complete[i,j] = bingo_complete[i,j-1]
			bingo_complete[i,j, user_col[j]] += 1
	return np.all(bingo_complete, axis=2)

def get_completeness_linear(user_col, k):
	"""
	input : user_col : a list of ids of user from 0 to k-1, with repetition, shuffled
	return : a matrix bingo[i][j] = Gives cumulative cost
	"""
	ids = [i for i in range(len(k))]
	l = len(user_col)
	return np.cumsum[user_col]

def get_completeness_matrix_sparse(user_col, k):
	"""
	input : user_col : a list of ids of user from 0 to k-1, with repetition, shuffled
	return : a matrix bingo[i][j] = True iff all users are present between i and j (included) in user_col
	"""
	bingo_complete = np.zeros(shape=(len(user_col), len(user_col),k), dtype=bool)
	for i in range(len(user_col)):  # lower bound
		bingo_complete[i,i,user_col[i]] = 1
		for j in range(i+1, len(user_col)):  # upper bound 
			bingo_complete[i,j] = bingo_complete[i,j-1]
			bingo_complete[i,j, user_col[j]] += 1
	return np.all(bingo_complete, axis=2)

def get_time_coherence(tminmax):
	"""
	input : tminmax : array of timestamps [[tmin, tmax]] order by tmin
	returns :
		coherence : an array of same len as tminmax
		coherence[i]==True iff tmax[i-1] <= tmin[i]
	"""
	tminmax_prev = np.roll(tminmax, 1, axis=0)
	tminmax_prev[0] = [0,0]

	return (tminmax[:,0] - tminmax_prev[:,1])>=0

def stretch_cost(trajs, i, j):
	"""
	reminder : 
		trajs[i, 0] = x_min of i-th point is trajs
		trajs[i, 1] = Y_min of i-th point is trajs
		trajs[i, 2] = t_min of i-th point is trajs
		trajs[i, 3] = site_id of i-th point is trajs
		trajs[i, 4] = x_max of i-th point is trajs
		trajs[i, 5] = Y_max of i-th point is trajs
		trajs[i, 6] = t_max of i-th point is trajs
		trajs[i, 7] = site_id of i-th point is trajs

	returns :
		stretch_cost = Delta_t * (Delta_x + Delta_y)
	"""

	# Delta_t = max_t - min_t
	delta_t = (trajs[i,6] - trajs[j,2])  # necessarily in this order bc trajs is ordered by time

	delta_space = (trajs[j:i+1, 4:6].max(axis=0) - trajs[j:i+1, 0:2].min(axis=0)).sum()

	return delta_t*delta_space

def stretch_cost2(trajs, i, j):
	"""
	reminder : 
		trajs[i, 0] = x_min of i-th point is trajs
		trajs[i, 1] = Y_min of i-th point is trajs
		trajs[i, 2] = t_min of i-th point is trajs
		trajs[i, 3] = site_id of i-th point is trajs
		trajs[i, 4] = x_max of i-th point is trajs
		trajs[i, 5] = Y_max of i-th point is trajs
		trajs[i, 6] = t_max of i-th point is trajs
		trajs[i, 7] = site_id of i-th point is trajs

	returns :
		stretch_cost = Delta_t * (Delta_x + Delta_y)
	"""

	# Delta_t = max_t - min_t
	delta_t = (trajs[i,6] - trajs[j,2])  # necessarily in this order bc trajs is ordered by time

	delta_space = (trajs[j:i+1, 4:6].max(axis=0) - trajs[j:i+1, 0:2].min(axis=0)).sum()

	return delta_t*delta_space

def kmerge2(trajs, k):
	"""
	input :
		trajs : np array of shape (nb_users, nb_points_per_user, 2, 3), 
			each point is [[xmin, ymin, tmin], [xmax, ymax, tmax]]
	returns : 
		partition : a list of cluster id clustering the points, with shape (nb_users * nb_points_per_user)
		partition[i] is the cluster id of i-th point, when you flatten and sort points by time

	"""
# 	k = 2 #trajs.shape[0]
	nb_points = len(trajs)#trajs.shape[1]
	#print((k, nb_points))
	#trajs = trajs.reshape(k * nb_points, 8)
	#print(trajs)

	# add a column to identify user
	#trajs = np.concatenate([trajs, np.reshape(np.repeat(np.arange(k), nb_points), (nb_points*k, 1))], axis=1)

	trajs = trajs[trajs[:, 2].argsort()]  # sorting by 2nd col, tmin
	trajs = trajs[trajs[:, 2]>0]  # get rid of points that were actually just padding


	# costs[i] gives you the best total cost to aggregate the given trajs up to index i
	costs = np.inf*np.ones(len(trajs))
	costs[0] = 0

	partition = np.zeros(len(trajs), dtype=int)  # partition[i] : earliest sample to which sample[i] is agg to
	
	#print(trajs[:, -1].astype(int))

	id_array = trajs[:, -1].astype(int)
	#completeness_matrix = get_completeness_linear(id_array, k)
	time_coherence = get_time_coherence(trajs[:,[2,6]])
	for i in range(trajs.shape[0]):
		# get highest index showing True in column i
		try:
			j = i-k+1
			if j < 0:
				j=0
			else:
				while len(set(id_array[j:i+1]))!=k and j > 0:
					j=j-1
		except:  # fails if there is no True in column
			j=0
		prev_cost = costs[max(j-1, 0)]  # if j==0 then the stretch cost for previous agg is 0

		if not len(set(id_array[j:i+1]))==k:
			# happens when too early in the list of samples
			jicost = stretch_cost(trajs, i, j)
			costs[i] = jicost + prev_cost

		elif not len(set(id_array[partition[j-1]:j]))==k or not time_coherence[j]:
			# happens when making a new complete cluster containing the i-th element
			# implies taking too many elements from the previous clusters
			# or breaking time coherence with previous cluster
			j = partition[j-1]
			jicost = stretch_cost(trajs, i, j)
			costs[i] = jicost + prev_cost
			partition[j:i+1] = j

		while j>=0 and len(set(id_array[partition[j-1]:j]))==k and time_coherence[j]:  
			# while we break neither completeness of previous cluster, nor time coherence
			#print(len(set(id_array[partition[j-1]:j]))==k)

			jicost = stretch_cost(trajs, i, j)
			if costs[i] > jicost + prev_cost:
				costs[i] = jicost + prev_cost
				partition[j:i+1] = j
			j = j-1


	generalized_traj = np.zeros((nb_points, 2, 4))
	for i, cluster_id in enumerate(np.unique(partition)):
		generalized_traj[i,0] = trajs[partition==cluster_id, 0:4].min(axis=0)
		generalized_traj[i,1] = trajs[partition==cluster_id, 4:8].max(axis=0)

	return generalized_traj, costs[-1]/np.unique(trajs[:, -1].astype(int), return_counts=True)[1]

def get_node_pairs_m(this_pair, k=2):
	return kmerge2(this_pair, k)[1]

def get_node_trajs_m(this_trajs, k=2):
	return kmerge2(this_trajs, k)


# In[5]:


from multiprocessing import Process
from multiprocessing import Pool, freeze_support
from itertools import repeat
import cProfile
import ast
import pickle

class glove_data_handler:
	def get_kmerge_pairs_cost(self, indx):
		global tdata_full
		
		file1 = open("glove5k/cost" + str(indx[0]) + ".txt", "w")
# 		file2 = open("glove5k/gtrajs" + str(indx[0]) + ".txt", "w")
		
# 		result=[]
		for idx in indx:
			tp=[]
			trajs_here=idx #i,j
			idd=0
			for j in range(len(trajs_here)):
	# 					print(len(tdata_full_shenzen[trajs[j]]))
				traj1=np.zeros((len(tdata_full[trajs_here[j]]), 9))

				for k in range(len(traj1)):
					traj1[k] = np.append(tdata_full[trajs_here[j]][k], idd)

				tp.extend( [*traj1] )

				idd = idd + 1
			tp = np.array(tp)
	# 					print(i, tp.shape[0], tp.shape[1], len(trajs), trajs)

	# 				this_results = get_node_trajs_m(tp, len(trajs))

			result = get_node_trajs_m(tp, len(trajs_here)) #.append( get_node_trajs_m(tp, len(trajs_here)) ) #generalised_traj, cost = kmerge(trajs[[i,j]])
	
			trace=str(int(result[1][0])) + " "
			# 	print(trace)

			file1.write(trace)#this_results
			
# 			trace=""
# 			for samp in result[0]:
# 			# 	print(gtraj[0][4:])
# # 				if int(samp[0])!=0:
# # 					for val in samp:
# # 						if len(trace)>0:
# # 							trace=trace + ' ' + str(val)
# # 						else:
# # 							trace=trace + str(val)
# 				trace=trace+str(samp)
# 				trace=trace+'\n'
# 			trace=trace+'\n'

# 			file2.write(trace)#this_results
			
			with open("glove5k/gtrajs" + str(idx) + '.pickle', 'wb') as handle:
				pickle.dump(result[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		file1.close()
		return 1

	def __init__(self, trajs, loose=False):
		self.trajs = trajs
		self.loose=loose
		
		self.active = np.ones(len(trajs), dtype=bool) # True iff traj is still up to be merged

		self.costs = np.zeros(shape=(int(len(trajs)*(len(trajs)-1)/2), 3))  # i,j,val, sorted by val        
		cost_counter = 0

		self.gen_traj = {}

		indexes=[]
		for i in tqdm(range(len(trajs))):
			for j in range(i+1, len(trajs)):
				indexes.append( (i,j) )
		
		ixes=[]
		num_cores=32 #32
		num_trajs=len(indexes)

		count=0
		for i in range(num_cores):
			il = int(num_trajs/num_cores)
			ixes.append([indexes[j] for j in range(i*il, i*il+il)])
			count=count+len(ixes[-1])

		if ixes[-1][-1]!=indexes[num_trajs-1]:
			if len([indexes[j] for j in range(count, num_trajs)])>0:
				ixes.append([indexes[j] for j in range(count, num_trajs)]) #indexes[-1][-1]+1

# 		print(ixes)

		with Pool() as pool_per_key:
			results = pool_per_key.starmap( self.get_kmerge_pairs_cost, zip(ixes) )

# 		print(ixes)

		m=0
# 		for res in results:
# 			for j in range(len(ixes[m])):
# 				self.costs[cost_counter] = [ixes[m][j][0], ixes[m][j][1], res[j][1][0]] #i, j, cost

# 				cost_counter += 1

# 				self.gen_traj[ixes[m][j]] = res[j][0] #(i,j), generalised_traj

# # 				print(j, ixes[m][j])

# 			m=m+1

# # 			print(res[1], res[0])

		files = glob.glob("glove5k/cost*")
		for j in tqdm(range(len(files))):
			with open(files[j]) as f:
				this_index = files[j].split("/")[1].split(".")[0][4:]

				i=0
				for idx in ixes:
					if str(idx[0])==this_index:
						this_index=i
						break
					i=i+1
				
				k=0
				for line in f:
					line = line.split(" ")
					
					for cst in line:
						if len(cst)>0:
	# 						print(ixes[this_index][k], ixes[this_index][-1], len(line), len(ixes[this_index]))
							self.costs[cost_counter] = [ixes[this_index][k][0], ixes[this_index][k][1], int(cst)] #i, j, cost
							k=k+1

							cost_counter += 1

# 		files = glob.glob("glove5k/*.pickle")
# 		for j in tqdm(range(len(files))):
# 			this_index = files[j].split("/")[1].split(".")[0][6:]

# 			with open("glove5k/gtrajs" + str(this_index) + '.pickle', 'rb') as handle:
# 				trj = pickle.load(handle)

# 			self.gen_traj[this_index] = trj
# # 				i=0
# # 				for idx in ixes:
# # 					if str(idx[0])==this_index:
# # 						this_index=i
# # 						break
# # 					i=i+1

# # 		# 				print([val[0] for val in self.indexes])
# # 		# 					print(this_index)
# # 				k=0
# # 				trj = ""

# # 				for line in f:
# # 					if line == '\n':
# # 						print(ast.literal_eval(trj))

# # 						self.gen_traj[ixes[this_index][k]] = ast.literal_eval(trj) #(i,j), generalised_traj

# # 						trj=""
# # 						k=k+1
# # 					else:
# # 						trj=trj+line

		self.costs = self.costs[self.costs[:, 2].argsort()]  # sort by cost of aggregating
		
# 		for val in self.costs:
# 			print(val)

		self.original_content = [[i] for i in range(len(trajs))] # list of all original trajs that make up i-th traj
# 		print(self.original_content)

	def get_cheapest_merge(self):
		"""
		Returns the cheapest 2-merge among all the current active trajectories
		"""
# 		print(self.costs)
		for m in range(len(self.costs)):
			[i,j,val] = self.costs[m] #0
			if i==j:
				continue
			else:
				break
		#num_merge = 0
		#while not (self.active[int(i)] and self.active[int(j)]):
		#    num_merge += 1
		#    [i,j,val] = self.costs[num_merge]
		#self.costs = self.costs[num_merge+1:]
		return int(i), int(j), val


	def delete_indices(self, indices):
		"""
		Keeps track of what trajectories are active

		Code optimisation possibility :
			discarding the given indices in self.costs in this function
			would allow us to consistently skip the while loop in get_cheapest_merge
			Both solutions seem to have the same execution time
			But filtering is cleaner to integrate in the solving loop because of the loop condition

			we should also delete the precomputed generalized trajs but it's harder
			because it would mess up the indices we use
		"""

		self.costs = self.costs[~np.isin(self.costs[:,0], indices)]
		self.costs = self.costs[~np.isin(self.costs[:,1], indices)]

		self.active[indices] = False



	def add_merged(self, i, j, active_bool):
		"""
		adds the result of merging trajectories i and j as a new trajectory
		new trajectory is set active as active_bools
		computes the costs of merging the old trajectories with the new one
		"""

# 		print(self.gen_traj[(i,j)])
		val = (i,j)
		
		with open("glove5k/gtrajs" + str(val) + '.pickle', 'rb') as handle:
			trj = pickle.load(handle)
		self.gen_traj[(i,j)] = trj

# 		print(len(self.trajs), self.trajs[-1])
		self.trajs.append(self.gen_traj[(i,j)].reshape(self.gen_traj[(i,j)].shape[0], 8)) #.reshape(self.gen_traj[(i,j)].shape[0], 8)
# 		print(len(self.trajs), self.trajs[-1])

# 		self.trajs = np.concatenate([self.trajs, [self.gen_traj[(i,j)]]])
		
# 		if not self.loose:
		self.active = np.concatenate([self.active, [active_bool]])

		costs_to_add = np.zeros(shape=(0,3))
		if active_bool:
			for a in range(len(self.active)-1):
				if self.active[a]:
# 					tp=[]
# 					tp.append(self.trajs[a])
# 					tp.append(self.trajs[-1])
# # 					print(tp)

					tp=[]
					trajs_here=[a,-1]
					idd=0
					for j in range(len(trajs_here)):
	# 					print(len(tdata_full_shenzen[trajs[j]]))
						traj1=np.zeros((len(tdata_full[trajs_here[j]]), 9))

						for k in range(len(traj1)):
							traj1[k] = np.append(tdata_full[trajs_here[j]][k], idd)

						tp.extend( [*traj1] )

						idd = idd + 1
					tp = np.array(tp)

					generalised_traj, cost = get_node_trajs_m(tp, len(trajs_here)) #kmerge(trajs[[i,j]])
# 					generalised_traj, cost = kmerge2(np.array(tp)) #self.trajs[[a,-1]]

					self.gen_traj[(a, len(self.active)-1)] = generalised_traj #reshape(self.gen_traj[(i,j)].shape[0], 8)
					costs_to_add = np.concatenate([costs_to_add, [[a, len(self.active)-1, cost[0]]]], axis=0)

		ii = np.searchsorted(self.costs[:,2], costs_to_add[:,2])
		self.costs = np.insert(self.costs, ii, costs_to_add, axis=0)

		self.original_content+= [self.original_content[i] + self.original_content[j]]


# # Generation of test data

# In[6]:


anon_thres = 2 #25, 5 originally by Benoit, 2, 200000000000000
nb_trajs = 200
nb_points = 5

# trajs = np.random.rand(nb_trajs, nb_points, 3) # points are [x,y,t]

# # translation wrt time so we have a pattern
# translation_speed = 15
# for k in range(trajs.shape[0]):
#     for i_point in range(trajs.shape[1]):
#         trajs[k,i_point,0] += translation_speed*trajs[k,i_point,2]  # translate along x
#         trajs[k,i_point,1] += translation_speed*np.sin(20*trajs[k,i_point,2])  # ossillate along y
        
# # Adding max bounds, equal to min bounds for the moment
# trajs = np.repeat(trajs, 2, axis=1).reshape((trajs.shape[0], trajs.shape[1], 2, 3))
   
# trajs.shape

# print(trajs)


# In[7]:


# print(trajs[[1, -1]])

# print(trajs[1], trajs[-1])


# ## Generating real trajectories

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

import geopandas as gpd
from geovoronoi import  coords_to_points, voronoi_regions_from_coords

from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
from scipy import spatial
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from collections import defaultdict
import time
import random
import pstats

def defaultPlotting():
	sns.set_context("paper")
	#sns.set(rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":20})#'figure.figsize':(11.7,8.27),
	
defaultPlotting()

do_pickle=False
do_undirected=True


# In[9]:


from datetime import datetime, timedelta

already_processed=True
treat_ts=True
shenzen_dataset=False #True
mil_trajs=False#True

class trajAnonymizer:
	def __init__(self, trajs):
		self.trajs = trajs
		self.epoch = datetime(1970,1,1)
		self.week_ts = 1209600
		
	def getNTrajs(self, N, week_pair=0):
		idxs=[8, 20, 21, 9, 23, 22, 25, 19, 18, 24, 15, 1, 14, 2, 16, 17, 3, 7, 13, 12, 6, 10, 4, 5, 11]
# 		data = self.trajs[week_pair]
		weeks=[0, 3, 19, 21]
		with open('senegal_data_'+str(weeks[0])+'.pickle', 'rb') as handle:
			data=pickle.load(handle)
		
		if N != -1:
			data = data[data["user_id"]<=N]
		else:
			print("There are: ", len(set(data["user_id"])), "Trajs. in the datatset.")
		
# 		dts = data["timestamp"].values
# 		for i, dt in tqdm(enumerate(dts)):
# 			#print(dt)
# 			dts[i] = self.totimestamp(dt) # - (idxs[week_pair]-1)*self.week_ts
# 		data["timestamp"] = dts#data["timestamp"].apply(self.totimestamp)

		print("Got the trajectories!")
		#print(data)
		return data
	
	def getNTrajsRand(self, samp_trajs, week_pair=0):
# 		data = self.trajs[week_pair]
		weeks=[0, 3, 19, 21]
		with open('senegal_data_'+str(weeks[0])+'.pickle', 'rb') as handle:
			data=pickle.load(handle)
		
		data = data[data["user_id"].isin(samp_trajs)]
# 		dts = data["timestamp"].values
# 		for i, dt in tqdm(enumerate(dts)):
# 			#print(dt)
# 			dts[i] = self.totimestamp(dt)
# 		data["timestamp"] = dts#data["timestamp"].apply(self.totimestamp)

		print("Got the trajectories!")
		#print(data)
		return data
	
	def getMilTrajs(self):
		data = self.trajs[0]
		last_uid= max(list(data["user_id"]))
# 		print(data)
		for j in range(1, 3):
# 			print(last_uid)
			dt = self.trajs[j]
# 			print(dt)
			dt["user_id"]=dt["user_id"]+last_uid
			
			data = pd.concat([data, dt], ignore_index=True)
			last_uid= max(list(data["user_id"]))
		dt = self.trajs[3]
		dt = dt[dt["user_id"]<=50000]
		dt["user_id"]=dt["user_id"]+last_uid
# 		print(last_uid)
		
		data = pd.concat([data, dt], ignore_index=True)
		
		print(data)
		print("There are: ", len(set(data["user_id"])), "Trajs. in the datatset.")
		dts = data["timestamp"].values
		uds = data["user_id"].values
		
		with open('week_start_million.pickle', 'rb') as handle:
			week_data = pickle.load(handle)
		
		l=320000
		
		for i, dt in tqdm(enumerate(dts)):
			#print(dt)
			if uds[i]<=l:
				dts[i] = self.totimestamp(dt)
			elif uds[i]>l and uds[i]<=2*l:
				dts[i] = self.totimestamp(dt)-week_data[1][0]+week_data[0][0]
			elif uds[i]>2*l and uds[i]<=3*l:
				dts[i] = self.totimestamp(dt)-week_data[2][0]+week_data[0][0]
			elif uds[i]>3*l:
				dts[i] = self.totimestamp(dt)-week_data[3][0]+week_data[0][0]
		data["timestamp"] = dts#data["timestamp"].apply(self.totimestamp)
		print("Got the trajectories!")
		#print(data)
		return data
	
	def getMilTrajsCorrect(self):
		weeks=[0, 3, 19, 21] #[0, 1, 2, 3]
		with open('senegal_data_'+str(weeks[0])+'.pickle', 'rb') as handle:
			data=pickle.load(handle)
		last_uid= max(list(data["user_id"]))
# 		print(data)
		for j in range(1,len(weeks)-1):
# 			print(last_uid)
			with open('senegal_data_'+str(weeks[j])+'.pickle', 'rb') as handle:
				dt=pickle.load(handle)
# 			print(dt)
			dt["user_id"]=dt["user_id"]+last_uid
			
			data = pd.concat([data, dt], ignore_index=True)
			last_uid= max(list(data["user_id"]))
		with open('senegal_data_'+str(weeks[-1])+'.pickle', 'rb') as handle:
			dt=pickle.load(handle)
		dt = dt[dt["user_id"]<=50000]
		dt["user_id"]=dt["user_id"]+last_uid
# 		print(last_uid)
		
		data = pd.concat([data, dt], ignore_index=True)
		
		print(data)
		print("There are: ", len(set(data["user_id"])), "Trajs. in the datatset.")
		print("Got the trajectories!")
		#print(data)
		return data
	
	def filter_anomalies(self, data, max_traj_length):
		count_id = data.user_id.value_counts().sort_index().values
		
		uids=[i for i in range(len(count_id)) if count_id[i] > max_traj_length]
		print(uids)
		data = data[data["user_id"] not in uids]
		
		return data
	
	def transformToKmergeData(self, data, max_traj_length):
		data = data.sort_values("user_id")
		count_id = data.user_id.value_counts().sort_index().values
		
# 		fig, ax = plt.subplots()
	
# 		df = pd.Series(count_id)
# 		print(count_id, df)
# 		sns.ecdfplot(df)
# 		plt.xlabel("Length of user trajs.")

# 		plt.savefig("trajectory_lengths.pdf")
		
		mask = count_id[:,None] > np.arange(count_id.max()) #count_id.max()
		vals = data[["x", "y", "timestamp", "site_id"]].values
		out_shp = mask.shape + (vals.shape[1],)
		out = np.full(out_shp, -1000)
		out[mask] = vals
		
		#ltrajs = out.shape[1]
		#out = np.delete(out, [i for i in range(max_traj_length, ltrajs)], axis=1)
		#print(out.shape, count_id[0], data.iloc[count_id[0]+1])
		
		#data=data[["x", "y", "timestamp"]].to_numpy()
		print("Converted the data to the K-merge shape.")
		return out
	
	def transformToKmergeData2(self, data):
		data = data.sort_values("user_id")
		count_id = data.user_id.value_counts().sort_index().values
		
		data = data[["x", "y", "timestamp", "site_id"]].values
		
		count=0
		fdata=[]
		for nsamp in count_id:
			fdata.append(data[count:count+nsamp])
			count=count+nsamp
		print("Converted the data to the K-merge shape.")
		return fdata
	
	def transformToKmergeData3(self, data): #Shenzen dataset
		data = data.sort_values("uid")
		count_id = data.uid.value_counts().sort_index().values
		
		data = data[["x", "y", "ts", "uid"]].values
		
		count=0
		fdata=[]
		for nsamp in count_id:
			fdata.append(data[count:count+nsamp])
			count=count+nsamp
		print("Converted the data to the K-merge shape.")
		return fdata
	
	def totimestamp(self, idt):
		dt = datetime.strptime(idt, "%Y-%m-%d %H:%M:%S")
		td = dt - self.epoch
		return td.total_seconds()
		#return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / 1e6

def add_cloumns_data():
	global user_data
	for j in tqdm(range(len(files))):
		#user_data[j] = pd.concat([user_data[j].columns.to_frame().T, user_data[j]], ignore_index=True)
		user_data[j].columns = ["user_id", "timestamp", "site_id"]
		user_data[j] = user_data[j].merge(antennas, on='site_id', how="left")

def repeat_samples(idata):
	global tdata_full
	for traj in idata:
		udata=[]
		for samp in traj:
			#print([*samp, *samp])
			udata.append([*samp, *samp])
		#print(np.vstack(udata).shape)
		tdata_full.append(np.vstack(udata))#np.vstack(udata)

def repeat_samples2(idata):
	global tdata_full_shenzen
	for traj in idata:
		udata=[]
		for samp in traj:
			#print([*samp, *samp])
			udata.append([*samp, *samp])
		#print(np.vstack(udata).shape)
		tdata_full_shenzen.append(np.vstack(udata))#np.vstack(udata)

def repeat_samples3(idata):
	global tdata_full_civ
	tid=0
	for traj in idata:
		udata=[]
		for samp in traj:
			samp[-1]=tid
			#print([*samp, *samp])
			udata.append([*samp, *samp])
		#print(np.vstack(udata).shape)
		tdata_full_civ.append(np.vstack(udata))#np.vstack(udata)
		tid=tid+1

def repeat_samples4(idata):
	dt=[]
	j=0
	for traj in tqdm(idata):
		udata=[]
		for samp in traj:
			#print([*samp, *samp])
			if samp[0]>0:
				udata.append([*samp, *samp])
		#print(np.vstack(udata).shape)
		dt.append(np.vstack(udata))#np.vstack(udata)
		
		if j%60000==0:
			with open('senegal_trajs_million'+str(int(j/60000))+'.pickle', 'wb') as handle:
				pickle.dump(dt, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
			del dt
			dt = []
		
		j=j+1
	with open('senegal_trajs_million'+str(int(j/60000)+1)+'.pickle', 'wb') as handle:
		pickle.dump(dt, handle, protocol=pickle.HIGHEST_PROTOCOL)

if not already_processed:
	files = glob.glob('/Users/abhimishra/Downloads/d4d/senegal/SET2/SET*.CSV')

	user_data=[]
	for j in tqdm(range(len(files))):
		with open('senegal_data_' + str(j) + '.pickle', 'rb') as handle:
			user_data.append( pickle.load(handle) )

	loc_file = glob.glob('/Users/abhimishra/Downloads/d4d/senegal/ContextData/SITE_ARR_LONLAT.CSV')

	loc_data = pd.read_csv(loc_file[0])
	print(loc_data)

	with open('../SenegalData/senegal_antennas.pickle', 'rb') as handle:
		antennas = pickle.load(handle)

	print(antennas)
	
	add_cloumns_data()
	
	print(user_data[0])

	for j in tqdm(range(len(files))):
		user_data[j].drop(["lon", "lat", "arr_id", "wkt", "geometry"], axis=1, inplace=True)

	print(user_data[0])
	
	with open('senegal_data_processed.pickle', 'wb') as handle:
		pickle.dump(user_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[10]:


# if not mil_trajs:
# 	if treat_ts and not shenzen_dataset:
# 		with open('../SenegalData/senegal_data_processed.pickle', 'rb') as handle:
# 			p_user_data = pickle.load(handle)


# In[11]:


sample_senegal=True
rand_sample_senegal=True #False
sub_sampled_datasets=[1,2,5,10,25,50,100,200]

# if sample_senegal:
# 	for sz in sub_sampled_datasets:
# 		num=sz*1000

# 		rand_samples=set()
# 		while len(rand_samples)<num:
# 			rand_samples.add(random.randint(0, 319365))

# 		with open('senegal_data_rand_'+str(num)+'.pickle', 'wb') as handle:
# 			pickle.dump(rand_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

# if sample_senegal:
# 	num=50000

# 	rand_samples=set()
# 	while len(rand_samples)<num:
# 		rand_samples.add(random.randint(0, 319000))

# 	with open('senegal_data_rand_'+'week21_'+str(num)+'.pickle', 'wb') as handle: #"week21_"
# 		pickle.dump(rand_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[12]:


if treat_ts and not shenzen_dataset:
	if not mil_trajs and not rand_sample_senegal:
		p_user_data=0
		ta=trajAnonymizer(p_user_data)
		
		week=0
		
# 		data = ta.getNTrajs(-1, week)#-1, 0
		
# 		with open('senegal_data_'+str(week)+'.pickle', 'wb') as handle:
# 			pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
		#data = ta.filter_anomalies(data, 1000)
		
		# 	#tdata= ta.transformToKmergeData(data, 1000)
		
		with open('senegal_data_'+str(week)+'.pickle', 'rb') as handle:
			data = pickle.load(handle)
		
		tdata= ta.transformToKmergeData2(data)
		
# 		del p_user_data
		
		chosen_k = 2#2

		tdata_full = []
		#print(tdata_full.shape)

		repeat_samples(tdata)
		del tdata
	
	elif rand_sample_senegal:
		p_user_data=0
		
		ta=trajAnonymizer(p_user_data)
		
# 		j=1
# 		num=sub_sampled_datasets[j]*1000
		
# 		with open('senegal_data_rand_'+str(num)+'.pickle', 'rb') as handle:
# 			rand_samples = pickle.load(handle)

		num=5000 #5000
		
		with open('senegal_data_rand_'+str(num)+'.pickle', 'rb') as handle:
			rand_samples = pickle.load(handle)
		
		print(len(rand_samples))
		
# 		print(rand_samples)
		data = ta.getNTrajsRand(rand_samples, 0)#-1, 0
		#data = ta.filter_anomalies(data, 1000)
		
		# 	#tdata= ta.transformToKmergeData(data, 1000)
		tdata= ta.transformToKmergeData2(data)
		
		del p_user_data
		
		chosen_k = 2#2 

		tdata_full = []
		#print(tdata_full.shape)

		repeat_samples(tdata)
		del tdata
	
	else:
# 		ta=trajAnonymizer(p_user_data)
		
# # 		data = ta.getMilTrajs()
# 		data = ta.getMilTrajsCorrect()
	
# 		with open('senegal_data_million.pickle', 'wb') as handle:
# 			pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
# # 		print(data)
		
# 		with open('senegal_data_million.pickle', 'rb') as handle:
# 			data = pickle.load(handle)
		
# 		ta=trajAnonymizer(data)
# 		tdata= ta.transformToKmergeData2(data)
		
# 		del data

# 		chosen_k = 500#2

# 		tdata_full = []
# 		#print(tdata_full.shape)

# 		repeat_samples4(tdata)
# 		del tdata

# 		print(tdata_full[-1][0], len(tdata_full))


		
		files=glob.glob("./senegal_trajs_million*")
		tdata_full=[]

		j=0
		for i in tqdm(range(len(files))):
			file=glob.glob("./senegal_trajs_million"+str(i)+"*")[0]
			with open(file, 'rb') as handle:
				dt=pickle.load(handle)

				for traj in dt:
					tdata_full.append(traj)
				del dt



		# 	with h5py.File("senegal_trajs_million_comp"+str(i)+".h5", 'w') as hf:
		# 		dname="stmc" + str(i)
		# 		hf.create_dataset(dname,  data=tdata_full)
		# 	del tdata_full

		# tdata_full=np.vstack(tdata_full)


		num_trajs = len(tdata_full)
		print(num_trajs, len(tdata_full), tdata_full[-1][0])
# 		pass

if shenzen_dataset:
	with open('shenzen_trajs.pickle', 'rb') as handle:
		shenzen_data = pickle.load(handle)

	ta1=trajAnonymizer(shenzen_data)
	tdata1= ta1.transformToKmergeData3(shenzen_data)

	tdata_full_shenzen = []

	repeat_samples2(tdata1)

	print(tdata_full_shenzen[-1][0])

	del shenzen_data, tdata1
	print(tdata_full_shenzen[0][0])


# In[13]:


print(len(tdata_full), tdata_full[-1][0])

# # # print(data)

# # # del data


# In[14]:


import matplotlib.pyplot as plt
from scipy import spatial
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

# %matplotlib inline

# fig, ax = plt.subplots(1,1,figsize=(20,10))

# for traj in trajs:
#     traj_ordered = traj[traj[:,0, 2].argsort()]  # sort by time
#     ax.plot(traj_ordered[:, 0, 0], 
#             traj_ordered[:, 0, 1], 
#             marker='o', linewidth=0.5)
    
# ax.spines.right.set_visible(False)
# ax.spines.top.set_visible(False)
# ax.spines.bottom.set_visible(False)
# ax.spines.left.set_visible(False)

# ax.set_title('Individual trajectories')


# plt.show()


# # Initializing problem

# In[15]:


loose=False #False

gdh = glove_data_handler(tdata_full, loose)#trajs,[:25]
index_not_sol = []
k_list = np.ones(len(tdata_full)) #trajs, [:25]


# # Solving

# In[17]:


# def solve(self, c=None):
# 	c = c if c else Chrono('Solving with Glove...')
	
# 	while len(self.costs)>0:  # while there are still trajectories to merge

# 		i, j, cost = self.get_cheapest_merge()

# # 		if len(self.costs)%10==0:
# # 			c.tprint(f'{len(self.costs):<3} to go : merged {i:>3} (k={self.k_list[i]:.0f}) & {j:<3} (k={self.k_list[j]:.0f}) into {len(self.trajs):<3} for cost {cost:.2f}')

# 		index_not_sol += [i,j]  # keep track of what has been discarded

# 		self.add_merged(i, j)  # add the new trajectory, inactive if above the threshold

# 		self.delete_indices([i,j])  # discard used trajectories

# 	sol_mask = np.ones(len(self.trajs), dtype=bool)
# 	sol_mask[index_not_sol] = False
# 	c.tprint('Aggregation complete !')        
# 	return self.trajs[sol_mask], self.k_list[sol_mask]

# c = Chrono('Gloving...') # CHRONO WAS PREVIOUSLY THERE! 

while len(gdh.costs)>0:  # while there are still trajectories to merge

	i, j, cost = gdh.get_cheapest_merge()

# 	c.tprint(f'{len(gdh.costs):<3} to go : merged {i:>3} (k={k_list[i]:.0f}) & {j:<3} (k={k_list[j]:.0f}) for cost {cost:.2f}')

	new_k = k_list[i]+k_list[j]
	k_list = np.concatenate([k_list, [new_k]])

	index_not_sol += [i,j]  # keep track of what has been discarded


	if not loose:
		gdh.add_merged(i, j, new_k <= anon_thres)  # add the new trajectory, inactive if above the threshold
	else:
		gdh.add_merged(i, j, True)

	gdh.delete_indices([i,j])  # discard used trajectories

# c.done()


# # Analyzing solution

# In[35]:


sol_mask = np.ones(len(gdh.trajs), dtype=bool)
sol_mask[index_not_sol] = False


# In[36]:


print('trajs_ori : ', len(tdata_full))
print('trajs : ', len(gdh.trajs))
print('nb of agg trajs : ', sol_mask.sum())
print('volume after agg : ', k_list[sol_mask].sum())


# In[145]:


# # num_trajs=25
# print(sol_mask, index_not_sol, len(gdh.trajs), len(np.array(gdh.trajs)[sol_mask]))

# # print(np.array(gdh.trajs)[sol_mask][0])

# for gtraj in np.array(gdh.trajs)[sol_mask]:
# 	print(gtraj[0])
# 	break


# In[146]:


# print(gdh.gen_traj.keys())


# In[147]:


num_trajs=5000 #1000 

if not loose:
	file = open("generalised_glove/size" + str(num_trajs) + ".txt", "w") #loose_
else:
	file = open("generalised_glove/loose_size" + str(num_trajs) + ".txt", "w") #loose_

trace=""
for gtraj in np.array(gdh.trajs)[sol_mask]:
# 	print(gtraj[0][4:])
	if gtraj[0][0]!=0 or gtraj[1][0]!=0:
		flag=0
		for samp in gtraj:
			for x in samp[:3]:
				if int(x)!=0:
					trace=trace + ' ' + str(int(x))
				else:
					flag=1
			for x in samp[4:-1]:
				if int(x)!=0:
					trace=trace + ' ' + str(int(x))
				else:
					flag=1
			if flag==1:
				break
			trace = trace + '\n'
	
	trace=trace+'\n'
# 	print(trace)

file.write(trace)#this_results


# In[3]:


# import matplotlib.pyplot as plt
# from scipy import spatial
# from matplotlib.collections import PatchCollection
# from matplotlib.patches import Rectangle
# %matplotlib inline

# nb_gen_traj_to_plot = 4

# fig, axes = plt.subplots(2,1,figsize=(15,20))
# ax = axes[0]
# ax.set_prop_cycle(color=[
#     '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
#     '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
#     '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
#     '#17becf', '#9edae5'])
# nb_ori_traj_plotted = 0
# for itraj in np.arange(len(gdh.trajs))[sol_mask][:nb_gen_traj_to_plot]:
#     nb_ori_traj_plotted += len(gdh.original_content[itraj])
#     # print points of each original trajectory that make up the generalized trajectory
#     for ori_traj in trajs[gdh.original_content[itraj]]:
#         traj_ordered = ori_traj[ori_traj[:,0, 2].argsort()]  # sort by time
#         ax.plot(traj_ordered[:, 0, 0], 
#                 traj_ordered[:, 0, 1], 
#                 marker='o', linewidth=1)

    
# ax.spines.right.set_visible(False)
# ax.spines.top.set_visible(False)
# ax.spines.bottom.set_visible(False)
# ax.spines.left.set_visible(False)

# ax.set_title('{} individual trajectories'.format(nb_ori_traj_plotted))

# ax = axes[1]

# for itraj in np.arange(len(gdh.trajs))[sol_mask][:nb_gen_traj_to_plot]:
#     traj = gdh.trajs[itraj]
    
#     traj_nopad = traj[traj[:,0,2]>0]
    
#     traj_ordered = traj_nopad[traj_nopad[:, 0, 2].argsort()]
    
#     plottraj = ax.plot((traj_ordered[:,1,0]+traj_ordered[:,0,0])/2,  # middle of the box
#                        (traj_ordered[:,1,1]+traj_ordered[:,0,1])/2, 
#                         linewidth=0.5)
    
#     # print generalized zone of each point
#     rectangles = []
#     for point in traj_nopad:
#         if (point[1]-point[0]).sum()>0:
#             rectangles += [Rectangle((point[0,0], point[0,1]), 
#                                       point[1,0]-point[0,0], 
#                                       point[1,1]-point[0,1])]
#     pc = PatchCollection(rectangles, facecolor=plottraj[-1].get_color(), alpha=0.5)
#     ax.add_collection(pc)
    
#     # print points of each original trajectory that make up the generalized trajectory
#     for ori_traj in trajs[gdh.original_content[itraj]]:
#         traj_ordered = ori_traj[ori_traj[:,0, 2].argsort()]  # sort by time
#         ax.plot(traj_ordered[:, 0, 0], 
#                 traj_ordered[:, 0, 1], 
#                 marker='o', linewidth=0, color = plottraj[-1].get_color())

# ax.spines.right.set_visible(False)
# ax.spines.top.set_visible(False)
# ax.spines.bottom.set_visible(False)
# ax.spines.left.set_visible(False)
# ax.set_title('{} aggregated trajectories'.format(nb_gen_traj_to_plot))
# plt.savefig('/Users/benoit/Desktop/glove.png')

# plt.show()


# In[149]:




# fig, ax = plt.subplots(1,1,figsize=(10,10))

# ax.set_prop_cycle(color=[
#     '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
#     '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
#     '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
#     '#17becf', '#9edae5'])
# nb_ori_traj_plotted = 0
# for itraj in np.arange(len(gdh.trajs))[sol_mask][:nb_gen_traj_to_plot]:
#     nb_ori_traj_plotted += len(gdh.original_content[itraj])
#     # print points of each original trajectory that make up the generalized trajectory
#     for ori_traj in trajs[gdh.original_content[itraj]]:
#         traj_ordered = ori_traj[ori_traj[:,0, 2].argsort()]  # sort by time
#         ax.plot(traj_ordered[:, 0, 0], 
#                 traj_ordered[:, 0, 1], 
#                 marker='o', linewidth=1, c='gray')

    
# ax.spines.right.set_visible(False)
# ax.spines.top.set_visible(False)
# ax.spines.bottom.set_visible(False)
# ax.spines.left.set_visible(False)

# ax.set_title('{} individual trajectories'.format(nb_ori_traj_plotted))
# plt.savefig('/Users/benoit/Desktop/gloveori.png')


# plt.show()


# In[150]:




# fig, ax = plt.subplots(1,1,figsize=(10,10))

# for itraj in np.arange(len(gdh.trajs))[sol_mask][:nb_gen_traj_to_plot]:
#     traj = gdh.trajs[itraj]
    
#     traj_nopad = traj[traj[:,0,2]>0]
    
#     traj_ordered = traj_nopad[traj_nopad[:, 0, 2].argsort()]
    
#     plottraj = ax.plot((traj_ordered[:,1,0]+traj_ordered[:,0,0])/2,  # middle of the box
#                        (traj_ordered[:,1,1]+traj_ordered[:,0,1])/2, 
#                         linewidth=0.5)
    
#     # print generalized zone of each point
#     rectangles = []
#     for point in traj_nopad:
#         if (point[1]-point[0]).sum()>0:
#             rectangles += [Rectangle((point[0,0], point[0,1]), 
#                                       point[1,0]-point[0,0], 
#                                       point[1,1]-point[0,1])]
#     pc = PatchCollection(rectangles, facecolor=plottraj[-1].get_color(), alpha=0.5)
#     ax.add_collection(pc)
    
#     # print points of each original trajectory that make up the generalized trajectory
#     for ori_traj in trajs[gdh.original_content[itraj]]:
#         traj_ordered = ori_traj[ori_traj[:,0, 2].argsort()]  # sort by time
#         ax.plot(traj_ordered[:, 0, 0], 
#                 traj_ordered[:, 0, 1], 
#                 marker='o', linewidth=0, color = plottraj[-1].get_color())

# ax.spines.right.set_visible(False)
# ax.spines.top.set_visible(False)
# ax.spines.bottom.set_visible(False)
# ax.spines.left.set_visible(False)
# ax.set_title('{} aggregated trajectories'.format(nb_gen_traj_to_plot))

# plt.savefig('/Users/benoit/Desktop/gloveagg.png')

# plt.show()


# # Testing the production code

# In[151]:


# from glove.trajectory_glove import GloveAggregator


# In[152]:


# anon_thres = 6
# nb_trajs = 10
# nb_points = 10

# trajs = np.random.rand(nb_trajs, nb_points, 3) # points are [x,y,t]

# # translation wrt time so we have a pattern
# translation_speed = 10
# for k in range(trajs.shape[0]):
#     for i_point in range(trajs.shape[1]):
#         trajs[k,i_point,0] += translation_speed*trajs[k,i_point,2]  # translate along x
#         trajs[k,i_point,1] += translation_speed*np.sin(20*trajs[k,i_point,2])  # ossillate along y
        
# # Adding max bounds, equal to min bounds for the moment
# trajs = np.repeat(trajs, 2, axis=1).reshape((trajs.shape[0], trajs.shape[1], 2, 3))


# In[153]:


# ga = GloveAggregator(trajs, anon_thres)


# In[154]:


# 
# trajs_agg, k_list = ga.solve()


# In[155]:


# k_list.sum()

