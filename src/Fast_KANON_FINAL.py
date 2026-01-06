#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_cell_magic('javascript', '', '\nIPython.tab_as_tab_everywhere = function(use_tabs) {\n    if (use_tabs === undefined) {\n        use_tabs = true; \n    }\n\n    // apply setting to all current CodeMirror instances\n    IPython.notebook.get_cells().map(\n        function(c) {  return c.code_mirror.options.indentWithTabs=use_tabs;  }\n    );\n    // make sure new CodeMirror instances created in the future also use this setting\n    CodeMirror.defaults.indentWithTabs=use_tabs;\n\n    };\n\nIPython.tab_as_tab_everywhere()')


# In[2]:


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
	sns.set_context("talk")
# 	sns.set(rc={"figure.figsize":(11.7,8.27), "font.size":20,"axes.titlesize":20,"axes.labelsize":16})#'figure.figsize':(11.7,8.27),
	
defaultPlotting()

do_pickle=False
do_undirected=True


# In[3]:


#['/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P08.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P20.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P21.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P09.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P23.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P22.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P25.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P19.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P18.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P24.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P15.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P01.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P14.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P02.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P16.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P17.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P03.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P07.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P13.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P12.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P06.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P10.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P04.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P05.CSV', 
#'/Users/abhimishra/Downloads/d4d/senegal/SET2/SET2_P11.CSV']


# In[ ]:





# ## Pre-processing Trajectories

# In[4]:


from datetime import datetime, timedelta

already_processed=True
treat_ts=True
shenzen_dataset=False #True
taka_dataset=False #True
mil_trajs=True #True

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

def repeat_samples5(idata):
	global tdata_full_taka
	for traj in idata:
		udata=[]
		for samp in traj:
			#print([*samp, *samp])
			udata.append([*samp, *samp])
		#print(np.vstack(udata).shape)
		tdata_full_taka.append(np.vstack(udata))#np.vstack(udata)

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


# In[5]:


# if not mil_trajs:
# 	if treat_ts and not shenzen_dataset:
# 		with open('../SenegalData/senegal_data_processed.pickle', 'rb') as handle:
# 			p_user_data = pickle.load(handle)


# In[6]:


sample_senegal=True
rand_sample_senegal=False #False 
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


# In[7]:


if treat_ts and not shenzen_dataset and not taka_dataset:
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
		
		chosen_k = 500#2

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

		num=50000
		
		with open('senegal_data_rand_'+str(num)+'.pickle', 'rb') as handle:
			rand_samples = pickle.load(handle)
		
		print(len(rand_samples))
		
# 		print(rand_samples)
		data = ta.getNTrajsRand(rand_samples, 0)#-1, 0
		#data = ta.filter_anomalies(data, 1000)
		
		# 	#tdata= ta.transformToKmergeData(data, 1000)
		tdata= ta.transformToKmergeData2(data)
		
		del p_user_data
		
		chosen_k = 500#2

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


# In[8]:


# if taka_dataset:
# 	dataset1 = pd.read_csv('/home/tribe/mabhishe/myspace/YJ_Dataset/yjmob100k-dataset1.csv')


# In[9]:


# print(dataset1)

# def transform_dataset_taka(dataset):
# 	# Calculate the timestamp in seconds and add as "ts" column
# 	dataset['ts'] = dataset['d'] * 24 * 3600 + dataset['t'] * 1800
# 	dataset['ts'] = dataset['ts'].astype(int)  # Keep timestamp as integer

# 	# Convert grid (x, y) to meters based on the center of each 500m x 500m grid cell
# 	grid_size_meters = 500
# 	dataset['x'] = (dataset['x'] - 1) * grid_size_meters + grid_size_meters / 2
# 	dataset['y'] = (dataset['y'] - 1) * grid_size_meters + grid_size_meters / 2

# 	# Drop original "d" and "t" columns
# 	dataset = dataset.drop(columns=['d', 't'])

# 	return dataset

# dataset1 = transform_dataset_taka(dataset1)
# print(dataset1)


# In[10]:


# import pandas as pd

# def filter_dataset_by_time_gap(dataset, time_limit=14 * 24 * 3600):
# 	"""
# 	Filters out rows where the timestamp ('ts') difference from the first 
# 	timestamp of each 'uid' exceeds the specified time limit.

# 	Parameters:
# 	- dataset: DataFrame containing 'uid' and 'ts' columns.
# 	- time_limit: Maximum allowed time difference in seconds (default is 2 weeks).

# 	Returns:
# 	- Filtered DataFrame with rows removed based on the time difference criterion.
# 	"""
# 	# Sort by 'uid' and 'ts' to ensure time-based calculations are correct
# 	dataset = dataset.sort_values(by=['uid', 'ts']).copy()

# 	# Calculate the time difference from the first 'ts' for each 'uid'
# 	dataset['time_gap'] = dataset.groupby('uid')['ts'].transform(lambda x: x - x.iloc[0])

# 	# Filter out rows with time difference more than 2 weeks
# 	filtered_dataset = dataset[dataset['time_gap'] <= time_limit].drop(columns=['time_gap'])

# 	return filtered_dataset

# # Apply the function to transformed_dataset1
# dataset1 = filter_dataset_by_time_gap(dataset1)

# # Verify the result
# print(dataset1)


# In[11]:


# dataset1 = dataset1.reset_index(drop=True)
# print(dataset1)

# if taka_dataset:
# 	ta1=trajAnonymizer(dataset1)
# 	tdata1= ta1.transformToKmergeData3(dataset1)

# 	tdata_full_taka = []

# 	repeat_samples5(tdata1)

# 	print(tdata_full_taka[-1][0])

# 	del dataset1, tdata1
# 	print(tdata_full_taka[0][0])


# In[12]:


# print(max([val[0][2] for val in tdata_full_taka]))


# In[13]:


# # Save the list of DataFrames to a pickle file
# with open("tdata_full_taka.pkl", "wb") as file:
# 	pickle.dump(tdata_full_taka, file)


# In[14]:


if taka_dataset:
	# Load the list of DataFrames from the pickle file
	with open("tdata_full_taka.pkl", "rb") as file:
		tdata_full_taka = pickle.load(file)
	
	print(tdata_full_taka[0][0])
	print(tdata_full_taka[-1][0])


# In[15]:


# import numpy as np

# def check_for_nan_values(trajectories):
# 	"""
# 	Checks if there are any NaN values in a list of trajectories.

# 	Parameters:
# 	- trajectories: List of trajectories where each trajectory contains samples with 8 values.

# 	Returns:
# 	- True if there is any NaN in the list, False otherwise.
# 	"""
# 	for trajectory in trajectories:
# 		if any(np.isnan(sample).any() for sample in trajectory):
# 			return True  # NaN found
# 	return False  # No NaN values found

# # Example usage
# # Assuming trajectories is a list of lists where each inner list contains samples with 8 values
# has_nan = check_for_nan_values(tdata_full_taka)
# print(f"Contains NaN values: {has_nan}")


# In[16]:


# print(len(list(set([int(val[0][3]) for val in tdata_full_taka]))))


# In[17]:


# print(data)
# # print(min(data["timestamp"]), max(data["timestamp"]), min(data["user_id"]), max(data["user_id"]))

# print(tdata_full[-1], len(tdata_full))


# In[18]:


# print(min(data["timestamp"]), max(data["timestamp"]), min(data["user_id"]), max(data["user_id"]))


# In[19]:


# l=320000

# week_start=[]

# week_start.append( (min(data[data["user_id"]<=l]["timestamp"]), max(data[data["user_id"]<=l]["timestamp"])) )
# week_start.append( (min(data[data["user_id"]>l][data["user_id"]<=2*l]["timestamp"]), max(data[data["user_id"]>l][data["user_id"]<=2*l]["timestamp"])) )
# week_start.append( (min(data[data["user_id"]>2*l][data["user_id"]<=3*l]["timestamp"]), max(data[data["user_id"]>2*l][data["user_id"]<=3*l]["timestamp"])) )
# week_start.append( (min(data[data["user_id"]>3*l]["timestamp"]), max(data[data["user_id"]>3*l]["timestamp"])) )


# In[20]:


# print(len(tdata_full), tdata_full[-1][0], tdata_full[-1].shape)

# # # # print(data)

# # # # del data


# In[21]:


# print(week_start)


# In[22]:


# with open('week_start_million.pickle', 'rb') as handle:
# 	week_data = pickle.load(handle)
# print(week_data)


# In[23]:


#1357516800
#1357516800
#1357516800
#1357516800


# In[24]:


# with open('senegal_data_rand_'+str(2000)+'.pickle', 'rb') as handle:
# 	rand_samples = pickle.load(handle)

# print(sorted(rand_samples))


# In[25]:


# l=320000

# week_start=[]

# week_start.append( (min(data[data["user_id"]<=l]["timestamp"]), max(data[data["user_id"]<=l]["timestamp"])) )
# week_start.append( (min(data[data["user_id"]>l][data["user_id"]<=2*l]["timestamp"]), max(data[data["user_id"]>l][data["user_id"]<=2*l]["timestamp"])) )
# week_start.append( (min(data[data["user_id"]>2*l][data["user_id"]<=3*l]["timestamp"]), max(data[data["user_id"]>2*l][data["user_id"]<=3*l]["timestamp"])) )
# week_start.append( (min(data[data["user_id"]>3*l]["timestamp"]), max(data[data["user_id"]>3*l]["timestamp"])) )

# with open('week_start_million.pickle', 'rb') as handle:
# 	week_data = pickle.load(handle)
# print(week_data)

# with open('week_start_million.pickle', 'wb') as handle:
# 	pickle.dump(week_start, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print(min(data[data["user_id"]<=l]["timestamp"]), max(data[data["user_id"]<=l]["timestamp"]))

# print(min(data[data["user_id"]>l][data["user_id"]<=2*l]["timestamp"]), max(data[data["user_id"]>l][data["user_id"]<=2*l]["timestamp"]))

# print(min(data[data["user_id"]>2*l][data["user_id"]<=3*l]["timestamp"]), max(data[data["user_id"]>2*l][data["user_id"]<=3*l]["timestamp"]))

# print(min(data[data["user_id"]>3*l]["timestamp"]), max(data[data["user_id"]>3*l]["timestamp"]))


# ## SafePath dataset

# In[26]:


# min_t=10**25
# avg_len=0

# with open('index_senegal.pickle', 'rb') as handle:
# 	idex_data = pickle.load(handle)

# # print(idex_data)

# for traj in tqdm(tdata_full):
# 	for samp in traj:
# 		min_t = min(min_t, int(samp[2]))
# 	avg_len = avg_len + len(traj)
# avg_len=avg_len/len(tdata_full)

# print(min_t, avg_len)

# file = open("spd_5000.txt", "w")

# max_t=0
# for traj in tqdm(tdata_full):
# 	this_traj=""
# 	for samp in traj:
# 		this_traj = this_traj + str(idex_data[int(samp[3])])+"."+str(int(samp[2])-min_t)+" " #int(samp[3])
# 		max_t = max(max_t, int(samp[2])-min_t)
# 	file.write(this_traj+"\n")

# print(max_t)

# file.close()


# In[27]:


# file = glob.glob("./MetroDataSet*")[0]

# filep = open("MetroDataSetProcessed.txt", "w")

# with open(file) as f:
# 	for line in f:
# 		if line[0]=="L":
# 			line=line.split(":")[0].split(",")
			
# 			st=""
# 			j=0
# 			for l in line:
# 				if j!=len(line)-1:
# 					st = st + l.split(".")[0][1:] + "." + l.split(".")[1][1:] + " "
# 				else:
# 					st = st + l.split(".")[0][1:] + "." + l.split(".")[1][1:]
# 				j=j+1
# 			filep.write(st+"\n")

# filep.close()


# In[28]:


# file = glob.glob("./MetroDataSet*")[0]

# filep = open("MetroDataSetProcessedUnique.txt", "w")

# utrajs=defaultdict(int)

# with open(file) as f:
# 	for line in f:
# 		if line[0]=="L":
# 			line=line.split(":")[0].split(",")
			
# 			st=""
# 			j=0
			
# 			if utrajs[str(line)]==0:
# 				for l in line:
# 					if j!=len(line)-1:
# 						st = st + l.split(".")[0][1:] + "." + l.split(".")[1][1:] + " "
# 					else:
# 						st = st + l.split(".")[0][1:] + "." + l.split(".")[1][1:]
# 					j=j+1
				
# 				filep.write(st+"\n")
# 			utrajs[str(line)]=utrajs[str(line)]+1

# filep.close()

# utrajs_all = dict(sorted(utrajs.items(), key=lambda item: item[1]))
# count=defaultdict(int)

# for key in utrajs_all.keys():
# 	count[utrajs[key]]=count[utrajs_all[key]]+1

# count = dict(sorted(count.items(), key=lambda item: item[0]))
# print(count, sum([key*count[key] for key in count.keys()]))


# ## Wait For Me dataset

# In[29]:


# min_t=10**25
# min_x=10**25
# min_y=10**25

# with open('index_senegal.pickle', 'rb') as handle:
# 	idex_data = pickle.load(handle)

# # print(idex_data)

# for traj in tqdm(tdata_full):
# 	for samp in traj:
# 		min_x = min(min_x, int(samp[0]))
# 		min_y = min(min_y, int(samp[1]))
# 		min_t = min(min_t, int(samp[2]))

# print(min_t)

# file = open("wait4me_1M.txt", "w") #5000

# j=1
# for traj in tqdm(tdata_full):
# 	this_traj=""
# 	for samp in traj:
# 		this_traj = this_traj + str(j)+"\t"+str(int(samp[2])-min_t)+"\t"+str(samp[0]-min_x)+"\t"+str(samp[1]-min_y)+"\n" #int(samp[3])
	
# 	file.write(this_traj)
# 	j=j+1

# file.close()


# ## Analyse dataset

# In[30]:


# bs_count=defaultdict(int)
# singletons=[]

# for i, traj in tqdm(enumerate(tdata_full)):
# 	a=set()
# 	for samp in traj:
# 		a.add(int(samp[3]))
# 	bs_count[i]=len(a)
# 	if len(a)==1:
# 		singletons.append(i)

# plt.figure(figsize=(12, 12))
# axi = plt.gca()
# #fig, axi = plt.subplots(1, 1)

# sns.ecdfplot(list(bs_count.values()))

# # plt.xlabel("Trajectory")
# plt.xlabel("Number of BS observed")

# # plt.ylim([0, 0.15e10])
# # plt.ylim([0, 0.15e12])
# plt.xlim([0, 60])

# plt.savefig("bs_count.pdf")


# In[31]:


# print(len(singletons))

# with open('singletons_million.pickle', 'wb') as handle:
# 	pickle.dump(singletons, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[32]:


# # #473, 682547
# # #450, 225314
# # #450, 1005060
# # #401, 492697
# # #133, 981211

# with open('fault_tajs_million.pickle', 'rb') as handle:
# 	ft=pickle.load(handle)

# print(len(ft))


# In[33]:


# idxs=ft[random.randint(0, len(ft))]

# print(idxs)
# print(tdata_full[idxs[0]])
# print(tdata_full[idxs[1]])


# ## MetroDataset

# In[34]:


# utrajs = defaultdict(int)

# def getMetroDataset(ifile):
# 	#Let's add the cross-entropy too maybe!
# 	global utrajs
	
# 	otraj=defaultdict(list)
	
# 	j=0
# 	with open(ifile) as f:
# 		for line in tqdm(f):
# # 			print(j)
# 			line=line.split(" ")
# 			for l in line:
# 				if len(l)>=3:
# 					otraj[j].append( (int(l.split(".")[0]), int(l.split(".")[1])) )
					
# 			utrajs[str(otraj[j])] = utrajs[str(otraj[j])] + 1
# 			j=j+1
# 	f.close()
# 	return otraj

# tdata_full_metro = getMetroDataset('SafePathGenralizedTrajs/MetroDataSetProcessed.txt')

# tdata_full_metro_unique = getMetroDataset('./MetroDataSetProcessedUnique.txt')


# In[35]:


# print(tdata_full_metro[list(tdata_full_metro.keys())[0]])

# print(tdata_full_metro_unique[list(tdata_full_metro_unique.keys())[0]])


# In[36]:


# # print(list(utrajs.keys())[0])

# print(utrajs[str(tdata_full_metro[0])])


# In[37]:


# all_to_one_map = defaultdict(int)

# for i, tj in enumerate(tdata_full_metro_unique.values()):
# 	all_to_one_map[str(tj)]=i

# # print(all_to_one_map[str(tdata_full_metro[10000])], str(tdata_full_metro[10000]), tdata_full_metro_unique[170])


# In[38]:


# def transformMetroDatasetToKmerge(dt):
# # 	count_id = data.user_id.value_counts().sort_index().values

# # 	data = data[["x", "y", "timestamp", "site_id"]].values

# # 	count=0

# 	fdata=[]
# # 	for nsamp in count_id:
# # 		fdata.append(data[count:count+nsamp])
# # 		count=count+nsamp
# 	kys=list(dt.keys())
# 	for j in tqdm(range(len(dt.keys()))):
# 		traj=[]
# 		for samp in dt[kys[j]]:
# 			traj.append([samp[0], samp[0], samp[1], samp[0], samp[0], samp[0], samp[1], samp[0]])
# 		traj = np.array(traj)
# # 		print(traj, traj.shape)
# 		fdata.append(traj)
	
# 	print("Converted the data to the K-merge shape.")
# 	return fdata

# tdata_full_metro_kmerge = transformMetroDatasetToKmerge(tdata_full_metro)
# tdata_full_metro_unique_kmerge = transformMetroDatasetToKmerge(tdata_full_metro_unique)


# In[39]:


# civ_processed = False
# tdata_full_civ=[]

# if not civ_processed:
# 	user_data = pd.read_csv('/home/tribe/mabhishe/myspace/civ/POS_SAMPLE_0.TSV', names=["uid", "ts", "site_id"], sep='\t')
	
# 	user_data=user_data[user_data["site_id"]!=-1]
# # 	print(user_data["site_id"])
	
# 	antennas = pd.read_csv('/home/tribe/mabhishe/myspace/civ/ANT_POS.TSV', names=['site_id', 'lon', 'lat'], sep='\t')
# 	antennas = antennas.groupby(['lon','lat']).first().reset_index()

# 	antennas['wkt'] = antennas.apply(lambda r: 'POINT({} {})'.format(r['lon'], r['lat']), axis=1)
# 	antennas['geometry'] = gpd.GeoSeries.from_wkt(antennas['wkt'])
# 	antennas = gpd.GeoDataFrame(antennas).set_crs('epsg:4326').to_crs('epsg:2165')

# 	antennas['x'] = antennas.geometry.apply(lambda p: p.x)
# 	antennas['y'] = antennas.geometry.apply(lambda p: p.y)
	
# # 	user_data.columns = ["user_id", "timestamp", "site_id"]
# 	user_data = user_data.merge(antennas, on='site_id', how="left")
# # 	print(user_data)
	
# 	user_data.drop(["lon", "lat", "site_id", "wkt", "geometry"], axis=1, inplace=True)
# 	user_data=user_data.dropna().reset_index(drop=True)
	
# 	print(user_data)
	
# 	ta1=trajAnonymizer(user_data)
# 	dts = user_data["ts"].values
# 	for i, dt in tqdm(enumerate(dts)):
# 		#print(dt)
# 		dts[i] = ta1.totimestamp(dt)
# 	user_data["ts"] = dts
# 	tdata1= ta1.transformToKmergeData3(user_data)

# 	tdata_full_civ = []

# 	repeat_samples3(tdata1)

# 	print(tdata_full_civ[-1][0])

# 	del user_data, tdata1


# 
# # K-NN

# In[40]:


# print(tdata_full_civ[:100])
# print(rcm_array_civ)


# In[41]:


from sklearn.neighbors import kneighbors_graph
# # from pynndescent import NNDescent
from collections import defaultdict
import time
import numba

rcm_dict=True
od_dict = defaultdict(tuple)
od_dict2 = defaultdict(tuple)
od_dict3 = defaultdict(tuple)
od_dict4 = defaultdict(tuple)

@numba.njit
def distance_metric1(x, y):
	x, y = x[x!=-1000], y[y!=-1000]
	l=min(len(x), len(y))
	return sum(np.abs(x[:l][::4]-y[:l][::4]) + np.abs(x[:l][1::4]-y[:l][1::4]))

@numba.njit
def distance_metric2(x, y):
	x, y = x[x!=-1000], y[y!=-1000]
	l=min(len(x), len(y))
	return np.sum(np.abs(x[:l][::4]-y[:l][::4]) + np.abs(x[:l][1::4]-y[:l][1::4]) + np.abs(x[:l][2::4]-y[:l][2::4]))

@numba.njit
def distance_metric3(x, y):
	x, y = x[x!=-1000], y[y!=-1000]
	l=min(len(x), len(y))
	return np.sum(np.sqrt((x[:l][::4]-y[:l][::4])**2 + (x[:l][1::4]-y[:l][1::4])**2) + np.abs(x[:l][2::4]-y[:l][2::4]))

@numba.njit
def distance_metric4(x, y):
	x, y = x[x!=-1000], y[y!=-1000]
	l=min(len(x), len(y))
	return np.sum((x[:l][::4]-y[:l][::4])**2 + (x[:l][1::4]-y[:l][1::4])**2 + (x[:l][2::4]-y[:l][2::4])**2)

@numba.njit
def distance_metric5(x, y):
	global rcm_dict

	if not rcm_dict:
		x, y = x[x!=-1000], y[y!=-1000]

		rcmx = [np.sum(x[::4])/(len(x)/4), np.sum(x[1::4])/(len(x)/4)]
		rcmy = [np.sum(y[::4])/(len(y)/4), np.sum(y[1::4])/(len(y)/4)]

		#print(rcmx, rcmy)

		return (np.sum (np.abs(np.array(rcmy) - np.array(rcmx))), (rcmx, rcmy) ) #np.sum (np.abs(np.array(rcmy) - np.array(rcmx))**2 )
	else:
		return np.sum (np.abs(y - x))

@numba.njit
def distance_metric6(x, y):
	x, y = x[x!=-1000], y[y!=-1000]

	rcmx = (np.sum(x[::4])/(len(x)/4), np.sum(x[1::4])/(len(x)/4))
	rcmy = (np.sum(y[::4])/(len(y)/4), np.sum(y[1::4])/(len(y)/4))

	da1 = np.diff(x[2::4].copy())
	ts_sum1 = np.sum(da1)

	da2 = np.diff(y[2::4].copy())
	#da2 = np.concatenate( ([np.mean(da2)], da2) )
	ts_sum2 = np.sum(da2)

	rgx = np.sum( ( ((x[::4]-rcmx[0]) - (x[1::4]-rcmx[1]))**2 )[1:] * da1 ) / ts_sum1
	rgy = np.sum( ( ((y[::4]-rcmy[0]) - (y[1::4]-rcmy[1]))**2 )[1:] * da2 ) / ts_sum2

	d = (np.sum((np.array(rcmy) - np.array(rcmx))**2))**0.5

	if (d < rgx + rgy and d!=0):
		a = rgx * rgx
		b = rgy * rgy

		x1 = (a - b + d * d) / (2 * d)
		z1 = x1 * x1
		y1 = (a - z1)**0.5

		if a < b:
			minr=a
		else:
			minr=b

		if (d <= abs(rgy - rgx)):
			comm = 1 - ( minr/(a+b) )
		else:
			comm = 1 - (a * np.arcsin(y1 / rgx) + b * np.arcsin(y1 / rgy) - y1 * (x1 + (z1 + b - a)**0.5))/(3.14*(a+b))
	else:
		comm = 1
	return comm

@numba.njit
def distance_metric8(x, y):
	x, y = x[x!=-1000], y[y!=-1000]

	rcmx = (np.sum(x[::4])/(len(x)/4), np.sum(x[1::4])/(len(x)/4))
	rcmy = (np.sum(y[::4])/(len(y)/4), np.sum(y[1::4])/(len(y)/4))

	da1 = np.diff(x[2::4].copy())
	ts_sum1 = np.sum(da1)

	da2 = np.diff(y[2::4].copy())
	#da2 = np.concatenate( ([np.mean(da2)], da2) )
	ts_sum2 = np.sum(da2)

	rgx = np.sum( ( ((x[::4]-rcmx[0]) - (x[1::4]-rcmx[1]))**2 )[1:] * da1 ) / ts_sum1
	rgy = np.sum( ( ((y[::4]-rcmy[0]) - (y[1::4]-rcmy[1]))**2 )[1:] * da2 ) / ts_sum2

	d = (np.sum((np.array(rcmy) - np.array(rcmx))**2))**0.5

	if (d < rgx + rgy and d!=0):
		a = rgx * rgx
		b = rgy * rgy

		x1 = (a - b + d * d) / (2 * d)
		z1 = x1 * x1
		y1 = (a - z1)**0.5

		if a < b:
			minr=a
		else:
			minr=b

		if (d <= abs(rgy - rgx)):
			comm = 1 - ( minr/(a+b) )
		else:
			comm = 1 - (a * np.arcsin(y1 / rgx) + b * np.arcsin(y1 / rgy) - y1 * (x1 + (z1 + b - a)**0.5))/(3.14*(a+b))
	else:
		comm = 1
	return comm * np.sum (np.abs(np.array(rcmy) - np.array(rcmx))**2 )

@numba.njit
def hellinger3(p, q):
	_SQRT2 = np.sqrt(2)     # sqrt(2) with default precision np.float64
	return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2 #https://gist.github.com/larsmans/3116927

@numba.njit
def distance_metric7(x, y):
	x, y = x[x!=-1000], y[y!=-1000]
	dist1, dist2 = [],[]

	for i in range(1666):
		dist1.append(0)
		dist2.append(0)

	l1 = x[3::4]
	l2 = y[3::4]

	for i in range(len(l1)):
		dist1[int(l1[i])]=dist1[int(l1[i])]+1

	for i in range(len(l2)):
		dist2[int(l2[i])]=dist2[int(l2[i])]+1

	sum1, sum2 = 0,0

	for i in range(1666):
		sum1=sum1+dist1[i]
		sum2=sum2+dist2[i]

	for i in range(1666):
		dist1[i]=dist1[i]/sum1
		dist2[i]=dist2[i]/sum2

	hd = hellinger3(np.array(dist1), np.array(dist2))
	#print(sum1, sum2, l)
	return hd

@numba.njit
def distance_metric9(x, y):
	x, y = x[x!=-1000], y[y!=-1000]
	dist1 = []

	for i in range(1666):
		dist1.append(0)

	l1 = x[3::4]
	l2 = y[3::4]

	for i in range(len(l1)):
		dist1[int(l1[i])]=dist1[int(l1[i])]+1

	sum1=0
	count=0

	for i in range(len(l2)):
		if dist1[int(l2[i])]==1:
			count = count + 1
			dist1[int(l2[i])]=0

	return - count

@numba.njit
def distance_metric10(t1, t2):
# 	x, y = x[x!=-1000], y[y!=-1000]

# 	rcmx = (np.sum(x[::4])/(len(x)/4), np.sum(x[1::4])/(len(x)/4))
# 	rcmy = (np.sum(y[::4])/(len(y)/4), np.sum(y[1::4])/(len(y)/4))

# 	da1 = np.diff(x[2::4].copy())
# 	ts_sum1 = np.sum(da1)

# 	da2 = np.diff(y[2::4].copy())
# 	#da2 = np.concatenate( ([np.mean(da2)], da2) )
# 	ts_sum2 = np.sum(da2)

# 	rgx = np.sum( ( ((x[::4]-rcmx[0]) - (x[1::4]-rcmx[1]))**2 )[1:] * da1 ) / ts_sum1
# 	rgy = np.sum( ( ((y[::4]-rcmy[0]) - (y[1::4]-rcmy[1]))**2 )[1:] * da2 ) / ts_sum2

# 	d = (np.sum((np.array(rcmy) - np.array(rcmx))**2))**0.5

# 	if (d < rgx + rgy and d!=0):
# 		a = rgx * rgx
# 		b = rgy * rgy

# 		x1 = (a - b + d * d) / (2 * d)
# 		z1 = x1 * x1
# 		y1 = (a - z1)**0.5

# 		if a < b:
# 			minr=a
# 		else:
# 			minr=b

# 		if (d <= abs(rgy - rgx)):
# 			comm = 1 - ( minr/(a+b) )
# 		else:
# 			comm = 1 - (a * np.arcsin(y1 / rgx) + b * np.arcsin(y1 / rgy) - y1 * (x1 + (z1 + b - a)**0.5))/(3.14*(a+b))
# 	else:
# 		comm = 1
# 	return comm
	rcmx, rcmy = [], []

	rcmx.append(t1[0])
	rcmx.append(t1[1])

	rcmy.append(t2[0])
	rcmy.append(t2[1])

	rgx=t1[2]
	rgy=t2[2]

	#topleft, topright, bottomleft, bottomright
	S1=((rcmx[0]-rgx, rcmx[1]+rgx), (rcmx[0]+rgx, rcmx[1]+rgx), (rcmx[0]-rgx, rcmx[1]-rgx), (rcmx[0]+rgx, rcmx[1]-rgx))
	S2=((rcmy[0]-rgy, rcmy[1]+rgy), (rcmy[0]+rgy, rcmy[1]+rgy), (rcmy[0]-rgy, rcmy[1]-rgy), (rcmy[0]+rgy, rcmy[1]-rgy))

	a1=4*rgx*rgx
	a2=4*rgy*rgy

	x_overlap = max(0, min(S1[1][0], S2[1][0]) - max(S1[0][0], S2[0][0]));
	y_overlap = max(0, min(S1[1][1], S2[1][1]) - max(S1[2][1], S2[2][1]));
	overlapArea = x_overlap * y_overlap;

	if overlapArea==0:
		return np.sum (np.abs(np.array(rcmy) - np.array(rcmx)))

	else:
		Cij = (2*overlapArea)/(a1 + a2)
		return (1-Cij)/(1+Cij)

def compareDistances(knn_array):
	global tdata_full
	this_cost=0

	knn_array = np.triu(knn_array, k=0)
	for usr in tqdm(knn_array):
		this_cost=this_cost+kmerge(tdata_full[usr.astype(bool)])[1]
	return this_cost

def compareDistances2(knn_array):
	global tdata_full_shenzen
	this_cost=0

	knn_array = np.triu(knn_array, k=0)
	for usr in tqdm(knn_array):
		this_cost=this_cost+kmerge(tdata_full_shenzen[usr.astype(bool)])[1]
	return this_cost

def compareDistances3(knn_array):
	global tdata_full_taka
	this_cost=0

	knn_array = np.triu(knn_array, k=0)
	for usr in tqdm(knn_array):
		this_cost=this_cost+kmerge(tdata_full_taka[usr.astype(bool)])[1]
	return this_cost

def compareDistancesNN(knn_array):
	global tdata_full
	this_cost=0
	ca=[]

	for usr in tqdm(knn_array):
		ba = np.full(knn_array.shape[0], False)
		th=usr[0]
# 		for i, val in enumerate(usr):
# 			if val < th:
# 				usr[i]=-1
		for val in usr:
# 			if val!=-1:
			ba[val]=True

		cost = kmerge(tdata_full[ba])[1]
		this_cost=this_cost+ cost
		ca.append(cost)
	return this_cost, ca

def compareDistancesNN2(knn_array):
	global tdata_full_shenzen
	this_cost=0
	ca=[]

	for usr in tqdm(knn_array):
		ba = np.full(knn_array.shape[0], False)
		th=usr[0]
# 		for i, val in enumerate(usr):
# 			if val < th:
# 				usr[i]=-1
		for val in usr:
# 			if val!=-1:
			ba[val]=True

		cost = kmerge(tdata_full_shenzen[ba])[1]
		this_cost=this_cost+ cost
		ca.append(cost)
	return this_cost, ca

def compareDistancesNN3(knn_array):
	global tdata_full_taka
	this_cost=0
	ca=[]

	for usr in tqdm(knn_array):
		ba = np.full(knn_array.shape[0], False)
		th=usr[0]
# 		for i, val in enumerate(usr):
# 			if val < th:
# 				usr[i]=-1
		for val in usr:
# 			if val!=-1:
			ba[val]=True

		cost = kmerge(tdata_full_taka[ba])[1]
		this_cost=this_cost+ cost
		ca.append(cost)
	return this_cost, ca

def calculate_COM():
	global tdata_full, od_dict #nodes

	rcm_array=[]
	j=0
	for usr in tdata_full:
		#usr=usr[usr!=-1000]
		xr, yr = 0, 0
		od_dict[j]=( (usr[0][0], usr[0][1]), (usr[-1][0], usr[-1][1]) )
		#print(( (usr[0][0], usr[0][1]), (usr[-1][0], usr[-1][1]) ))
		for sample in usr:
			xr = xr + sample[0]
			yr = yr + sample[1]
			#rcm = [np.sum(usr[::4])/(len(usr)/4), np.sum(usr[1::4])/(len(usr)/4)]
		xr, yr = xr/len(usr), yr/len(usr)
		#print(usr, xr, yr)
		rcm_array.append([xr, yr])
		j=j+1
	return rcm_array

def calculate_COM_Million():
	files=glob.glob("./senegal_trajs_million*")

	global od_dict #nodes, tdata_full, 

	rcm_array=[]

	j=0
	for i in range(len(files)):
		file=glob.glob("./senegal_trajs_million"+str(i)+"*")[0]
		with open(file, 'rb') as handle:
			tdata_full = pickle.load(handle)

		for usr in tdata_full:
			#usr=usr[usr!=-1000]
			xr, yr = 0, 0
			od_dict[j]=( (usr[0][0], usr[0][1]), (usr[-1][0], usr[-1][1]) )
# 			print(( (usr[0][0], usr[0][1]), (usr[-1][0], usr[-1][1]) ))
			for sample in usr:
				xr = xr + float(sample[0])
				yr = yr + float(sample[1])
				#rcm = [np.sum(usr[::4])/(len(usr)/4), np.sum(usr[1::4])/(len(usr)/4)]
			xr, yr = xr/len(usr), yr/len(usr)
			#print(usr, xr, yr)
			rcm_array.append([xr, yr])
			j=j+1
	return rcm_array

def calculate_COM2():
	global tdata_full_shenzen, od_dict2 #nodes

	rcm_array=[]
	j=0
	for usr in tdata_full_shenzen:
		#usr=usr[usr!=-1000]
		xr, yr = 0, 0
		od_dict2[j]=( (usr[0][0], usr[0][1]), (usr[-1][0], usr[-1][1]) )
		#print(( (usr[0][0], usr[0][1]), (usr[-1][0], usr[-1][1]) ))
		for sample in usr:
			xr = xr + sample[0]
			yr = yr + sample[1]
			#rcm = [np.sum(usr[::4])/(len(usr)/4), np.sum(usr[1::4])/(len(usr)/4)]
		xr, yr = xr/len(usr), yr/len(usr)
		#print(usr, xr, yr)
		rcm_array.append([xr, yr])
		j=j+1
	return rcm_array

def calculate_COM3():
	global tdata_full_civ, od_dict3 #nodes

	rcm_array=[]
	j=0
	for usr in tdata_full_civ:
		#usr=usr[usr!=-1000]
		xr, yr = 0, 0
		od_dict3[j]=( (usr[0][0], usr[0][1]), (usr[-1][0], usr[-1][1]) )
		#print(( (usr[0][0], usr[0][1]), (usr[-1][0], usr[-1][1]) ))
		for sample in usr:
			xr = xr + sample[0]
			yr = yr + sample[1]
			#rcm = [np.sum(usr[::4])/(len(usr)/4), np.sum(usr[1::4])/(len(usr)/4)]
		xr, yr = xr/len(usr), yr/len(usr)
# 		print(xr, yr)
		rcm_array.append([xr, yr])
		j=j+1
	return rcm_array

def calculate_COM4():
	global tdata_full_taka, od_dict4 #nodes

	rcm_array=[]
	j=0
	for usr in tdata_full_taka:
		#usr=usr[usr!=-1000]
		xr, yr = 0, 0
		od_dict4[j]=( (usr[0][0], usr[0][1]), (usr[-1][0], usr[-1][1]) )
		#print(( (usr[0][0], usr[0][1]), (usr[-1][0], usr[-1][1]) ))
		for sample in usr:
			xr = xr + sample[0]
			yr = yr + sample[1]
			#rcm = [np.sum(usr[::4])/(len(usr)/4), np.sum(usr[1::4])/(len(usr)/4)]
		xr, yr = xr/len(usr), yr/len(usr)
		#print(usr, xr, yr)
		rcm_array.append([xr, yr])
		j=j+1
	return rcm_array

def calculate_RG():
	global tdata_full

# 	x, y = x[x!=-1000], y[y!=-1000]

	rg_array=[]

	for usr in tqdm(tdata_full):
		usr = usr[usr[:, 2].argsort()]

		rcmx = (np.sum(usr[:,0])/(len(usr)), np.sum(usr[:,1])/(len(usr)))

		da1 = np.diff(usr[:,2].copy())
		ts_sum1 = np.sum(da1)

		rgx = np.sqrt(np.sum( ( ((usr[:,0]-rcmx[0]) - (usr[:,1]-rcmx[1]))**2 )[1:] * da1 ) / ts_sum1)
		rg_array.append(rgx)

# 	print(max(rg_array))

	return rg_array

def calculate_RG_CIV():
	global tdata_full_civ

# 	x, y = x[x!=-1000], y[y!=-1000]

	rg_array=[]

	for usr in tqdm(tdata_full_civ):
		usr = usr[usr[:, 2].argsort()]

		rcmx = (np.sum(usr[:,0])/(len(usr)), np.sum(usr[:,1])/(len(usr)))

		da1 = np.diff(usr[:,2].copy())
		ts_sum1 = np.sum(da1)

		if ts_sum1!=0:
			rgx = np.sqrt(np.sum( ( ((usr[:,0]-rcmx[0]) - (usr[:,1]-rcmx[1]))**2 )[1:] * da1 ) / ts_sum1)
			rg_array.append(rgx)
		else:
			rg_array.append(0)

# 	print(max(rg_array))

	return rg_array

def calculate_RG_Shenzen():
	global tdata_full_shenzen

# 	x, y = x[x!=-1000], y[y!=-1000]

	rg_array=[]

	for usr in tqdm(tdata_full_shenzen):
		usr = usr[usr[:, 2].argsort()]

		rcmx = (np.sum(usr[:,0])/(len(usr)), np.sum(usr[:,1])/(len(usr)))

		da1 = np.diff(usr[:,2].copy())
		ts_sum1 = np.sum(da1)

		rgx = np.sqrt(np.sum( ( ((usr[:,0]-rcmx[0]) - (usr[:,1]-rcmx[1]))**2 )[1:] * da1 ) / ts_sum1)
		rg_array.append(rgx)

# 	print(max(rg_array))

	return rg_array

# def calculate_RG_Taka():
# 	global tdata_full_taka

# # 	x, y = x[x!=-1000], y[y!=-1000]

# 	rg_array=[]

# 	for usr in tqdm(tdata_full_taka):
# 		usr = usr[usr[:, 2].argsort()]

# 		rcmx = (np.sum(usr[:,0])/(len(usr)), np.sum(usr[:,1])/(len(usr)))

# 		da1 = np.diff(usr[:,2].copy())
# 		ts_sum1 = np.sum(da1)

# 		rgx = np.sqrt(np.sum( ( ((usr[:,0]-rcmx[0]) - (usr[:,1]-rcmx[1]))**2 )[1:] * da1 ) / ts_sum1)
# 		rg_array.append(rgx)

# # 	print(max(rg_array))

# 	return rg_array

def calculate_RG_Taka():
	global tdata_full_taka

	rg_array = []

	for usr in tqdm(tdata_full_taka):
		# Sort usr based on the third column (assumed timestamp)
		usr = usr[usr[:, 2].argsort()]

		# Calculate center of mass (rcmx)
		rcmx = (np.sum(usr[:, 0]) / len(usr), np.sum(usr[:, 1]) / len(usr))

		# If there's only one entry, directly assign rgx to avoid calculations that cause NaN
		if len(usr) == 1:
			rgx = 0  # or np.nan, depending on preference
		else:
			# Calculate time differences and handle normal case
			da1 = np.diff(usr[:, 2].copy())
			ts_sum1 = np.sum(da1)

			if ts_sum1 == 0:
				rgx = np.nan  # Handle zero time difference to avoid division by zero
			else:
				rgx = np.sqrt(np.sum((((usr[:, 0] - rcmx[0]) - (usr[:, 1] - rcmx[1])) ** 2)[1:] * da1) / ts_sum1)

		rg_array.append(rgx)

	return rg_array


# In[42]:


# @numba.njit
def distance_metric_metro1(x, y):
	count=0
	
	loc_dict={}
	for i in range(68):
		loc_dict[i+1]=0
	
	for val in y[:8]:
		if val>0:
			loc_dict[val]=1
	
	l=0
	for val in x[:8]:
		if val>0:
			l=l+1
			if loc_dict[val]==1:
				count=count+1
	return (l-count)/l

def distance_metric_metro2(x, y):
	count=0
	
	loc_dict={}
	for i in range(68):
		loc_dict[i+1]=0
	
	for val in y[:8]:
		if val>0:
			loc_dict[val]=1
	
	l=0
	for val in x[:8]:
		if val>0:
			l=l+1
			if loc_dict[val]==1:
				count=count+1
	
	comp1=(l-count)/l
	
	diff=0
	for j in range(8, 16):
		diff=diff+abs(y[j]-x[j])
	
	comp2=diff/(168*8)
	return comp1 + comp2

def shape_metro_trajs(idata):
	vals=[]
	
	for key in tqdm(idata.keys()):
		val=[]
		
		for samp in idata[key]:
			val.append(samp[0])
		
		for j in range(8-len(idata[key])):
			val.append(0)
		
		for samp in idata[key]:
			val.append(samp[1])
		
		for j in range(8-len(idata[key])):
			val.append(0)
		vals.append(val)
	
	return vals #np.array(vals).reshape(1, -1)


# In[43]:


# K_SIZE=10 #500
# num=1000

# dist_strategy=2

# val_array = shape_metro_trajs(tdata_full_metro)
# # print(val_array[:5])

# start = time.time()
# A = kneighbors_graph(val_array[:num], K_SIZE, mode='connectivity', include_self=True, metric=distance_metric_metro2)

# with open('knn_matrix_metro' + str(dist_strategy) +  '.pickle', 'wb') as handle:
# 	pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)

# del A
# end = time.time()
# t2=end - start
# print("The dataset size is: ", num, " and time taken is: ", t2, "seconds")


# In[44]:


# K_SIZE=100 #500
# num=1000

# dist_strategy=2

# val_array = shape_metro_trajs(tdata_full_metro_unique)
# # print(val_array[:5])

# start = time.time()
# A = kneighbors_graph(val_array, K_SIZE, mode='connectivity', include_self=True, metric=distance_metric_metro2) #[:1000]

# with open('knn_matrix_metro_unique' + str(dist_strategy) +  '.pickle', 'wb') as handle:
# 	pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)

# del A
# end = time.time()
# t2=end - start
# print("The dataset size is: ", num, " and time taken is: ", t2, "seconds")


# In[45]:


def analyze_metro_knn(st):
	with open('knn_matrix_metro' + str(st) +  '.pickle', 'rb') as handle:
		knn_m = pickle.load(handle)

	rows, cols = knn_m.nonzero()
	
	min_d=[]
	
	for i in tqdm(range(len(rows))):
		if rows[i] != cols[i]:
			d=distance_metric_metro1(val_array[rows[i]], val_array[cols[i]])
			min_d.append(d)

			if rows[i]==1:
				print( (val_array[rows[i]], val_array[cols[i]]), rows[i], cols[i] )
	
	return min_d

def analyze_metro_knn_unique(st):
	with open('knn_matrix_metro_unique' + str(st) +  '.pickle', 'rb') as handle:
		knn_m = pickle.load(handle)

	rows, cols = knn_m.nonzero()
	
	min_d=[]
	
	for i in tqdm(range(len(rows))):
		if rows[i] != cols[i]:
			d=distance_metric_metro1(val_array[rows[i]], val_array[cols[i]])
			min_d.append(d)

			if rows[i]==1:
				print( (val_array[rows[i]], val_array[cols[i]]), rows[i], cols[i] )
	
	return min_d

def plot_distance_neighbours(dt):
	df=pd.DataFrame(dt).T
	
	df.columns = ["space match", "space match + tgap"]
	
	plt.figure(figsize=(7, 7))
	axi = plt.gca()
	#fig, axi = plt.subplots(1, 1)
	
# 	print(min_d)

	sns.ecdfplot(df)

	# plt.xlabel("Trajectory")
	plt.xlabel("Neighbour distance")
# 	plt.legend()

	# plt.ylim([0, 0.15e10])
	# plt.ylim([0, 0.15e12])
	plt.xlim([0, 1])

	plt.savefig('knn_matrix_metro.pdf')

# dt=[]
# dt.append( analyze_metro_knn(1) )
# dt.append( analyze_metro_knn(2) )

# plot_distance_neighbours(dt)


# In[46]:


def plot_distance_neighbours_unique(dt):
	df=pd.DataFrame(dt).T
	
	df.columns = ["space match", "space match + tgap"]
	
	plt.figure(figsize=(7, 7))
	axi = plt.gca()
	#fig, axi = plt.subplots(1, 1)
	
# 	print(min_d)

	sns.ecdfplot(df)

	# plt.xlabel("Trajectory")
	plt.xlabel("Neighbour distance")
# 	plt.legend()

	# plt.ylim([0, 0.15e10])
	# plt.ylim([0, 0.15e12])
	plt.xlim([0, 1])

	plt.savefig('knn_matrix_metro_unique.pdf')

# dt=[]
# dt.append( analyze_metro_knn_unique(1) )
# dt.append( analyze_metro_knn_unique(2) )

# plot_distance_neighbours_unique(dt)


# In[47]:


# num = 50000 #1000

# print(tdata_full[-1])

# rg_array = calculate_RG()

# # with open('rg_array_senegal_million.pickle', 'wb') as handle:
# # 	pickle.dump(rg_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('rg_array_senegal_partial' + str(num) +  '.pickle', 'wb') as handle:
# 	pickle.dump(rg_array, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[48]:


# with open('knn_matrix_senegal_partial_new' + str(num) +  '.pickle', 'rb') as handle:
# 	aknn=pickle.load(handle)


# In[49]:


# rows, cols = aknn.nonzero()

# K_SIZE=500
# num=50000

# rcm_array = calculate_COM()

# with open('rg_array_senegal_partial' + str(50000) +  '.pickle', 'rb') as handle:
# 	rg_array = pickle.load(handle)

# for i in tqdm(range(len(rcm_array))):
# 	rcm_array[i].append(rg_array[i])

# min_d=[]
# max_d=[]

# for i in tqdm(range(len(rows))):
# 	if i%500==1:
# 		d=distance_metric10(rcm_array[rows[i]], rcm_array[cols[i]])
# 		min_d.append(d)
# 	if i%500==499:
# 		d=distance_metric10(rcm_array[rows[i]], rcm_array[cols[i]])
# 		max_d.append(d)
# # 		print(d) #rcm_array[rows[i]], rcm_array[cols[i]], 
	
# # 	if i > 10000:
# # 		break
	
# plt.figure(figsize=(12, 12))
# axi = plt.gca()
# #fig, axi = plt.subplots(1, 1)

# sns.ecdfplot(min_d, label="Min. neighbour distance")
# sns.ecdfplot(max_d, label="Max. neighbour distance")

# # plt.xlabel("Trajectory")
# plt.xlabel("Neighbour distance")
# plt.legend()

# # plt.ylim([0, 0.15e10])
# # plt.ylim([0, 0.15e12])
# plt.xlim([0, 1])

# plt.savefig("50k_distance_metric.pdf")


# In[50]:


# with open('knn_matrix_senegal_million_new.pickle', 'rb') as handle:
# 	knn_m = pickle.load(handle)

# rows, cols = knn_m.nonzero()

# rcm_array = calculate_COM()

# with open('rg_array_senegal_million.pickle', 'rb') as handle:
# 	rg_array = pickle.load(handle)

# for i in tqdm(range(len(rcm_array))):
# 	rcm_array[i].append(rg_array[i])


# In[51]:


# min_d=[]
# max_d=[]

# for i in tqdm(range(len(rows))):
# 	if i%500==1:
# 		d=distance_metric10(rcm_array[rows[i]], rcm_array[cols[i]])
# 		min_d.append(d)
# 	if i%500==499:
# 		d=distance_metric10(rcm_array[rows[i]], rcm_array[cols[i]])
# 		max_d.append(d)
# # 		print(d) #rcm_array[rows[i]], rcm_array[cols[i]], 
	
# # 	if i > 10000:
# # 		break
	
# plt.figure(figsize=(12, 12))
# axi = plt.gca()
# #fig, axi = plt.subplots(1, 1)

# sns.ecdfplot(min_d, label="Min. neighbour distance")
# sns.ecdfplot(max_d, label="Max. neighbour distance")

# # plt.xlabel("Trajectory")
# plt.xlabel("Neighbour distance")
# plt.legend()

# # plt.ylim([0, 0.15e10])
# # plt.ylim([0, 0.15e12])
# plt.xlim([0, 1])

# plt.savefig("million_distance_metric.pdf")


# In[52]:


# print(len(tdata_full_taka[0]))


# In[53]:


# # rcm_there=False

# # if not rcm_there:
# # 	rcm_array = calculate_COM_Million()
# # 	with open('com_values_senegal_million2.pickle', 'wb') as handle: #2
# # 		pickle.dump(rcm_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # else:
# # 	with open('com_values_senegal_million2.pickle', 'rb') as handle:
# # 		rcm_array = pickle.load(handle)

# # rcm_array = calculate_COM()

# # with open('week_tests/com_values' + str(week) +  '.pickle', 'wb') as handle:
# # 	pickle.dump(rcm_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # with open('com_values_week21.pickle', 'rb') as handle:
# # 	rcm_array = pickle.load(handle)

# # print(len(rcm_array))

# # rcm_array_shenzen = calculate_COM2()
# # rcm_array_civ = calculate_COM3()

# # # with open('com_values_shenzen.pickle', 'wb') as handle:
# # # 	pickle.dump(rcm_array_shenzen, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # # with open('od_values_shenzen.pickle', 'wb') as handle:
# # # 	pickle.dump(od_dict2, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # with open('com_values_civ.pickle', 'wb') as handle:
# # 	pickle.dump(rcm_array_civ, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # with open('od_values_civ.pickle', 'wb') as handle:
# # 	pickle.dump(od_dict3, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # # Obtain various KNNs

# ck_vals=[2, 5, 10, 20, 30, 40, 50, 100, 250, 500, 1000, 2000]
# # tt=[] #267.69, 435.24, 602.01, 1008.28, 1515.34, 2388.26

# # this_fl=["com_trajectories_size10.pickle", "com_trajectories_size37.pickle", "com_trajectories_size54.pickle", "com_trajectories_size67.pickle"]

# # for file in this_fl:
# # 	with open(file, 'rb') as handle:
# # 		trajs_this = pickle.load(handle)
	
# # 	this_rcm = []
# # 	for traj in trajs_this:
# # 		this_rcm.append(rcm_array[traj])
	
# # 	for K_SIZE in ck_vals:
# # 		try:
# # 			start = time.time()
# # 			A = kneighbors_graph(this_rcm, K_SIZE, mode='connectivity', include_self=True, metric=distance_metric5)

# # # 			with open('size_knn/knn_matrix_' + file.split(".")[0].split("_")[2] + "_" + str(K_SIZE) +  '.pickle', 'wb') as handle:
# # # 				pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # 			with open('size_knn/knn_matrix_' + file.split(".")[0].split("_")[2] + "_" + str(K_SIZE) +  '.pickle', 'wb') as handle:
# # 				pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # 			del A
# # 			end = time.time()
# # 			t2=end - start
# # 			print("The KNN size is: ", K_SIZE, " and time taken is: ", t2, "seconds")
# # 		except:
# # 			print("Expected n_neighbors <= n_samples")



# # for K_SIZE in ck_vals:
# # 	start = time.time()
# # 	A = kneighbors_graph(rcm_array, K_SIZE, mode='connectivity', include_self=True, metric=distance_metric5)

# # # 			with open('size_knn/knn_matrix_' + file.split(".")[0].split("_")[2] + "_" + str(K_SIZE) +  '.pickle', 'wb') as handle:
# # # 				pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # 	with open('knn_matrix_senegal_million2' + str(K_SIZE) +  '.pickle', 'wb') as handle:
# # 		pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # 	del A
# # 	end = time.time()
# # 	t2=end - start
# # 	print("The KNN size is: ", K_SIZE, " and time taken is: ", t2, "seconds")



# # K_SIZE=500
# # for sz in sub_sampled_datasets:
# # 	num=sz*1000

# # 	with open('senegal_data_rand_'+str(num)+'.pickle', 'rb') as handle:
# # 		rand_samples = pickle.load(handle)
	
# # 	this_rcm=[]
# # 	for rs in rand_samples:
# # 		this_rcm.append(rcm_array[rs])
	
# # 	start = time.time()
# # 	A = kneighbors_graph(this_rcm, K_SIZE, mode='connectivity', include_self=True, metric=distance_metric5)

# # # 			with open('size_knn/knn_matrix_' + file.split(".")[0].split("_")[2] + "_" + str(K_SIZE) +  '.pickle', 'wb') as handle:
# # # 				pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # 	with open('knn_matrix_senegal_partial' + str(num) +  '.pickle', 'wb') as handle:
# # 		pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # 	del A
# # 	end = time.time()
# # 	t2=end - start
# # 	print("The dataset size is: ", num, " and time taken is: ", t2, "seconds")




# # # K_SIZE=500
# # nums=[1000, 2000, 5000, 10000, 25000, 50000, 100000, 200000, 300000]#1000

# # num = 100000

# # for K_SIZE in ck_vals:
# # 	with open('rg_array_senegal_partial' + str(num) +  '.pickle', 'rb') as handle:
# # 		rg_array = pickle.load(handle)
	
# # 	rcm_array = calculate_COM()[:len(rg_array)]

# # 	for i in tqdm(range(len(rcm_array))):
# # 		rcm_array[i].append(rg_array[i])

# # 	# i=random.randint(0, 10000)
# # 	# j=random.randint(0, 10000)
# # 	# print(distance_metric10(rcm_array[i], rcm_array[j]), rcm_array[i], rcm_array[j], i, j)

# # 	# print(rcm_array[0], rg_array[0], tdata_full[0])

# # 	start = time.time()
# # 	A = kneighbors_graph(rcm_array, K_SIZE, mode='connectivity', include_self=True, metric=distance_metric10)

# # 	# 			with open('size_knn/knn_matrix_' + file.split(".")[0].split("_")[2] + "_" + str(K_SIZE) +  '.pickle', 'wb') as handle:
# # 	# 				pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # 	# with open('knn_matrix_senegal_partial_new' + str(num) +  '.pickle', 'wb') as handle:
# # 	# 	pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # 	del A
# # 	end = time.time()
# # 	t2=end - start
# # 	print("The dataset size is: ", K_SIZE, " and time taken is: ", t2, "seconds")




# # K_SIZE=500

# # rcm_array = calculate_COM()

# # with open('rg_array_senegal_million.pickle', 'rb') as handle:
# # 	rg_array = pickle.load(handle)

# # for i in tqdm(range(len(rcm_array))):
# # 	rcm_array[i].append(rg_array[i])

# # # i=random.randint(0, 10000)
# # # j=random.randint(0, 10000)
# # # print(distance_metric10(rcm_array[i], rcm_array[j]), rcm_array[i], rcm_array[j], i, j)

# # # print(rcm_array[0], rg_array[0], tdata_full[0])

# # start = time.time()
# # A = kneighbors_graph(rcm_array, K_SIZE, mode='connectivity', include_self=True, metric=distance_metric10)

# # # 			with open('size_knn/knn_matrix_' + file.split(".")[0].split("_")[2] + "_" + str(K_SIZE) +  '.pickle', 'wb') as handle:
# # # 				pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # with open('knn_matrix_senegal_million_new.pickle', 'wb') as handle:
# # 	pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # del A
# # end = time.time()
# # t2=end - start
# # print("The dataset size is: ", num, " and time taken is: ", t2, "seconds")


# # K_SIZE=500

# # # rcm_array = calculate_COM()
# # rcm_array = calculate_COM2()

# # # rcm_array = calculate_COM3()

# # # with open('rg_array_senegal_million.pickle', 'rb') as handle:
# # # 	rg_array = pickle.load(handle)

# # # rg_array = calculate_RG()
# # rg_array = calculate_RG_Shenzen()

# # # rg_array = calculate_RG_CIV()

# # for i in tqdm(range(len(rcm_array))):
# # 	rcm_array[i].append(rg_array[i])

# # i=random.randint(0, 10000)
# # j=random.randint(0, 10000)
# # print(distance_metric10(rcm_array[i], rcm_array[j]), rcm_array[i], rcm_array[j], i, j)

# # print(rcm_array[0], rg_array[0], tdata_full_shenzen[0])

# # start = time.time()
# # A = kneighbors_graph(rcm_array, K_SIZE, mode='connectivity', include_self=True, metric=distance_metric10)

# # # 			with open('size_knn/knn_matrix_' + file.split(".")[0].split("_")[2] + "_" + str(K_SIZE) +  '.pickle', 'wb') as handle:
# # # 				pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # with open('knn_matrix_shenzen_new.pickle', 'wb') as handle:
# # 	pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # K_SIZE=500

# # # rcm_array = calculate_COM()
# # rcm_array = calculate_COM4()

# # # rcm_array = calculate_COM3()

# # # with open('rg_array_senegal_million.pickle', 'rb') as handle:
# # # 	rg_array = pickle.load(handle)

# # # rg_array = calculate_RG()
# # rg_array = calculate_RG_Taka()

# # # rg_array = calculate_RG_CIV()

# # for i in tqdm(range(len(rcm_array))):
# # 	rcm_array[i].append(rg_array[i])

# # i=random.randint(0, 10000)
# # j=random.randint(0, 10000)
# # print(distance_metric10(rcm_array[i], rcm_array[j]), rcm_array[i], rcm_array[j], i, j)

# # print(rcm_array[0], rg_array[0], tdata_full_taka[0])

# # start = time.time()
# # A = kneighbors_graph(rcm_array, K_SIZE, mode='connectivity', include_self=True, metric=distance_metric10)

# # # 			with open('size_knn/knn_matrix_' + file.split(".")[0].split("_")[2] + "_" + str(K_SIZE) +  '.pickle', 'wb') as handle:
# # # 				pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # with open('knn_matrix_taka_new.pickle', 'wb') as handle:
# # 	pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # del A
# # end = time.time()
# # t2=end - start
# # print("The dataset size is: ", num, " and time taken is: ", t2, "seconds")


# # K_SIZE=500
# # start = time.time()
# # A = kneighbors_graph(rcm_array, K_SIZE, mode='connectivity', include_self=True, metric=distance_metric5)

# # # 			with open('size_knn/knn_matrix_' + file.split(".")[0].split("_")[2] + "_" + str(K_SIZE) +  '.pickle', 'wb') as handle:
# # # 				pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # with open('week_tests/knn_matrix' + str(week) +  '.pickle', 'wb') as handle:
# # 	pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # del A
# # del rcm_array, tdata_full
# # end = time.time()
# # t2=end - start
# # print("The time taken is: ", t2, "seconds")


# # K_SIZE=500
# # start = time.time()
# # A = kneighbors_graph(rcm_array, K_SIZE, mode='connectivity', include_self=True, metric=distance_metric5)

# # with open('size_knn/knn_matrix_' + file.split(".")[0].split("_")[2] + "_" + str(K_SIZE) +  '.pickle', 'wb') as handle:
# # 	pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # # with open('knn_matrix_week21.pickle', 'wb') as handle:
# # # 	pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # del A
# # del rcm_array, tdata_full
# # end = time.time()
# # t2=end - start
# # print("The time taken is: ", t2, "seconds")

# # 	tt.append(t2)
# # c2 = compareDistances(A.toarray())

# # print(tt)

# # fig, ax = plt.subplots()

# # sns.lineplot(x=ck_vals, y=tt, marker='x')

# # # plt.ylim([0, 2500])
# # plt.xlim([0, 2100])
# # plt.xlabel("Num. of nearest neighbours")
# # plt.ylabel("Time to construct KNN (s)")

# # # plt.savefig("knn-sizes_shenzen.pdf")
# # plt.savefig("knn-sizes_civ.pdf")


# In[54]:


# # Find the indices of rows with NaN values in rcm_array
# nan_indices = np.where(np.isnan(rcm_array).any(axis=1))[0]

# # Print the indices
# print("Indices with NaN values:", nan_indices)


# In[55]:


# print(rcm_array[67643])


# In[56]:


# with open('knn_matrix_senegal_partial' + str(1000) +  '.pickle', 'rb') as handle:
# 	knn_m_partial = pickle.load(handle)

# rows1, cols1 = knn_m_partial.nonzero()

# print(rg_array)


# In[57]:


# print(cols1, len(cols1), rows1)


# In[58]:


# fig, ax = plt.subplots(1,2, sharey=True) #figsize=(3.54, 2.5)

# # plt.rcParams.update({'font.size': 16})

# # plt.rcParams["font.size"] = "17"  # Adjust font size for better readability
# # plt.rc('axes', titlesize=17)     # Font size of the title
# # plt.rc('axes', labelsize=17)     # Font size of the x and y labels
# # plt.rc('xtick', labelsize=17)    # Font size of the x tick labels
# # plt.rc('ytick', labelsize=17)    # Font size of the y tick labels
# # plt.rc('legend', fontsize=17)    # Font size of the legend
# # plt.rc('figure', titlesize=17)   # Font size of the figure title

# # ck_vals = [10, 25, 50, 100, 200, 300, 1000]
# # tt1 = [27.1, 101.8, 322.5, 1022.3, 3087.4, 5045.7, 21431.2]

# ck_vals = [10, 100, 200, 300, 1000]
# tt1 = [27.1, 1222.3, 3087.4, 5045.7, 21431.2]

# sns.lineplot(x=ck_vals, y=np.array(tt1)/3600, marker='o', markersize=5, palette=["black"], ax=ax[1])

# # plt.ylim([0, 2500])
# # plt.xlim([0, 2100])
# ax[1].set_xticks(ticks=[10, 100, 200, 300, 1000], labels=["10k", "100k", "200k", "300k", "1M"], rotation=90)
# ax[1].set_xlabel("Num. of trajectories")
# ax[0].set_ylabel("KNN build time(Hr.)")

# # # plt.savefig("knn-sizes_shenzen.pdf")
# # plt.tight_layout()
# # plt.savefig("knn_time_final.pdf")

# # fig, ax = plt.subplots()

# # ck_vals = [2, 5, 10, 50, 100, 250, 500, 1000, 2000]
# # tt2 = [499.6, 557.8, 627.3, 906.7, 1478.7, 2886.3, 5045.7, 9028.7, 15542.6]

# ck_vals = [10, 250, 500, 1000, 2000]
# tt2 = [627.3, 2886.3, 5045.7, 9028.7, 15542.6] 

# sns.lineplot(x=ck_vals, y=np.array(tt2)/3600, marker='o', markersize=5, palette=["black"], ax=ax[0])

# ax[0].set_xticks(ticks=[10, 250, 500, 1000, 2000], labels=[10, 250, 500, "1k", "2k"], rotation=90)  

# # plt.ylim([0, 2500])
# # plt.xlim([0, 2100])
# ax[0].set_xlabel("Num. of neighbours")
# # plt.ylabel("KNN build time(Hr.)")

# ax[0].xaxis.set_label_coords(0.5, -0.40)  # Adjust x-axis label position
# ax[1].xaxis.set_label_coords(0.5, -0.40)  # Adjust x-axis label position

# plt.tight_layout()

# # plt.savefig("knn-sizes_shenzen.pdf")
# plt.savefig("knn_sizes_final.pdf")


# ## Choosing the distance metric

# In[59]:


from multiprocessing import Process
from multiprocessing import Pool, freeze_support
from itertools import repeat
import cProfile

# def compareDistances2NN(knn_array):
# 	global tdata_full
# 	this_cost=0
# 	ca=[]
	
# 	for usr in tqdm(knn_array):
# 		ba = np.full(knn_array.shape[0], False)
# 		th=usr[0]
# # 		for i, val in enumerate(usr):
# # 			if val < th:
# # 				usr[i]=-1
# 		for val in usr:
# # 			if val!=-1:
# 			ba[val]=True

# 		cost = kmerge(tdata_full[ba])[1]
# 		this_cost=this_cost+ cost
# 		ca.append(cost)
# 	return this_cost, ca

# chosen_k=2
# cost_array=[]
# dist_array=[]

# dp=["d5"]#["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9"]
# dispatcher = {"d1": [distance_metric1], "d2": [distance_metric2], "d3": [distance_metric3], "d4": [distance_metric4], "d5": [distance_metric5], "d6": [distance_metric6], "d7": [distance_metric7], "d8": [distance_metric8], "d9": [distance_metric9]}

# def get_node_pairs(this_pair, node_pair, dm, i, j):
# 	global chosen_k, tdata_full, dispatcher

# 	#print(this_pair)

# 	costs = kmerge(this_pair)[1] #the new cost
# # 	for f in dispatcher[dm]:
# # 		d=f(node_pair[0], node_pair[1])
# # 		distances = d
		
# 	#print(i, j, costs, distances)

# 	return costs#(costs, distances)


# ## K-Merge

# In[60]:


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

def get_node_trajs_m(this_trajs, k):
	return kmerge2(this_trajs, k)[0]


# In[61]:


def stretch_cost3(trajs, i, j): #For the metro, semantic dataset.
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
		stretch_cost = Delta_t * Binary spatial cost
	"""

	# Delta_t = max_t - min_t
	delta_t = (trajs[i,6] - trajs[j,2])  # necessarily in this order bc trajs is ordered by time
	
# 	print(i, j, trajs[j:i+1], trajs[j:i+1, 4:6], trajs[j:i+1, 0:2], trajs[j:i+1][:-1]) #trajs[j:i+1, 4], trajs[j:i+1, 4]
	
	if len(trajs[j:i+1])==1:
		all_Samp=trajs[j:i+1][0]
		this_samp=trajs[j:i+1][0]
	else:
		all_Samp=[tr[0] for tr in trajs[j:i+1][:-1]]
		this_samp=trajs[j:i+1][-1][0]
	
	#Notes: The last sample is perhaps the non-generalized sample@
	#We have to merge the first n-1 to the last one. CHECK!
	
	#i>j
	#i is the index of the new sample while the index j represents the previous partition linked to it. 

	if this_samp in all_Samp:#trajs[j:i+1, 4][0]==trajs[j:i+1, 4][0]:
		delta_space = 0
	else:
		delta_space = 1

	return (1+delta_t)*(1+delta_space)

def kmerge3(trajs, k): #For the metro, semantic dataset.
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
# 	print(trajs.shape[0])
	
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
			jicost = stretch_cost3(trajs, i, j)
			costs[i] = jicost + prev_cost

		elif not len(set(id_array[partition[j-1]:j]))==k or not time_coherence[j]:
			# happens when making a new complete cluster containing the i-th element
			# implies taking too many elements from the previous clusters
			# or breaking time coherence with previous cluster
			j = partition[j-1]
			jicost = stretch_cost3(trajs, i, j)
			costs[i] = jicost + prev_cost
			partition[j:i+1] = j

		while j>=0 and len(set(id_array[partition[j-1]:j]))==k and time_coherence[j]:  
			# while we break neither completeness of previous cluster, nor time coherence
			#print(len(set(id_array[partition[j-1]:j]))==k)

			jicost = stretch_cost3(trajs, i, j)
			if costs[i] > jicost + prev_cost:
				costs[i] = jicost + prev_cost
				partition[j:i+1] = j
			j = j-1


	generalized_traj = []#np.zeros((nb_points, 2, 4)) #Maybe we represent the trajs. by three tuple: set of loactions
													#representing the generalised location, min. timestamp and a 
													#max. timestamp 
	for m in range(len(np.unique(partition))):
		generalized_traj.append([[], 10**20, -10**20])
	
# 	print(np.unique(partition))
	
	for i, cluster_id in enumerate(np.unique(partition)):
# 		print(cluster_id, trajs[partition==cluster_id])
# 		print(cluster_id, [trajs[partition==cluster_id][p][2] for p in range(len(trajs[partition==cluster_id]))])
		
# 		generalized_traj[i,0] = trajs[partition==cluster_id, 0:4].min(axis=0)
# 		generalized_traj[i,1] = trajs[partition==cluster_id, 4:8].max(axis=0)
		generalized_traj[i][0].extend([trajs[partition==cluster_id][p][0] for p in range(len(trajs[partition==cluster_id]))])
		
		generalized_traj[i][1] = min([generalized_traj[i][1]] + [trajs[partition==cluster_id][p][2] for p in range(len(trajs[partition==cluster_id]))])
		
		generalized_traj[i][2] = max([generalized_traj[i][2]] + [trajs[partition==cluster_id][p][2] for p in range(len(trajs[partition==cluster_id]))])
		
		generalized_traj[i][0] = list(set(generalized_traj[i][0]))
	
# 	print(trajs, generalized_traj)
	return generalized_traj, costs[-1]/np.unique(trajs[:, -1].astype(int), return_counts=True)[1]

def get_node_pairs_metro_m(this_pair, k=2):
	return kmerge3(this_pair, k)[1]

def get_node_trajs_metro_m(this_trajs, k):
	return kmerge3(this_trajs, k)


# In[62]:


# st=2

# with open('knn_matrix_metro_unique' + str(st) +  '.pickle', 'rb') as handle: #unique
# 	knn_m = pickle.load(handle)

# print("Making full K-merge matrix")

# rows, cols = knn_m.nonzero()

# def get_kmerge_pairs_cost_metro(indx):
# 	global tdata_full_metro, tdata_full_metro_kmerge, tdata_full_metro_unique, tdata_full_metro_unique_kmerge
# 	chosen_k=100
# 	val=1

# 	file1 = open("merge_cost_metro/cost" + str(indx[0]) + ".txt", "w")
# # 	file2 = open("generalised_metro/traj" + str(indx[0]) + ".txt", "w")

# 	trace=""
	
# 	for i in indx:
# 		for j in range(chosen_k):
# 			traj1=np.zeros((len(tdata_full_metro_unique_kmerge[rows[i*chosen_k + j]]), 9))
# 			traj2=np.zeros((len(tdata_full_metro_unique_kmerge[cols[i*chosen_k + j]]), 9))

# 			idd=0
# 			for k in range(len(traj1)):
# 				traj1[k] = np.append(tdata_full_metro_unique_kmerge[rows[i*chosen_k + j]][k], idd)

# 			for k in range(len(traj2)):
# 				traj2[k] = np.append(tdata_full_metro_unique_kmerge[cols[i*chosen_k + j]][k], idd+1)

# # 				tp.append( np.array([*traj1, *traj2]) )

# 			#tp.append( np.array([tdata_full[rows[i*chosen_k + j]], tdata_full[cols[i*chosen_k + j]]]) )
# 			result = get_node_trajs_metro_m(np.array([*traj1, *traj2]), 2)
# 			trace=trace + str(int(result[1][0])) + " "
# # 			trace=trace + str(result[0]) + '\n'
			
# # 			for gsamp in result[0]:
# # 				file2.write( str(gsamp[0]) + " " + str(gsamp[1]) + " " + str(gsamp[2]) + '\n' )
# # 			file2.write('\n')
# 	file1.write(trace)
	
# 	file1.close()
# # 	file2.close()
# 	return val

# # 	for idx in indx:
# # 		tp=[]
# # 		trajs_here=idx #i,j
# # 		print(tdata_full_metro_kmerge[trajs_here[0]], tdata_full_metro_kmerge[trajs_here[1]])
		
# # 		idd=0
# # 		for j in range(len(trajs_here)):
# # 			traj1=np.zeros((len(tdata_full_metro_kmerge[trajs_here[j]]), 9))

# # 			for k in range(len(traj1)):
# # 				traj1[k] = np.append(tdata_full_metro_kmerge[trajs_here[j]][k], idd)

# # 			tp.extend( [*traj1] )

# # 			idd = idd + 1
# # 		tp = np.array(tp)
# # # 					print(i, tp.shape[0], tp.shape[1], len(trajs), trajs)

# # # 				this_results = get_node_trajs_m(tp, len(trajs))

# # 		result = get_node_trajs_metro_m(tp, len(trajs_here)) #.append( get_node_trajs_m(tp, len(trajs_here)) ) #generalised_traj, cost = kmerge(trajs[[i,j]])
		
# # # 		print(result[0])

# indexes=[]
# num_cores=32
# num_trajs=len(tdata_full_metro_unique_kmerge)

# for i in range(num_cores):
# 	il = int(num_trajs/num_cores)
# 	indexes.append([j for j in range(i*il, i*il+il)])

# if indexes[-1][-1]!=num_trajs-1:
# 	indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])

# with Pool() as pool_per_key:
# 	results = pool_per_key.starmap( get_kmerge_pairs_cost_metro, zip(indexes) )

# # get_kmerge_pairs_cost_metro(indexes[1])


# In[63]:


# #For unique Metro trajectories!


# st=2

# with open('knn_matrix_metro_unique' + str(st) +  '.pickle', 'rb') as handle:
# 	knn_m = pickle.load(handle)

# print("Making full K-merge matrix")

# rows, cols = knn_m.nonzero()

# sp_accuracy=[]
# t_accuracy=[]

# def get_kmerge_pairs_cost_metro(indx):
# 	global tdata_full_metro, tdata_full_metro_kmerge, tdata_full_metro_unique, tdata_full_metro_unique_kmerge
	
# 	sp_accuracy_here=[]
# 	t_accuracy_here=[]
	
# 	chosen_k=10
# 	val=1

# 	file1 = open("merge_cost_metro/cost" + str(indx[0]) + ".txt", "w")
# 	file2 = open("generalised_metro/traj" + str(indx[0]) + ".txt", "w")

# 	trace=""
	
# 	for i in indx:
# 		for j in range(chosen_k):
# 			traj1=np.zeros((len(tdata_full_metro_unique_kmerge[rows[i*chosen_k + j]]), 9))
# 			traj2=np.zeros((len(tdata_full_metro_unique_kmerge[cols[i*chosen_k + j]]), 9))

# 			idd=0
# 			for k in range(len(traj1)):
# 				traj1[k] = np.append(tdata_full_metro_unique_kmerge[rows[i*chosen_k + j]][k], idd)

# 			for k in range(len(traj2)):
# 				traj2[k] = np.append(tdata_full_metro_unique_kmerge[cols[i*chosen_k + j]][k], idd+1)

# # 				tp.append( np.array([*traj1, *traj2]) )

# 			#tp.append( np.array([tdata_full[rows[i*chosen_k + j]], tdata_full[cols[i*chosen_k + j]]]) )
# 			result = get_node_trajs_metro_m(np.array([*traj1, *traj2]), 2)
# 			trace=trace + str(int(result[1][0])) + " "
# # 			trace=trace + str(result[0]) + '\n'
			
# 			for gsamp in result[0]:
# 				file2.write( str(gsamp[0]) + " " + str(gsamp[1]) + " " + str(gsamp[2]) + '\n' )
# 				sp_accuracy_here.append(len(gsamp[0]))
# 				t_accuracy_here.append(abs(gsamp[2] - gsamp[1]))
# 			file2.write('\n')
	
# 	file1.write(trace)
	
# 	file1.close()
# 	file2.close()
# 	return (sp_accuracy_here, t_accuracy_here)

# # 	for idx in indx:
# # 		tp=[]
# # 		trajs_here=idx #i,j
# # 		print(tdata_full_metro_kmerge[trajs_here[0]], tdata_full_metro_kmerge[trajs_here[1]])
		
# # 		idd=0
# # 		for j in range(len(trajs_here)):
# # 			traj1=np.zeros((len(tdata_full_metro_kmerge[trajs_here[j]]), 9))

# # 			for k in range(len(traj1)):
# # 				traj1[k] = np.append(tdata_full_metro_kmerge[trajs_here[j]][k], idd)

# # 			tp.extend( [*traj1] )

# # 			idd = idd + 1
# # 		tp = np.array(tp)
# # # 					print(i, tp.shape[0], tp.shape[1], len(trajs), trajs)

# # # 				this_results = get_node_trajs_m(tp, len(trajs))

# # 		result = get_node_trajs_metro_m(tp, len(trajs_here)) #.append( get_node_trajs_m(tp, len(trajs_here)) ) #generalised_traj, cost = kmerge(trajs[[i,j]])
		
# # # 		print(result[0])

# indexes=[]
# num_cores=32
# num_trajs=len(tdata_full_metro_kmerge_unique)

# for i in range(num_cores):
# 	il = int(num_trajs/num_cores)
# 	indexes.append([j for j in range(i*il, i*il+il)])

# if indexes[-1][-1]!=num_trajs-1:
# 	indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])

# with Pool() as pool_per_key:
# 	results = pool_per_key.starmap( get_kmerge_pairs_cost_metro, zip(indexes) )

# for res in results:
# 	sp_accuracy.extend(res[0])
# 	t_accuracy.extend(res[1])

# # get_kmerge_pairs_cost_metro(indexes[1])


# In[64]:


# sp_accuracy=[]
# t_accuracy=[]

# def get_kmerge_closest_pairs(indx):
# 	global tdata_full_metro, tdata_full_metro_kmerge, tdata_full_metro_unique, tdata_full_metro_unique_kmerge
	
# 	sp_accuracy_here=[]
# 	t_accuracy_here=[]
	
# 	chosen_k=10
# 	val=1

# 	file2 = open("generalised_metro/traj_closest" + str(indx[0]) + ".txt", "w")

# 	trace=""
	
# 	for i in indx:
# 		for j in range(chosen_k):
# 			if i!=cols[i*chosen_k + j]:
# 				traj1=np.zeros((len(tdata_full_metro_unique_kmerge[rows[i*chosen_k + j]]), 9))
# 				traj2=np.zeros((len(tdata_full_metro_unique_kmerge[cols[i*chosen_k + j]]), 9))

# 				idd=0
# 				for k in range(len(traj1)):
# 					traj1[k] = np.append(tdata_full_metro_unique_kmerge[rows[i*chosen_k + j]][k], idd)

# 				for k in range(len(traj2)):
# 					traj2[k] = np.append(tdata_full_metro_unique_kmerge[cols[i*chosen_k + j]][k], idd+1)

# 	# 				tp.append( np.array([*traj1, *traj2]) )

# 				#tp.append( np.array([tdata_full[rows[i*chosen_k + j]], tdata_full[cols[i*chosen_k + j]]]) )
# 				result = get_node_trajs_metro_m(np.array([*traj1, *traj2]), 2)
# 				trace=trace + str(int(result[1][0])) + " "
# 	# 			trace=trace + str(result[0]) + '\n'

# 				for gsamp in result[0]:
# 					file2.write( str(gsamp[0]) + " " + str(gsamp[1]) + " " + str(gsamp[2]) + '\n' )
# 					sp_accuracy_here.append(len(gsamp[0]))
# 					t_accuracy_here.append(abs(gsamp[2] - gsamp[1]))
# 				file2.write('\n')
			
# 				break

# 	file2.close()
# 	return (sp_accuracy_here, t_accuracy_here)

# # 	for idx in indx:
# # 		tp=[]
# # 		trajs_here=idx #i,j
# # 		print(tdata_full_metro_kmerge[trajs_here[0]], tdata_full_metro_kmerge[trajs_here[1]])
		
# # 		idd=0
# # 		for j in range(len(trajs_here)):
# # 			traj1=np.zeros((len(tdata_full_metro_kmerge[trajs_here[j]]), 9))

# # 			for k in range(len(traj1)):
# # 				traj1[k] = np.append(tdata_full_metro_kmerge[trajs_here[j]][k], idd)

# # 			tp.extend( [*traj1] )

# # 			idd = idd + 1
# # 		tp = np.array(tp)
# # # 					print(i, tp.shape[0], tp.shape[1], len(trajs), trajs)

# # # 				this_results = get_node_trajs_m(tp, len(trajs))

# # 		result = get_node_trajs_metro_m(tp, len(trajs_here)) #.append( get_node_trajs_m(tp, len(trajs_here)) ) #generalised_traj, cost = kmerge(trajs[[i,j]])
		
# # # 		print(result[0])

# indexes=[]
# num_cores=32
# num_trajs=len(tdata_full_metro_unique_kmerge)

# for i in range(num_cores):
# 	il = int(num_trajs/num_cores)
# 	indexes.append([j for j in range(i*il, i*il+il)])

# if indexes[-1][-1]!=num_trajs-1:
# 	indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])

# with Pool() as pool_per_key:
# 	results = pool_per_key.starmap( get_kmerge_closest_pairs, zip(indexes) )

# for res in results:
# 	sp_accuracy.extend(res[0])
# 	t_accuracy.extend(res[1])

# # get_kmerge_pairs_cost_metro(indexes[1])


# In[65]:


# ## fig, ax = plt.subplots()

# sns.ecdfplot(sp_accuracy)

# # plt.ylim([0, 2500])
# # plt.xlim([0, 2100])
# plt.xlabel("Num. of locations in generalised sample")
# plt.ylabel("CDF")

# # plt.savefig("knn-sizes_shenzen.pdf")
# plt.savefig("spatial_accuracy_metro.pdf")

# fig, ax = plt.subplots()

# sns.ecdfplot(t_accuracy)

# # plt.ylim([0, 2500])
# # plt.xlim([0, 2100])
# plt.xlabel("Time (units)")
# plt.ylabel("CDF")

# # plt.savefig("knn-sizes_shenzen.pdf")
# plt.savefig("temporal_accuracy_metro.pdf")


# In[66]:


# # with open('knn_matrix500.pickle', 'rb') as handle:
# # 	knn_m = pickle.load(handle)

# with open('knn_matrix_taka_new.pickle', 'rb') as handle: #500
# 	knn_m = pickle.load(handle)

# num_trajs = len(tdata_full_taka)

# # num_trajs = len(tdata_full)
# print(num_trajs)

# print("Making full K-merge matrix")

# rows, cols = knn_m.nonzero()
# chosen_k=500

# num_cores=1000 #32
# indexes=[]

# # def get_kmerge_pairs_cost(indexes):
# # 	global tdata_full, chosen_k
# # # 	file = open("kmerge_cost_random_5000_" + str(indexes[0]) + ".txt", "w")
# # 	this_results=defaultdict(tuple)

# # 	for i in indexes:
# # 		for j in range(chosen_k):
# # 			traj1=np.zeros((len(tdata_full[rows[i*chosen_k + j]]), 9))
# # 			traj2=np.zeros((len(tdata_full[cols[i*chosen_k + j]]), 9))

# # 			idd=0
# # 			for k in range(len(traj1)):
# # 				traj1[k] = np.append(tdata_full[rows[i*chosen_k + j]][k], idd)

# # 			for k in range(len(traj2)):
# # 				traj2[k] = np.append(tdata_full[cols[i*chosen_k + j]][k], idd+1)

# # # 			tp.append( np.array([*traj1, *traj2]) )

# # 			#tp.append( np.array([tdata_full[rows[i*chosen_k + j]], tdata_full[cols[i*chosen_k + j]]]) )
		
# # 			this_results[(i,j)] = get_node_pairs_m(np.array([*traj1, *traj2]))
# # # 		print(i, len(tp))
# # 	with open('final5000cost/kmerge_costs_trajs_final' + str(indexes[0]) + '.pickle', 'wb') as handle:
# # 		pickle.dump(this_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # 	return this_results

# # # trajs_cost=defaultdict(tuple)
# # with Pool() as pool_per_key:
# # 	for i in range(10):
# # 		if i!=9:
# # 			val = indexes[int(i*len(indexes)/10):int(i*len(indexes)/10) + int(len(indexes)/10)]
# # 			results = pool_per_key.starmap( get_kmerge_pairs_cost, zip(val) )
# # 			del results
# # 		else:
# # 			val = indexes[int(i*len(indexes)/10):int(i*len(indexes)/10) + int(len(indexes)/10)+1]
# # 			results = pool_per_key.starmap( get_kmerge_pairs_cost, zip(val) )
# # 			del results

# # 	for res in results:
# # 		for key in res.keys():
# # 			trajs_cost[key]=res[key]
# # 	with open('kmerge_costs_trajs_final.pickle', 'wb') as handle:
# # 		pickle.dump(trajs_cost, handle, protocol=pickle.HIGHEST_PROTOCOL)

# for i in range(num_cores):
# 	il = int(num_trajs/num_cores)
# 	indexes.append([j for j in range(i*il, i*il+il)])

# if indexes[-1][-1]!=num_trajs-1:
# 	indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])

# def get_kmerge_pairs_cost(indexes):
# 	global tdata_full_taka, chosen_k
# # 	file = open("kmerge_cost_random_5000_" + str(indexes[0]) + ".txt", "w")
# 	this_results=defaultdict(tuple)

# 	for i in indexes:
# 		for j in range(chosen_k):
# 			traj1=np.zeros((len(tdata_full_taka[rows[i*chosen_k + j]]), 9))
# 			traj2=np.zeros((len(tdata_full_taka[cols[i*chosen_k + j]]), 9))

# 			idd=0
# 			for k in range(len(traj1)):
# 				traj1[k] = np.append(tdata_full_taka[rows[i*chosen_k + j]][k], idd)

# 			for k in range(len(traj2)):
# 				traj2[k] = np.append(tdata_full_taka[cols[i*chosen_k + j]][k], idd+1)

# # 			tp.append( np.array([*traj1, *traj2]) )

# 			#tp.append( np.array([tdata_full[rows[i*chosen_k + j]], tdata_full[cols[i*chosen_k + j]]]) )
		
# 			this_results[(i,j)] = get_node_pairs_m(np.array([*traj1, *traj2]))
# # 		print(i, len(tp))
# 	with open('final5000costtakanew/kmerge_costs_trajs_final_taka' + str(indexes[0]) + '.pickle', 'wb') as handle:
# 		pickle.dump(this_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
# 	return this_results

# # trajs_cost=defaultdict(tuple)
# with Pool() as pool_per_key:
# 	for i in range(10):
# 		if i!=9:
# 			val = indexes[int(i*len(indexes)/10):int(i*len(indexes)/10) + int(len(indexes)/10)]
# 			results = pool_per_key.starmap( get_kmerge_pairs_cost, zip(val) )
# 			del results
# 		else:
# 			val = indexes[int(i*len(indexes)/10):int(i*len(indexes)/10) + int(len(indexes)/10)+1]
# 			results = pool_per_key.starmap( get_kmerge_pairs_cost, zip(val) )
# 			del results


# In[67]:


# # with open('knn_matrix500.pickle', 'rb') as handle:
# # 	knn_m = pickle.load(handle)

# with open('knn_matrix_shenzen_new.pickle', 'rb') as handle: #500
# 	knn_m = pickle.load(handle)

# num_trajs = len(tdata_full_shenzen)

# # num_trajs = len(tdata_full)
# print(num_trajs)

# print("Making full K-merge matrix")

# rows, cols = knn_m.nonzero()
# chosen_k=500

# num_cores=1000 #32
# indexes=[]

# # def get_kmerge_pairs_cost(indexes):
# # 	global tdata_full, chosen_k
# # # 	file = open("kmerge_cost_random_5000_" + str(indexes[0]) + ".txt", "w")
# # 	this_results=defaultdict(tuple)

# # 	for i in indexes:
# # 		for j in range(chosen_k):
# # 			traj1=np.zeros((len(tdata_full[rows[i*chosen_k + j]]), 9))
# # 			traj2=np.zeros((len(tdata_full[cols[i*chosen_k + j]]), 9))

# # 			idd=0
# # 			for k in range(len(traj1)):
# # 				traj1[k] = np.append(tdata_full[rows[i*chosen_k + j]][k], idd)

# # 			for k in range(len(traj2)):
# # 				traj2[k] = np.append(tdata_full[cols[i*chosen_k + j]][k], idd+1)

# # # 			tp.append( np.array([*traj1, *traj2]) )

# # 			#tp.append( np.array([tdata_full[rows[i*chosen_k + j]], tdata_full[cols[i*chosen_k + j]]]) )
		
# # 			this_results[(i,j)] = get_node_pairs_m(np.array([*traj1, *traj2]))
# # # 		print(i, len(tp))
# # 	with open('final5000cost/kmerge_costs_trajs_final' + str(indexes[0]) + '.pickle', 'wb') as handle:
# # 		pickle.dump(this_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # 	return this_results

# # # trajs_cost=defaultdict(tuple)
# # with Pool() as pool_per_key:
# # 	for i in range(10):
# # 		if i!=9:
# # 			val = indexes[int(i*len(indexes)/10):int(i*len(indexes)/10) + int(len(indexes)/10)]
# # 			results = pool_per_key.starmap( get_kmerge_pairs_cost, zip(val) )
# # 			del results
# # 		else:
# # 			val = indexes[int(i*len(indexes)/10):int(i*len(indexes)/10) + int(len(indexes)/10)+1]
# # 			results = pool_per_key.starmap( get_kmerge_pairs_cost, zip(val) )
# # 			del results

# # 	for res in results:
# # 		for key in res.keys():
# # 			trajs_cost[key]=res[key]
# # 	with open('kmerge_costs_trajs_final.pickle', 'wb') as handle:
# # 		pickle.dump(trajs_cost, handle, protocol=pickle.HIGHEST_PROTOCOL)

# for i in range(num_cores):
# 	il = int(num_trajs/num_cores)
# 	indexes.append([j for j in range(i*il, i*il+il)])

# if indexes[-1][-1]!=num_trajs-1:
# 	indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])

# def get_kmerge_pairs_cost(indexes):
# 	global tdata_full_shenzen, chosen_k
# # 	file = open("kmerge_cost_random_5000_" + str(indexes[0]) + ".txt", "w")
# 	this_results=defaultdict(tuple)

# 	for i in indexes:
# 		for j in range(chosen_k):
# 			traj1=np.zeros((len(tdata_full_shenzen[rows[i*chosen_k + j]]), 9))
# 			traj2=np.zeros((len(tdata_full_shenzen[cols[i*chosen_k + j]]), 9))

# 			idd=0
# 			for k in range(len(traj1)):
# 				traj1[k] = np.append(tdata_full_shenzen[rows[i*chosen_k + j]][k], idd)

# 			for k in range(len(traj2)):
# 				traj2[k] = np.append(tdata_full_shenzen[cols[i*chosen_k + j]][k], idd+1)

# # 			tp.append( np.array([*traj1, *traj2]) )

# 			#tp.append( np.array([tdata_full[rows[i*chosen_k + j]], tdata_full[cols[i*chosen_k + j]]]) )
		
# 			this_results[(i,j)] = get_node_pairs_m(np.array([*traj1, *traj2]))
# # 		print(i, len(tp))
# 	with open('final5000costshenzennew/kmerge_costs_trajs_final_shenzen' + str(indexes[0]) + '.pickle', 'wb') as handle:
# 		pickle.dump(this_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
# 	return this_results

# # trajs_cost=defaultdict(tuple)
# with Pool() as pool_per_key:
# 	for i in range(10):
# 		if i!=9:
# 			val = indexes[int(i*len(indexes)/10):int(i*len(indexes)/10) + int(len(indexes)/10)]
# 			results = pool_per_key.starmap( get_kmerge_pairs_cost, zip(val) )
# 			del results
# 		else:
# 			val = indexes[int(i*len(indexes)/10):int(i*len(indexes)/10) + int(len(indexes)/10)+1]
# 			results = pool_per_key.starmap( get_kmerge_pairs_cost, zip(val) )
# 			del results


# In[68]:


# # with open('knn_matrix500.pickle', 'rb') as handle:
# # 	knn_m = pickle.load(handle)

# # with open('knn_matrix_shenzen500.pickle', 'rb') as handle:
# # 	knn_m = pickle.load(handle)

# with open('knn_matrix_civ_new.pickle', 'rb') as handle:
# 	knn_m = pickle.load(handle)

# num_trajs = len(tdata_full_civ)

# # num_trajs = len(tdata_full)
# print(num_trajs)

# print("Making full K-merge matrix")

# rows, cols = knn_m.nonzero()
# chosen_k=500

# num_cores=32 #100
# indexes= [] #[43066, 43067, 43068, 43069, 43070, 43071, 43072, 43073, 43074, 43075, 43076, 43077, 43078, 43079, 43080, 43081, 43082, 43083, 43084, 43085, 43086, 43087, 43088, 43089, 43090, 43091, 43092, 43093, 43094, 43095, 43096, 43097, 43098, 43099, 43100, 43101, 43102, 43103, 43104, 43105, 43106, 43107, 43108, 43109, 43110, 43111, 43112, 43113, 43114, 43115, 43116, 43117, 43118, 43119, 43120, 43121, 43122, 43123, 43124, 43125, 43126, 43127, 43128, 43129, 43130, 43131, 43132, 43133, 43134, 43135, 43136, 43137, 43138, 43139, 43140, 43141, 43142, 43143, 43144, 43145, 43146, 43147]#[]

# for i in range(num_cores):
# 	il = int(num_trajs/num_cores)
# 	indexes.append([j for j in range(i*il, i*il+il)])

# if indexes[-1][-1]!=num_trajs-1:
# 	indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])

# def get_kmerge_pairs_cost(indexes):
# 	global tdata_full_civ, chosen_k
# 	file = open("final5000costcivnew/kmerge_costs_trajs_final" + str(indexes[0]) + ".txt", "w")
# # 	this_results=defaultdict(tuple)

# 	for i in indexes:
# 		for j in range(chosen_k):
# # 			tid=int(tdata_full_civ[i][-1])
# 			traj1=np.zeros((len(tdata_full_civ[rows[i*chosen_k + j]]), 9))
# 			traj2=np.zeros((len(tdata_full_civ[cols[i*chosen_k + j]]), 9))

# 			idd=0
# 			for k in range(len(traj1)):
# 				traj1[k] = np.append(tdata_full_civ[rows[i*chosen_k + j]][k], idd)

# 			for k in range(len(traj2)):
# 				traj2[k] = np.append(tdata_full_civ[cols[i*chosen_k + j]][k], idd+1)

# # # 			tp.append( np.array([*traj1, *traj2]) )

# # 			tp.append( np.array([tdata_full[rows[i*chosen_k + j]], tdata_full[cols[i*chosen_k + j]]]) )
		
# 			this_results = (i, j, get_node_pairs_m(np.array([*traj1, *traj2]))[0] )
# 			file.write(str(this_results))

# # 		print(i, len(tp))
# # 	with open('final5000costciv/kmerge_costs_trajs_final' + str(indexes[0]) + '.pickle', 'wb') as handle:
# # 		pickle.dump(this_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
# 	file.close()
# 	x=1
# 	return x

# # get_kmerge_pairs_cost(indexes[0])

# # trajs_cost=defaultdict(tuple)

# # get_kmerge_pairs_cost(indexes)

# with Pool() as pool_per_key:
# 	results = pool_per_key.starmap( get_kmerge_pairs_cost, zip(indexes) )


# In[69]:


# import h5py

# with open('knn_matrix_senegal_million500.pickle', 'rb') as handle:
# 	knn_m = pickle.load(handle)

# print("Making full K-merge matrix")

# num_trajs=len(tdata_full)

# rows, cols = knn_m.nonzero()
# chosen_k=500

# num_cores=1000 #100
# indexes= [] #[43066, 43067, 43068, 43069, 43070, 43071, 43072, 43073, 43074, 43075, 43076, 43077, 43078, 43079, 43080, 43081, 43082, 43083, 43084, 43085, 43086, 43087, 43088, 43089, 43090, 43091, 43092, 43093, 43094, 43095, 43096, 43097, 43098, 43099, 43100, 43101, 43102, 43103, 43104, 43105, 43106, 43107, 43108, 43109, 43110, 43111, 43112, 43113, 43114, 43115, 43116, 43117, 43118, 43119, 43120, 43121, 43122, 43123, 43124, 43125, 43126, 43127, 43128, 43129, 43130, 43131, 43132, 43133, 43134, 43135, 43136, 43137, 43138, 43139, 43140, 43141, 43142, 43143, 43144, 43145, 43146, 43147]#[]

# for i in range(num_cores):
# 	il = int(num_trajs/num_cores)
# 	indexes.append([j for j in range(i*il, i*il+il)])

# if indexes[-1][-1]!=num_trajs-1:
# 	indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])

# def get_kmerge_pairs_cost(indexes):
# 	global tdata_full, chosen_k
# 	file = open("cost_kmerge_senegal_million/kmerge_costs_trajs_million" + str(indexes[0]) + ".txt", "w")
# # 	this_results=defaultdict(tuple)

# 	for i in indexes:
# 		this_results=[]
# 		for j in range(chosen_k):
# # 			tid=int(tdata_full_civ[i][-1])
# 			idx1, idx2 = rows[i*chosen_k + j], cols[i*chosen_k + j]
			
# # 			traj1=np.zeros((len(tdata_full[idx1]), 9))
# # 			traj2=np.zeros((len(tdata_full[idx2]), 9))

# # 			idd=0
# # 			for k in range(len(traj1)):
# # 				traj1[k] = np.append(tdata_full[idx1][k], idd)

# # 			for k in range(len(traj2)):
# # 				traj2[k] = np.append(tdata_full[idx2][k], idd+1)
			
# 			tp=[]
# 			for samp in tdata_full[idx1]:
# 				tp.append([*samp,0])
			
# 			for samp in tdata_full[idx2]:
# 				tp.append([*samp,1])
			
# # 			tp.append( np.array([tdata_full[rows[i*chosen_k + j]], tdata_full[cols[i*chosen_k + j]]]) )
			
# # 			print(np.vstack(np.array(tp)))
		
# 			this_results.append( int(get_node_pairs_m(np.vstack(np.array(tp)))[0]) )#np.array([*traj1, *traj2])
# # 			print(traj1, traj2)
# 		file.write(str(this_results))

# # 		print(i, len(tp))
# # 	with open('final5000costciv/kmerge_costs_trajs_final' + str(indexes[0]) + '.pickle', 'wb') as handle:
# # 		pickle.dump(this_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
# 	file.close()
# 	x=1
# 	return x

# # get_kmerge_pairs_cost(indexes[0])

# # trajs_cost=defaultdict(tuple)

# # start = time.time()

# # with Pool() as pool_per_key:
# # 	results = pool_per_key.starmap( get_kmerge_pairs_cost, zip(indexes) )

# # end = time.time()
# # print("Time: ", end-start, " s")


# In[70]:


# with open('knn_matrix_senegal_million500.pickle', 'rb') as handle:
# 	knn_m = pickle.load(handle)

# print("Making full K-merge matrix")

# num_trajs=len(tdata_full)

# rows, cols = knn_m.nonzero()
# chosen_k=500

# num_cores=1000 #100
# indexes= [] #[43066, 43067, 43068, 43069, 43070, 43071, 43072, 43073, 43074, 43075, 43076, 43077, 43078, 43079, 43080, 43081, 43082, 43083, 43084, 43085, 43086, 43087, 43088, 43089, 43090, 43091, 43092, 43093, 43094, 43095, 43096, 43097, 43098, 43099, 43100, 43101, 43102, 43103, 43104, 43105, 43106, 43107, 43108, 43109, 43110, 43111, 43112, 43113, 43114, 43115, 43116, 43117, 43118, 43119, 43120, 43121, 43122, 43123, 43124, 43125, 43126, 43127, 43128, 43129, 43130, 43131, 43132, 43133, 43134, 43135, 43136, 43137, 43138, 43139, 43140, 43141, 43142, 43143, 43144, 43145, 43146, 43147]#[]

# for i in range(num_cores):
# 	il = int(num_trajs/num_cores)
# 	indexes.append([j for j in range(i*il, i*il+il)])

# if indexes[-1][-1]!=num_trajs-1:
# 	indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])

# def get_kmerge_pairs_cost(indexes):
# 	global tdata_full, chosen_k
# 	file = open("cost_kmerge_senegal_million/kmerge_costs_trajs_million" + str(indexes[0]) + ".txt", "w")
# # 	this_results=defaultdict(tuple)

# 	for i in indexes:
# 		this_results=[]
# 		for j in range(chosen_k):
# # 			tid=int(tdata_full_civ[i][-1])
# 			idx1, idx2 = rows[i*chosen_k + j], cols[i*chosen_k + j]
			
# # 			traj1=np.zeros((len(tdata_full[idx1]), 9))
# # 			traj2=np.zeros((len(tdata_full[idx2]), 9))

# # 			idd=0
# # 			for k in range(len(traj1)):
# # 				traj1[k] = np.append(tdata_full[idx1][k], idd)

# # 			for k in range(len(traj2)):
# # 				traj2[k] = np.append(tdata_full[idx2][k], idd+1)
			
# 			tp=[]
# 			for samp in tdata_full[idx1]:
# 				tp.append([*samp,0])
			
# 			for samp in tdata_full[idx2]:
# 				tp.append([*samp,1])
			
# # 			tp.append( np.array([tdata_full[rows[i*chosen_k + j]], tdata_full[cols[i*chosen_k + j]]]) )
			
# # 			print(np.vstack(np.array(tp)))
		
# 			this_results.append( int(get_node_pairs_m(np.vstack(np.array(tp)))[0]) )#np.array([*traj1, *traj2])
# # 			print(traj1, traj2)
# 		file.write(str(this_results))

# # 		print(i, len(tp))
# # 	with open('final5000costciv/kmerge_costs_trajs_final' + str(indexes[0]) + '.pickle', 'wb') as handle:
# # 		pickle.dump(this_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
# 	file.close()
# 	x=1
# 	return x

# get_kmerge_pairs_cost(indexes[0])

# trajs_cost=defaultdict(tuple)

# start = time.time()

# with Pool() as pool_per_key:
# 	results = pool_per_key.starmap( get_kmerge_pairs_cost, zip(indexes) )

# end = time.time()
# print("Time: ", end-start, " s")


# In[71]:


# with open('knn_matrix_senegal_million_new.pickle', 'rb') as handle:
# 	knn_m = pickle.load(handle)

# print("Making full K-merge matrix")

# num_trajs=len(tdata_full)

# rows, cols = knn_m.nonzero()
# chosen_k=500

# num_cores=1000 #100
# indexes= [] #[43066, 43067, 43068, 43069, 43070, 43071, 43072, 43073, 43074, 43075, 43076, 43077, 43078, 43079, 43080, 43081, 43082, 43083, 43084, 43085, 43086, 43087, 43088, 43089, 43090, 43091, 43092, 43093, 43094, 43095, 43096, 43097, 43098, 43099, 43100, 43101, 43102, 43103, 43104, 43105, 43106, 43107, 43108, 43109, 43110, 43111, 43112, 43113, 43114, 43115, 43116, 43117, 43118, 43119, 43120, 43121, 43122, 43123, 43124, 43125, 43126, 43127, 43128, 43129, 43130, 43131, 43132, 43133, 43134, 43135, 43136, 43137, 43138, 43139, 43140, 43141, 43142, 43143, 43144, 43145, 43146, 43147]#[]

# for i in range(num_cores):
# 	il = int(num_trajs/num_cores)
# 	indexes.append([j for j in range(i*il, i*il+il)])

# if indexes[-1][-1]!=num_trajs-1:
# 	indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])

# def get_kmerge_pairs_cost(indexes):
# 	global tdata_full, chosen_k
# 	file = open("cost_kmerge_senegal_million_new/kmerge_costs_trajs_million" + str(indexes[0]) + ".txt", "w")
# # 	this_results=defaultdict(tuple)

# 	for i in indexes:
# 		this_results=[]
# 		for j in range(chosen_k):
# # 			tid=int(tdata_full_civ[i][-1])
# 			idx1, idx2 = rows[i*chosen_k + j], cols[i*chosen_k + j]
			
# # 			traj1=np.zeros((len(tdata_full[idx1]), 9))
# # 			traj2=np.zeros((len(tdata_full[idx2]), 9))

# # 			idd=0
# # 			for k in range(len(traj1)):
# # 				traj1[k] = np.append(tdata_full[idx1][k], idd)

# # 			for k in range(len(traj2)):
# # 				traj2[k] = np.append(tdata_full[idx2][k], idd+1)
			
# 			tp=[]
# 			for samp in tdata_full[idx1]:
# 				tp.append([*samp,0])
			
# 			for samp in tdata_full[idx2]:
# 				tp.append([*samp,1])
			
# # 			tp.append( np.array([tdata_full[rows[i*chosen_k + j]], tdata_full[cols[i*chosen_k + j]]]) )
			
# # 			print(np.vstack(np.array(tp)))
		
# 			this_results.append( int(get_node_pairs_m(np.vstack(np.array(tp)))[0]) )#np.array([*traj1, *traj2])
# # 			print(traj1, traj2)
# 		file.write(str(this_results))

# # 		print(i, len(tp))
# # 	with open('final5000costciv/kmerge_costs_trajs_final' + str(indexes[0]) + '.pickle', 'wb') as handle:
# # 		pickle.dump(this_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
# 	file.close()
# 	x=1
# 	return x

# # get_kmerge_pairs_cost(indexes[0])

# # trajs_cost=defaultdict(tuple)

# start = time.time()

# with Pool() as pool_per_key:
# 	results = pool_per_key.starmap( get_kmerge_pairs_cost, zip(indexes) )

# end = time.time()
# print("Time: ", end-start, " s")


# In[72]:


# with open('compute_pairs_size_test.pickle', 'rb') as handle:
# 	compute_pairs = pickle.load(handle)

# print(len(compute_pairs))
# compute_pairs=list(compute_pairs)


# In[73]:


# # j=0
# # sz=sub_sampled_datasets[j]*1000
# # with open('knn_matrix_senegal_partial' + str(sz) +  '.pickle', 'rb') as handle:
# # 	knn_m = pickle.load(handle)

# # print("Making full K-merge matrix")

# num_trajs=len(compute_pairs)#len(tdata_full)

# # rows, cols = knn_m.nonzero()
# chosen_k=500

# num_cores=32 #100
# indexes= [] #[43066, 43067, 43068, 43069, 43070, 43071, 43072, 43073, 43074, 43075, 43076, 43077, 43078, 43079, 43080, 43081, 43082, 43083, 43084, 43085, 43086, 43087, 43088, 43089, 43090, 43091, 43092, 43093, 43094, 43095, 43096, 43097, 43098, 43099, 43100, 43101, 43102, 43103, 43104, 43105, 43106, 43107, 43108, 43109, 43110, 43111, 43112, 43113, 43114, 43115, 43116, 43117, 43118, 43119, 43120, 43121, 43122, 43123, 43124, 43125, 43126, 43127, 43128, 43129, 43130, 43131, 43132, 43133, 43134, 43135, 43136, 43137, 43138, 43139, 43140, 43141, 43142, 43143, 43144, 43145, 43146, 43147]#[]

# for i in range(num_cores):
# 	il = int(num_trajs/num_cores)
# 	indexes.append([j for j in range(i*il, i*il+il)])

# if indexes[-1][-1]!=num_trajs-1:
# 	indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])

# def get_kmerge_pairs_cost(indexes):
# 	global tdata_full, compute_pairs
# 	file = open("cost_size_test/kmerge_costs_trajs" + str(indexes[0]) + ".txt", "w")
# # 	this_results=defaultdict(tuple)

# 	for i in indexes:
# 		pair=compute_pairs[i]
# 		idx1, idx2 = pair[0], pair[1]

# 		tp=[]
# 		for samp in tdata_full[idx1]:
# 			tp.append([*samp,0])

# 		for samp in tdata_full[idx2]:
# 			tp.append([*samp,1])

# 		this_results = int(get_node_pairs_m(np.vstack(np.array(tp)))[0]) #np.array([*traj1, *traj2])
# # 			print(traj1, traj2)
# 		file.write(str(pair) + " " + str(this_results) + " ")
# 	file.close()
# 	x=1
# 	return x

# # # get_kmerge_pairs_cost(indexes[0])

# # trajs_cost=defaultdict(tuple)

# start = time.time()

# with Pool() as pool_per_key:
# 	results = pool_per_key.starmap( get_kmerge_pairs_cost, zip(indexes) )

# end = time.time()
# print("Time: ", end-start, " s")


# In[74]:


# # j=0
# # sz=sub_sampled_datasets[j]*1000
# # with open('knn_matrix_senegal_partial' + str(sz) +  '.pickle', 'rb') as handle:
# # 	knn_m = pickle.load(handle)

# with open('knn_matrix_week21.pickle', 'rb') as handle:
# 	knn_m = pickle.load(handle)

# # print("Making full K-merge matrix")

# num_trajs=len(tdata_full)

# rows, cols = knn_m.nonzero()
# chosen_k=500

# num_cores=32 #100
# indexes= [] #[43066, 43067, 43068, 43069, 43070, 43071, 43072, 43073, 43074, 43075, 43076, 43077, 43078, 43079, 43080, 43081, 43082, 43083, 43084, 43085, 43086, 43087, 43088, 43089, 43090, 43091, 43092, 43093, 43094, 43095, 43096, 43097, 43098, 43099, 43100, 43101, 43102, 43103, 43104, 43105, 43106, 43107, 43108, 43109, 43110, 43111, 43112, 43113, 43114, 43115, 43116, 43117, 43118, 43119, 43120, 43121, 43122, 43123, 43124, 43125, 43126, 43127, 43128, 43129, 43130, 43131, 43132, 43133, 43134, 43135, 43136, 43137, 43138, 43139, 43140, 43141, 43142, 43143, 43144, 43145, 43146, 43147]#[]

# for i in range(num_cores):
# 	il = int(num_trajs/num_cores)
# 	indexes.append([j for j in range(i*il, i*il+il)])

# if indexes[-1][-1]!=num_trajs-1:
# 	indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])

# def get_kmerge_pairs_cost(indexes):
# 	global tdata_full, chosen_k
# 	file = open("cost_week21/kmerge_costs_trajs" + str(indexes[0]) + ".txt", "w")
# # 	this_results=defaultdict(tuple)

# 	for i in indexes:
# 		for j in range(chosen_k):
# # 			tid=int(tdata_full_civ[i][-1])
# 			idx1, idx2 = rows[i*chosen_k + j], cols[i*chosen_k + j]

# 			tp=[]
# 			for samp in tdata_full[idx1]:
# 				tp.append([*samp,0])

# 			for samp in tdata_full[idx2]:
# 				tp.append([*samp,1])

# 			this_results = int(get_node_pairs_m(np.vstack(np.array(tp)))[0]) #np.array([*traj1, *traj2])
# 	# 			print(traj1, traj2)
# 			file.write(str(this_results) + " ")
# 	file.close()
# 	x=1
# 	return x

# # # get_kmerge_pairs_cost(indexes[0])

# # trajs_cost=defaultdict(tuple)

# start = time.time()

# with Pool() as pool_per_key:
# 	results = pool_per_key.starmap( get_kmerge_pairs_cost, zip(indexes) )

# end = time.time()
# print("Time: ", end-start, " s")


# In[75]:


# import pickle

# with open('dz_fault.pickle', 'rb') as handle:
# 	ft = pickle.load(handle)

# # print(ft[:1000])

# #510, 788525

# #473, 682547
# #450, 225314
# #450, 1005060
# #401, 492697
# #133, 981211

# num_trajs=len(ft)

# chosen_k=500

# num_cores=32 #100
# indexes= [] #[43066, 43067, 43068, 43069, 43070, 43071, 43072, 43073, 43074, 43075, 43076, 43077, 43078, 43079, 43080, 43081, 43082, 43083, 43084, 43085, 43086, 43087, 43088, 43089, 43090, 43091, 43092, 43093, 43094, 43095, 43096, 43097, 43098, 43099, 43100, 43101, 43102, 43103, 43104, 43105, 43106, 43107, 43108, 43109, 43110, 43111, 43112, 43113, 43114, 43115, 43116, 43117, 43118, 43119, 43120, 43121, 43122, 43123, 43124, 43125, 43126, 43127, 43128, 43129, 43130, 43131, 43132, 43133, 43134, 43135, 43136, 43137, 43138, 43139, 43140, 43141, 43142, 43143, 43144, 43145, 43146, 43147]#[]

# for i in range(num_cores):
# 	il = int(num_trajs/num_cores)
# 	indexes.append([j for j in range(i*il, i*il+il)])

# if indexes[-1][-1]!=num_trajs-1:
# 	indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])

# def get_kmerge_pairs_cost(indexes):
# 	global tdata_full, chosen_k, ft
# 	file = open("fault_million_costs/kmerge_costs_trajs" + str(indexes[0]) + ".txt", "w")

# 	for i in indexes:
# 		idx1, idx2 = ft[i][0], ft[i][1]
		
# 		traj1=np.zeros((len(tdata_full[idx1]), 9))
# 		traj2=np.zeros((len(tdata_full[idx2]), 9))

# 		idd=0
# 		for k in range(len(traj1)):
# 			traj1[k] = np.append(tdata_full[idx1][k], idd)

# 		for k in range(len(traj2)):
# 			traj2[k] = np.append(tdata_full[idx2][k], idd+1)

# # # 			tp.append( np.array([*traj1, *traj2]) )

# # 			tp.append( np.array([tdata_full[rows[i*chosen_k + j]], tdata_full[cols[i*chosen_k + j]]]) )

# 		print(traj1, traj2)

# 		this_results= get_node_pairs_m(np.array([*traj1, *traj2]))[0]

# # 			print(traj1, traj2)

# 		file.write( str( (idx1, idx2) ) + str(this_results)+" " ) #str(ft[i]) +
# 	file.close()
# 	x=1
# 	return x

# # # get_kmerge_pairs_cost(indexes[0])

# # trajs_cost=defaultdict(tuple)

# start = time.time()

# with Pool() as pool_per_key:
# 	results = pool_per_key.starmap( get_kmerge_pairs_cost, zip(indexes) )

# end = time.time()
# print("Time: ", end-start, " s")


# In[76]:


# with open('knn_matrix_senegal_partial_new' + str(1000) +  '.pickle', 'rb') as handle: #50000
# 	knn_m = pickle.load(handle)

# print("Making full K-merge matrix")

# num_trajs=len(tdata_full)

# rows, cols = knn_m.nonzero()
# chosen_k=500

# num_cores=32 #100
# indexes= [] #[43066, 43067, 43068, 43069, 43070, 43071, 43072, 43073, 43074, 43075, 43076, 43077, 43078, 43079, 43080, 43081, 43082, 43083, 43084, 43085, 43086, 43087, 43088, 43089, 43090, 43091, 43092, 43093, 43094, 43095, 43096, 43097, 43098, 43099, 43100, 43101, 43102, 43103, 43104, 43105, 43106, 43107, 43108, 43109, 43110, 43111, 43112, 43113, 43114, 43115, 43116, 43117, 43118, 43119, 43120, 43121, 43122, 43123, 43124, 43125, 43126, 43127, 43128, 43129, 43130, 43131, 43132, 43133, 43134, 43135, 43136, 43137, 43138, 43139, 43140, 43141, 43142, 43143, 43144, 43145, 43146, 43147]#[]

# for i in range(num_cores):
# 	il = int(num_trajs/num_cores)
# 	indexes.append([j for j in range(i*il, i*il+il)])

# if indexes[-1][-1]!=num_trajs-1:
# 	indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])

# def get_kmerge_pairs_cost(indexes):
# 	global tdata_full, chosen_k
# 	file = open("cost_size_test/size_1/kmerge_costs_trajs" + str(indexes[0]) + ".txt", "w")
# # 	this_results=defaultdict(tuple)

# 	for i in indexes:
# 		this_results=[]
# 		for j in range(chosen_k):
# # 			tid=int(tdata_full_civ[i][-1])
# 			idx1, idx2 = rows[i*chosen_k + j], cols[i*chosen_k + j]
			
# # 			traj1=np.zeros((len(tdata_full[idx1]), 9))
# # 			traj2=np.zeros((len(tdata_full[idx2]), 9))

# # 			idd=0
# # 			for k in range(len(traj1)):
# # 				traj1[k] = np.append(tdata_full[idx1][k], idd)

# # 			for k in range(len(traj2)):
# # 				traj2[k] = np.append(tdata_full[idx2][k], idd+1)
			
# 			tp=[]
# 			for samp in tdata_full[idx1]:
# 				tp.append([*samp,0])
			
# 			for samp in tdata_full[idx2]:
# 				tp.append([*samp,1])
			
# # 			tp.append( np.array([tdata_full[rows[i*chosen_k + j]], tdata_full[cols[i*chosen_k + j]]]) )
			
# # 			print(np.vstack(np.array(tp)))
		
# 			this_results.append( int(get_node_pairs_m(np.vstack(np.array(tp)))[0]) )#np.array([*traj1, *traj2])
# # 			print(traj1, traj2)
# 		file.write(str(this_results))

# # 		print(i, len(tp))
# # 	with open('final5000costciv/kmerge_costs_trajs_final' + str(indexes[0]) + '.pickle', 'wb') as handle:
# # 		pickle.dump(this_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
# 	file.close()
# 	x=1
# 	return x

# # get_kmerge_pairs_cost(indexes[0])

# # trajs_cost=defaultdict(tuple)

# start = time.time()

# with Pool() as pool_per_key:
# 	results = pool_per_key.starmap( get_kmerge_pairs_cost, zip(indexes) )

# end = time.time()
# print("Time: ", end-start, " s")


# ## KNN Comparison

# In[77]:


class compareKNNs:
	def __init__(self, size_dt):
		self.size_dt=size_dt
		self.chosen_k=500
	
	def check_percent_similarity(self, cp):
		with open('knn_matrix500.pickle', 'rb') as handle:
			knn_m = pickle.load(handle)

# 		print("Making full K-merge matrix")

		rows1, cols1 = knn_m.nonzero()
		
		with open('knn_matrix_senegal_partial' + str(sz*1000) +  '.pickle', 'rb') as handle:
			knn_m2 = pickle.load(handle)

# 		print("Making partial K-merge matrix")

		rows2, cols2 = knn_m2.nonzero()
		
		num_trajs = sz*1000
		
		with open('senegal_data_rand_'+str(num_trajs)+'.pickle', 'rb') as handle:
			rand_samples = pickle.load(handle)
		
		tr=list(rand_samples)
		
		count=0
		for j in range(num_trajs):
			neighbours1=set()
			neighbours2=set()
			for i in range(self.chosen_k):
				neighbours1.add(cols1[tr[j]*self.chosen_k+i])
				neighbours2.add(tr[cols2[j*self.chosen_k+i]])
# 			count=count+len(neighbours1.intersection(neighbours2))
			
# 			vals=neighbours2.difference(neighbours1)
			
			vals=neighbours1.intersection(neighbours2)
			count=count+len(vals)
			for v in vals:
				cp.add( (j, v) )
		
# 		print("Count is: ", count)
# 		print("The similarity found for size: ", num_trajs, "is: ", (count/(num_trajs*self.chosen_k))*100, "percent")


# In[78]:


# compute_pairs=set()

# for sz in sub_sampled_datasets:
# 	ck = compareKNNs(sz)
# 	ck.check_percent_similarity(compute_pairs)
# 	print(len(compute_pairs))
# 	break


# In[79]:


# with open('common_pairs_size_test.pickle', 'wb') as handle:
# 	pickle.dump(compute_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[80]:


# with open('compute_pairs_size_test.pickle', 'rb') as handle:
# 	compute_pairs = pickle.load(handle)


# In[81]:


# import itertools

# cp = sorted(compute_pairs, key=lambda x : x[0])

# for i, val in enumerate(itertools.islice(cp, 10)):
# 	print(i, val)


# In[82]:


# nbrs = defaultdict(int)

# for val in compute_pairs:
# 	nbrs[val[0]]=nbrs[val[0]]+1

# fig, ax = plt.subplots()

# sns.ecdfplot(data=list(nbrs.values()))


# In[83]:


# with open('knn_matrix_senegal_partial' + str(2000) +  '.pickle', 'rb') as handle:
# 	knn_m = pickle.load(handle)

# rows1, cols1 = knn_m.nonzero()

# with open('senegal_data_rand_'+str(2000)+'.pickle', 'rb') as handle:
# 	rand_samples = pickle.load(handle)

# tr=list(rand_samples)

# count=0
# for i in range(2000):
# 	for j in range(500):
# 		pair = (tr[i], tr[cols1[i*500+j]])
# 		if pair in compute_pairs:
# 			count=count+1

# print(count)


# ## Extract Costs

# In[84]:


import sys

from itertools import chain
import networkx as nx
import gc

class extractCosts:
	def __init__(self, K, cores, use_metric=False):
		self.use_metric=use_metric
		
		self.num_million_trajs=1008169 #1008109 
		
		self.knn_m = 0
		self.knn_m_senegal=0
		self.knn_m_civ=0
		self.knn_m_million=0
		self.knn_m_partial=0
		self.knn_m_week=0
		self.knn_m_metro=0
		self.knn_m_taka=0
		
		self.g_million=0
		
		self.kmerge_cost=defaultdict(tuple)
		self.kmerge_cost_shenzen=defaultdict(tuple)
		self.kmerge_cost_civ=defaultdict(tuple)
		self.kmerge_cost_million=defaultdict(tuple)
		self.kmerge_cost_partial=defaultdict(tuple)
		self.kmerge_cost_week=defaultdict(tuple)
		self.kmerge_cost_metro=defaultdict(tuple)
		self.kmerge_cost_taka=defaultdict(tuple)
		
		self.indexes=[]
		self.chosen_k=K
		self.num_cores=cores
		self.rows=0
		self.cols=0
		
		self.num_trajs_million=0
		self.num_trajs_shenzen=0
		self.num_trajs_civ=0
		self.num_trajs=0
		self.num_trajs_partial=0
		self.num_trajs_week=0
		self.num_trajs_metro=0
		self.num_trajs_taka=0
		
		self.million_final_costs=[]#defaultdict(tuple)
		self.shenzen_final_costs=defaultdict(tuple)
		self.senegal_final_costs=defaultdict(tuple)
		self.civ_final_costs=defaultdict(tuple)
		self.partial_final_costs=defaultdict(tuple)
		self.week_final_costs=defaultdict(tuple)
		self.metro_final_costs=defaultdict(tuple)
		self.taka_final_costs=defaultdict(tuple)
		
		self.million_final_costs_metric=[]#defaultdict(tuple)
		self.shenzen_final_costs_metric=defaultdict(tuple)
		self.senegal_final_costs_metric=defaultdict(tuple)
		self.civ_final_costs_metric=defaultdict(tuple)
		self.partial_final_costs_metric=defaultdict(tuple)
		self.week_final_costs_metric=defaultdict(tuple)
		self.metro_final_costs_metric=defaultdict(tuple)
		self.taka_final_costs_metric=defaultdict(tuple)
	
	def getNumMillionTrajs(self):
		return self.num_trajs_million
	
	def getNumShenzenTrajs(self):
		return self.num_trajs_shenzen
	
	def getNumTakaTrajs(self):
		return self.num_trajs_taka
	
	def getNumSenegalTrajs(self):
		return self.num_trajs
	
	def getNumCivTrajs(self):
		return self.num_trajs_civ
	
	def getNumPartialTrajs(self):
		return self.num_trajs_partial
	
	def getNumWeekTrajs(self):
		return self.num_trajs_week
	
	def getNumMetroTrajs(self):
		return self.num_trajs_metro
	
	def setK(self, K):
		self.chosen_k=K
	
	def getK(self):
		return self.chosen_k
	
	def getMergeCostMetro(self):
		return self.kmerge_cost_metro
	
	def getMergeCostMillion(self):
		return self.kmerge_cost_million
	
	def getMergeCostSenegal(self):
		return self.kmerge_cost
	
	def getMergeCostShenzen(self):
		return self.kmerge_cost_shenzen
	
	def getMergeCostTaka(self):
		return self.kmerge_cost_taka
	
	def getMergeCostCiv(self):
		return self.kmerge_cost_civ
	
	def getMergeCostWeek(self):
		return self.kmerge_cost_week
	
	def getFinalMergeCostMetro(self):
		return self.metro_final_costs
	
	def getFinalMergeCostMillion(self):
		return self.million_final_costs
	
	def getFinalMergeCostSenegal(self):
		return self.senegal_final_costs
	
	def getFinalMergeCostShenzen(self):
		return self.shenzen_final_costs
	
	def getFinalMergeCostTaka(self):
		return self.taka_final_costs
	
	def getFinalMergeCostCiv(self):
		return self.civ_final_costs
	
	def getFinalMergeCostPartial(self):
		return self.partial_final_costs
	
	def getFinalMergeCostWeek(self):
		return self.week_final_costs
	
	def getMetroKNN(self):
		return self.knn_m_metro
	
	def getMillionKNN(self):
		return self.knn_m_million
	
	def getShenzenKNN(self):
		return self.knn_m
	
	def getTakaKNN(self):
		return self.knn_m_taka
	
	def getSenegalKNN(self):
		return self.knn_m_senegal
	
	def getCivKNN(self):
		return self.knn_m_civ
	
	def getPartialKNN(self):
		return self.knn_m_partial
	
	def getWeekKNN(self):
		return self.knn_m_week
	
	def getMillionGraph(self):
		return self.g_million
	
	def PartialFinalCosts(self, sz):
		self.num_trajs_partial = sz*1000
		files = glob.glob("cost_size_test/size_" + str(sz) + "/*")
		
		for file in tqdm(files):
			this_index = int(file.split("/")[2].split(".")[0].split("_")[2][5:])
			start = [id1[0] for id1 in self.indexes]
			
			for i, st in enumerate(start):
	# 			print(i, st, this_index)
				if st==this_index:
					this_index=i
# 			print(this_index, start[this_index], file)

			with open(file) as f:
				for line in f:
					count1=0
					vals=line.split('[')
					for val in vals:
						if len(val)>0:
							cst = val.split(',')
							count2=0
							for cval in cst:
								key = (self.rows[self.indexes[this_index][count1]*self.chosen_k + count2], self.cols[self.indexes[this_index][count1]*self.chosen_k + count2])
								
								if cval[-1]!=']':
									self.partial_final_costs[key]=int(cval)
								else:
									self.partial_final_costs[key]=int(cval[:-1])
# 								print(key, self.partial_final_costs[key])
								count2=count2+1
							count1=count1+1
# 					print(million_final_costs)
				f.close()
# 					print(list(million_final_costs.keys())[0], list(million_final_costs.keys())[-1])
		print("There are: ", len(self.partial_final_costs.keys()), " merge cost pairs.")
	
	def MultSizeFinalCosts(self, sz):
# 		with open('common_pairs_size_test.pickle', 'rb') as handle:
# 			common_pairs = pickle.load(handle)
		
# 		common_costs = defaultdict(int)
# 		for pair in common_pairs:
# 			common_costs[pair]=self.senegal_final_costs[pair]
		
# 		with open('common_costs_size_test.pickle', 'wb') as handle:
# 			pickle.dump(common_costs, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
# 		del common_costs


		with open('all_common_costs_size_test.pickle', 'rb') as handle:
			all_costs = pickle.load(handle)
		
# 		with open('common_costs_size_test.pickle', 'rb') as handle:
# 			common_costs = pickle.load(handle)
		
# 		print(len(common_pairs.keys()))

# 		diff_cost = defaultdict(int)

# 		files = glob.glob("cost_size_test/*.txt")
		
# 		count=0
# 		for file in tqdm(files):
# 			with open(file) as f:
# 				for line in f:
# 					vals=line.split('(')
# 					for val in vals:
# 						if len(val)>0:
# 							key=(int(val.split(',')[0]), int(val.split(',')[1].split(')')[0]))
# 							diff_cost[key]=int(val.split(',')[1].split(')')[1])
# 							count=count+1
# # 							print(key, int(val.split(',')[1].split(')')[1]))

# 		with open('diff_common_costs_size_test.pickle', 'rb') as handle:
# 			diff_cost = pickle.load(handle)
		
# 		all_cost=defaultdict(int)
		
# 		for key in common_costs.keys():
# 			all_cost[key] = common_costs[key]
		
# 		for key in diff_cost.keys():
# 			all_cost[key] = diff_cost[key]
		
# 		with open('all_common_costs_size_test.pickle', 'wb') as handle:
# 			pickle.dump(all_cost, handle, protocol=pickle.HIGHEST_PROTOCOL)

		self.num_trajs_partial = sz*1000

		with open('senegal_data_rand_'+str(self.num_trajs_partial)+'.pickle', 'rb') as handle:
			rand_samples = pickle.load(handle)

		with open('knn_matrix_senegal_partial' + str(self.num_trajs_partial) +  '.pickle', 'rb') as handle:
			self.knn_m_partial = pickle.load(handle)

		rows1, cols1 = self.knn_m_partial.nonzero()
		
		tr=list(rand_samples)
		
# 		print(rows1, cols1)
		
		for i in range(len(rows1)):
			key=(tr[rows1[i]], tr[cols1[i]])
			self.partial_final_costs[key]=all_costs[key]
		
		del all_costs
	
	def MetroFinalCosts(self, fname):
		files = glob.glob(fname)
		print("There are: ", len(files), " files.")
		count=0
		
		part=1
		self.metro_final_costs=defaultdict(tuple)
		fcount=0
		for file in tqdm(files):
			this_index = int(file.split("/")[1].split(".")[0][4:])
			start = [id1[0] for id1 in self.indexes]
			for i, stp in enumerate(start):
				if stp==this_index:
					this_index=i
# 			print(this_index)
# 			print(start[this_index], file, this_index, self.chosen_k)

			with open(file) as f:
				for line in f:
					count1=0
					vals=line.split(' ')
					
					for val in vals:
# 						print(self.indexes[this_index][int(count1/self.chosen_k)])
						if len(val)>0: #self.indexes[this_index][int(count1/self.chosen_k)] < self.indexes[-1][-1]:
							key = (self.rows[self.indexes[this_index][int(count1/self.chosen_k)]*self.chosen_k + count1%self.chosen_k], self.cols[self.indexes[this_index][int(count1/self.chosen_k)]*self.chosen_k + count1%self.chosen_k])
							self.metro_final_costs[key]=val
							count1=count1+1
						else:
							break
# 					print(million_final_costs)
			f.close()
	
	def MillionFinalCosts(self, fname, cost_computed): 
		if self.use_metric:
# 			rcm_array = calculate_COM()

# 			with open('rg_array_senegal_million.pickle', 'rb') as handle:
# 				rg_array = pickle.load(handle)

# 			for i in tqdm(range(len(rcm_array))):
# 				rcm_array[i].append(rg_array[i])
			
# 			# Compute and store edge costs as a list of (i, j, cost)
# 			self.million_final_costs = [distance_metric10(rcm_array[i], rcm_array[j]) for i, j in zip(self.rows, self.cols)]
			
# 			with open(f'cost_list_million_metric.pickle', 'wb') as handle:
# 				pickle.dump(self.million_final_costs, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
			with open(f'cost_list_million_metric.pickle', 'rb') as handle:
				self.million_final_costs = pickle.load(handle)
			
			return 
		
		files = glob.glob(fname)
		print("There are: ", len(files), " files.")
		count=0
		
		if not cost_computed:
			part=1
			million_final_costs=defaultdict(tuple)
			fcount=0
			for file in tqdm(files):
				this_index = int(file.split("/")[1].split(".")[0].split("_")[3][7:])
				start = [id1[0] for id1 in self.indexes]
				for i, st in enumerate(start):
		# 			print(i, st, this_index)
					if st==this_index:
						this_index=i
# 				print(start[this_index], file)
				
				fcount=fcount+1
# 				keys=set()

				with open(file) as f:
					for line in f:
						count1=0
						vals=line.split('[')
						for val in vals:
							if len(val)>0:
								cst = val.split(',')
								count2=0
								for cval in cst:
									key = (self.rows[self.indexes[this_index][count1]*self.chosen_k + count2], self.cols[self.indexes[this_index][count1]*self.chosen_k + count2])
									if cval[-1]!=']':
										million_final_costs[key]=int(cval)
									else:
										million_final_costs[key]=int(cval[:-1])
# 									print(key, million_final_costs[key])
									count2=count2+1

# 									if fcount==150*part:
# 										dname="million_merge_costs"+str(part)
# 										self.million_final_costs.append(dname)
# 										with open(dname, 'wb') as handle:
# 											pickle.dump(million_final_costs, handle, protocol=pickle.HIGHEST_PROTOCOL)
# 										part=part+1

# 										del million_final_costs
# 										million_final_costs=defaultdict(tuple)
								count1=count1+1
# 					print(million_final_costs)
					f.close()
# 				print(len(keys))
				
# 					print(list(million_final_costs.keys())[0], list(million_final_costs.keys())[-1])
# 			print("There are: ", count1, " merge cost pairs.")

# 			dname="million_merge_costs"+str(part)
# 			with open(dname, 'wb') as handle:
# 				pickle.dump(million_final_costs, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 			i=0
# 			K=500
# 			edge_count=defaultdict(int)
# 			for val in kmerge_cost:
# 				edge_count[int(i/K)]=edge_count[int(i/K)]+1
# 	# 			print(int(i/K), cols[int(i/K)*K+i%K])
# 				if edge_count[int(i/K)]<=250 and val!=0:
# 					self.g_million.add_edge(int(i/K), cols[int(i/K)*K+i%K], weight=val)
# 				i=i+1
	
			self.million_final_costs=[]
			K=500
		
			for i in tqdm(range(self.num_million_trajs*K)):
				self.million_final_costs.append(sys.maxsize)
			i=0
			for key in million_final_costs.keys():
				self.million_final_costs[K*key[0]+i%K]=million_final_costs[key]#*len(tdata_full[key[0]])
# 				self.g_million.add_edge(key[0], key[1], weight=million_final_costs[key])
				i=i+1
			
			del million_final_costs, self.rows, self.cols
			gc.collect()
			
# 			self.g_million = nx.Graph()
# 			self.g_million.add_nodes_from([i for i in range(self.num_million_trajs)])
			
# 			rows, cols = self.knn_m_million.nonzero()
			
# 			print("Now creating graphs")

# 			i=0
# 			K=500
# 			edge_count=defaultdict(int)
# 			for val in self.million_final_costs:
# 				edge_count[int(i/K)]=edge_count[int(i/K)]+1
# 	# 			print(int(i/K), cols[int(i/K)*K+i%K])
# 				if edge_count[int(i/K)]<=200 and  val!=0: #edge_count[int(i/K)]<=250 and 
# 					self.g_million.add_edge(int(i/K), self.cols[int(i/K)*K+i%K], weight=val)
# 				i=i+1
			
# 			print("Graphs created")
# 			print(self.g_million)
		else:
# 			self.million_final_costs=defaultdict(tuple)
# # 			part=1

# 			for part in tqdm(range(1, 8)):
# 				neighbs=defaultdict(int)
# 				with open("million_merge_costs"+str(part), 'rb') as handle:
# 					this_cst = pickle.load(handle)
# 				for key in this_cst.keys():
# # 					self.million_final_costs[key]=this_cst[key]
# 					neighbs[key[0]]=neighbs[key[0]]+1
# 				for key in neighbs:
# 					if neighbs[key]!=500:
# 						print(key, neighbs[key])
# 				del this_cst
			
# 			print(len(self.million_final_costs.keys()))
			
# 			dz_fault=[]
# 			count_this=0
# 			for key in self.million_final_costs.keys():
# 				if isinstance(self.million_final_costs[key], tuple):
# 					count_this=count_this+1
# # 					dz_fault.append(key)
# 			print(count_this)
			
# # 			with open("dz_fault.pickle", 'wb') as handle:
# # 				pickle.dump(dz_fault, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
# # # 			part=2
# # # 			with open("million_merge_costs"+str(part), 'rb') as handle:
# # # 				this_cst = pickle.load(handle)
# # # 			for key in this_cst.keys():
# # # 				self.million_final_costs[key]=this_cst[key]
# # # 			del this_cst
			


			self.million_final_costs=[]
			K=500
		
			for i in range(self.num_million_trajs*K):
				self.million_final_costs.append(sys.maxsize)
			
			for part in tqdm(range(1, 8)):
				dname="million_merge_costs"+str(part)
				with open(dname, 'rb') as handle:
					this_cst = pickle.load(handle)

# 				print(len(list(this_cst.keys())))
				i=0
				for key in this_cst.keys():
					self.million_final_costs[K*key[0]+i%K]=this_cst[key]#*len(tdata_full[key[0]])
					i=i+1
				del this_cst


# # # 			for l, val in enumerate(self.million_final_costs):
# # # 				if val==0:
# # # 					self.million_final_costs[l]=sys.maxsize
# # # 			print(len(self.million_final_costs))



# 			with open('dz_fault.pickle', 'rb') as handle:
# 				ft = pickle.load(handle)

# 			num_trajs=len(ft)

# 			chosen_k=500

# 			num_cores=32 #100
# 			indexes= [] #[43066, 43067, 43068, 43069, 43070, 43071, 43072, 43073, 43074, 43075, 43076, 43077, 43078, 43079, 43080, 43081, 43082, 43083, 43084, 43085, 43086, 43087, 43088, 43089, 43090, 43091, 43092, 43093, 43094, 43095, 43096, 43097, 43098, 43099, 43100, 43101, 43102, 43103, 43104, 43105, 43106, 43107, 43108, 43109, 43110, 43111, 43112, 43113, 43114, 43115, 43116, 43117, 43118, 43119, 43120, 43121, 43122, 43123, 43124, 43125, 43126, 43127, 43128, 43129, 43130, 43131, 43132, 43133, 43134, 43135, 43136, 43137, 43138, 43139, 43140, 43141, 43142, 43143, 43144, 43145, 43146, 43147]#[]

# 			for i in range(num_cores):
# 				il = int(num_trajs/num_cores)
# 				indexes.append([j for j in range(i*il, i*il+il)])

# 			if indexes[-1][-1]!=num_trajs-1:
# 				indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])

# 			files = glob.glob("fault_million_costs/*")
# 			print("There are: ", len(files), " files.")
# 			count=0
			
# 			trjs=set()

# 			for file in tqdm(files):
# 				this_index = int(file.split("/")[1].split(".")[0].split("_")[2][5:])
# 				start = [id1[0] for id1 in self.indexes]
# 				for i, st in enumerate(start):
# 		# 			print(i, st, this_index)
# 					if st==this_index:
# 						this_index=i
# # 				print(this_index)
				
# 				with open(file) as f:
# 					for line in f:
# 						vals=line.split('.')
# 						j=0
# 						for val in vals:
# 							if len(val)>0:
# 								if this_index+j >= len(ft):
# 									break
# 								key=ft[this_index+j]
# # 								print(key, this_index)
# 		# 						print(key, float(val.split(',')[2][:-2]))
# 								v=val.split(')')
# 								if len(v)>1:
# 									ct = int(v[1])
# # 									print(ct)
# # 									if ct!=0:
# 									self.million_final_costs[K*key[0]+j%K]=ct#*len(tdata_full[key[0]])
# # 									else:
# # 										self.million_final_costs[K*key[0]+j%K]=sys.maxsize #ct
# # 										trjs.add(key[0])
# # 										trjs.add(key[1])
# # 										count=count+1
# 			# 						break
# 									j=j+1
# 			print("There are: ", count, "zero merge cost pairs.")
			
# 			print("There are: ", len(trjs), "singleton trajs.")


		
		# rows, cols = knn_m_million.nonzero()

# 		K=500

# 		list_costs=[]

# 		for j in tqdm(range(self.num_million_trajs)):
# 			index0=j*K
# 			for i in range(K):
# 				ct=self.million_final_costs[(j, self.cols[index0+i])]#*len(tdata_full[j])
# 				if not isinstance(ct, tuple):
# 					list_costs.append(ct)
		
# 		del self.million_final_costs
		
# 		self.million_final_costs=list_costs
		
# 			with open("million_cost_list.pickle", 'rb') as handle:
# 				self.million_final_costs=pickle.load(handle)
		
# 		ct_dict=defaultdict(int)
# 		for key in self.million_final_costs.keys():
# 			ct_dict[key[0]]=ct_dict[key[0]]+1
		
# 		print([(key, ct_dict[key]) for key in ct_dict.keys() if ct_dict[key]!=self.chosen_k], len(ct_dict.keys()))
		
# 		print(set([i for i in range(self.num_trajs_civ)])-set(list(ct_dict.keys())))
	
	def ShenzenFinalCosts(self, fname):
		files = glob.glob(fname)
		print("There are: ", len(files), " files.")
		count=0
		
		for f in tqdm(files):
			with open(f, 'rb') as handle:
				this_cost = pickle.load(handle)
			
			for key in this_cost.keys():
				self.shenzen_final_costs[key]=this_cost[key]
				count=count+1
		
		print("There are: ", count, " merge cost pairs.")
	
	def TakaFinalCosts(self, fname):
		files = glob.glob(fname)
		print("There are: ", len(files), " files.")
		count=0
		
		for f in tqdm(files):
			with open(f, 'rb') as handle:
				this_cost = pickle.load(handle)
			
			for key in this_cost.keys():
				self.taka_final_costs[key]=this_cost[key]
				count=count+1
		
		print("There are: ", count, " merge cost pairs.")
	
	def SenegalFinalCosts(self, fname):
		files = glob.glob(fname) #Original one! Keep it uncommented in the end.
		print("There are: ", len(files), " files.")
		count=0
		
		count2=0
		
		for f in tqdm(files):
			with open(f, 'rb') as handle:
				this_cost = pickle.load(handle)
			
			for key in this_cost.keys():
				self.senegal_final_costs[key]=this_cost[key]
				count=count+1
				
				if this_cost[key][0]==0 or this_cost[key][1]==0:
					count2=count2+1
		
		print("There are: ", count, " merge cost pairs.")
		
		print("There are: ", count2, "zero merge cost pairs.")

# 		files = glob.glob(fname)
# 		print("There are: ", len(files), " files.")
# 		count=0
		
# 		self.rows, self.cols = self.knn_m_senegal.nonzero()
		
# 		for f in tqdm(files):
# 			with open(f, 'rb') as handle:
# 				this_cost = pickle.load(handle)
			
# 			for key in this_cost.keys():
# 				self.senegal_final_costs[(key[0], self.cols[key[0]*self.chosen_k+key[1]])]=this_cost[key]
# 				count=count+1
		
# 		print("There are: ", count, " merge cost pairs.")
		
# 		with open('common_pairs_size_test.pickle', 'rb') as handle:
# 			common_pairs = pickle.load(handle)

# # 		print("Making full K-merge matrix")

# # 		self.rows, self.cols = self.knn_m_senegal.nonzero()

# 		del self.rows
# 		del self.cols
		
# 		common_costs = defaultdict(int)
# 		for pair in tqdm(common_pairs):
# 			common_costs[pair]=self.senegal_final_costs[pair][0]
		
# 		del self.senegal_final_costs
		
# 		with open('common_costs_size_test.pickle', 'wb') as handle:
# 			pickle.dump(common_costs, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 		with open('diff_common_costs_size_test.pickle', 'rb') as handle:
# 			diff_cost = pickle.load(handle)
		
# 		all_cost=defaultdict(int)
		
# 		for key in common_costs.keys():
# 			all_cost[key] = common_costs[key]
		
# 		del common_costs
		
# 		for key in diff_cost.keys():
# 			all_cost[key] = diff_cost[key]

# 		del diff_cost
		
# 		self.num_trajs_partial = sz*1000

# 		with open('senegal_data_rand_'+str(self.num_trajs_partial)+'.pickle', 'rb') as handle:
# 			rand_samples = pickle.load(handle)

# 		with open('knn_matrix_senegal_partial' + str(self.num_trajs_partial) +  '.pickle', 'rb') as handle:
# 			self.knn_m_partial = pickle.load(handle)

# 		rows1, cols1 = self.knn_m_partial.nonzero()
		
# 		tr=list(rand_samples)
		
# 		print(rows1, cols1)
		
# 		for i in tqdm(range(len(rows1))):
# 			key=(tr[rows1[i]], tr[cols1[i]])
# 			self.partial_final_costs[key]=all_costs[key]
		
# 		del all_cost
# # 		del common_costs
	
	def SenegalFinalCostsPartial(self, fname, idx):
		global sub_sampled_datasets
		
		files = glob.glob(fname)
		print("There are: ", len(files), " files.")
		count=0
		
		num=sub_sampled_datasets[idx]*1000
		
		with open('senegal_data_rand_'+str(num)+'.pickle', 'rb') as handle:
			rand_samples = pickle.load(handle)
		
		selected_trajs = defaultdict(bool)
		l_senegal=319366
		
		for j in range(l_senegal):
			selected_trajs[j]=False
		
		for rs in rand_samples:
			selected_trajs[rs]=True
		
		for f in tqdm(files):
			with open(f, 'rb') as handle:
				this_cost = pickle.load(handle)
			
			for key in this_cost.keys():
				if selected_trajs[key[0]]:
					self.senegal_final_costs[key]=this_cost[key]
					count=count+1
			
			del this_cost
		
		print("There are: ", count, " merge cost pairs.")
	
	def CivFinalCosts(self, fname):
		files = glob.glob(fname)
		print("There are: ", len(files), " files.")
		count=0
		
		for file in tqdm(files):
			with open(file) as f:
				for line in f:
					vals=line.split('(')
					for val in vals:
						if len(val)>0:
							key=(int(val.split(',')[0]), int(val.split(',')[1]))
	# 						print(key, float(val.split(',')[2][:-2]))
							if key[0]!=43066:
								self.civ_final_costs[key]=float(val.split(',')[2][:-2])
								count=count+1
	# 						break
		
		print("There are: ", count, " merge cost pairs.")
		
		ct_dict=defaultdict(int)
		for key in self.civ_final_costs.keys():
			ct_dict[key[0]]=ct_dict[key[0]]+1
		
		print([(key, ct_dict[key]) for key in ct_dict.keys() if ct_dict[key]!=self.chosen_k], len(ct_dict.keys()))
		
# 		print(set([i for i in range(self.num_trajs_civ)])-set(list(ct_dict.keys())))

	
	def WeekFinalCosts(self, fname):
		files = glob.glob(fname)

		print(files, [id1[0] for id1 in self.indexes])
		
		count=0
		
		for j in tqdm(range(len(files))):
			with open(files[j]) as f:
				this_index = int(files[j].split("/")[1].split(".")[0].split("_")[2][5:])
				start = [id1[0] for id1 in self.indexes]
				for i, st in enumerate(start):
		# 			print(i, st, this_index)
					if st==this_index:
						this_index=i
				print(this_index)
				
				i=0
				for line in f:
					vals=line.split(' ')
					
					for val in tqdm(vals):
						if i>=len(self.indexes[this_index])*self.chosen_k:
							break
						key=(self.rows[self.indexes[this_index][int(i/self.chosen_k)]*self.chosen_k + i%self.chosen_k], self.cols[self.indexes[this_index][int(i/self.chosen_k)]*self.chosen_k + i%self.chosen_k])
						count=count+1
						self.week_final_costs[key] = int(val)
						
						i=i+1
		
		print("There are: ", count, " merge cost pairs.")
	
	def testKNN(self, fname):
		with open(fname, 'rb') as handle:
			self.knn_m_senegal = pickle.load(handle)

		print("Making full K-merge matrix")

		self.rows, self.cols = self.knn_m_senegal.nonzero()
		self.indexes=[]

		self.num_trajs = int(len(self.rows)/self.chosen_k)
		print("Total num of trajs. are: ", self.num_trajs)

		for i in range(self.num_cores):
			il = int(self.num_trajs/self.num_cores)
			self.indexes.append([j for j in range(i*il, i*il+il)])
	
	def shenzenKNN(self, fname):
		with open(fname, 'rb') as handle:
			self.knn_m = pickle.load(handle)

		print("Making full K-merge matrix")

		self.rows, self.cols = self.knn_m.nonzero()
		
		self.indexes=[]

		self.num_trajs_shenzen = int(len(self.rows)/self.chosen_k)
		print("Total num of trajs. are: ", self.num_trajs_shenzen)

		for i in range(self.num_cores):
			il = int(self.num_trajs_shenzen/self.num_cores)
			self.indexes.append([j for j in range(i*il, i*il+il)])
	
	def takaKNN(self, fname):
		with open(fname, 'rb') as handle:
			self.knn_m_taka = pickle.load(handle)

		print("Making full K-merge matrix")

		self.rows, self.cols = self.knn_m_taka.nonzero()
		
		self.indexes=[]

		self.num_trajs_taka = int(len(self.rows)/self.chosen_k)
		print("Total num of trajs. are: ", self.num_trajs_taka)

		for i in range(self.num_cores):
			il = int(self.num_trajs_taka/self.num_cores)
			self.indexes.append([j for j in range(i*il, i*il+il)])
	
	def weekKNN(self, fname):
		with open(fname, 'rb') as handle:
			self.knn_m_week = pickle.load(handle)

		print("Making full K-merge matrix")

		self.rows, self.cols = self.knn_m_week.nonzero()
		
		self.indexes=[]

		self.num_trajs_week = int(len(self.rows)/self.chosen_k)
		print("Total num of trajs. are: ", self.num_trajs_week)

		for i in range(self.num_cores):
			il = int(self.num_trajs_week/self.num_cores)
			self.indexes.append([j for j in range(i*il, i*il+il)])
		
		if self.indexes[-1][-1]!=self.num_trajs_week-1:
			self.indexes.append([j for j in range(self.indexes[-1][-1]+1, self.num_trajs_week)])
	
	def partialKNN(self, fname):
		with open(fname, 'rb') as handle:
			self.knn_m_partial = pickle.load(handle)

		print("Making full K-merge matrix")

		self.rows, self.cols = self.knn_m_partial.nonzero()
		
		self.indexes=[]

		self.num_trajs_partial = int(len(self.rows)/self.chosen_k)
		print("Total num of trajs. are: ", self.num_trajs_partial)

		for i in range(self.num_cores):
			il = int(self.num_trajs_partial/self.num_cores)
			self.indexes.append([j for j in range(i*il, i*il+il)])
		
		if self.indexes[-1][-1]!=self.num_trajs_partial-1:
			self.indexes.append([j for j in range(self.indexes[-1][-1]+1, self.num_trajs_partial)])
	
	def civKNN(self, fname):
		with open(fname, 'rb') as handle:
			self.knn_m_civ = pickle.load(handle)

		print("Making full K-merge matrix")

		self.rows, self.cols = self.knn_m_civ.nonzero()
		
		self.indexes=[]

		self.num_trajs_civ = int(len(self.rows)/self.chosen_k)
		print("Total num of trajs. are: ", self.num_trajs_civ)

		for i in range(self.num_cores):
			il = int(self.num_trajs_civ/self.num_cores)
			self.indexes.append([j for j in range(i*il, i*il+il)])
		
		if self.indexes[-1][-1]!=self.num_trajs_civ-1:
			self.indexes.append([j for j in range(self.indexes[-1][-1]+1, self.num_trajs_civ)])
	
	def metroKNN(self, fname):
		with open(fname, 'rb') as handle:
			self.knn_m_metro = pickle.load(handle)

		print("Making full K-merge matrix")

		self.rows, self.cols = self.knn_m_metro.nonzero()
		
		self.indexes=[]

		self.num_trajs_metro = int(len(self.rows)/self.chosen_k)
		print("Total num of trajs. are: ", self.num_trajs_metro)

		for i in range(self.num_cores):
			il = int(self.num_trajs_metro/self.num_cores)
			self.indexes.append([j for j in range(i*il, i*il+il)])
		
		if self.indexes[-1][-1]!=self.num_trajs_metro-1:
			self.indexes.append([j for j in range(self.indexes[-1][-1]+1, self.num_trajs_metro)])
	
	def millionKNN(self, fname):
		with open(fname, 'rb') as handle:
			self.knn_m_million = pickle.load(handle)

		print("Making full K-merge matrix")

		self.rows, self.cols = self.knn_m_million.nonzero()
		
		self.indexes=[]

		self.num_trajs_million = int(len(self.rows)/self.chosen_k)
		print("Total num of trajs. are: ", self.num_trajs_million)

		for i in range(self.num_cores):
			il = int(self.num_trajs_million/self.num_cores)
			self.indexes.append([j for j in range(i*il, i*il+il)])
		
		if self.indexes[-1][-1]!=self.num_trajs_civ-1:
			self.indexes.append([j for j in range(self.indexes[-1][-1]+1, self.num_trajs_million)])
	
	def totalKmergeCosts(self, fname):
		files = glob.glob(fname)

		print(files, [id1[0] for id1 in self.indexes])
		
		for j in tqdm(range(len(files))):
			with open(files[j]) as f:
				this_index = int(files[j].split("/")[1].split(".")[0].split("_")[2][1:])
				start = [id1[0] for id1 in self.indexes]
				for i, st in enumerate(start):
		# 			print(i, st, this_index)
					if st==this_index:
						this_index=i
				print(this_index)
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
						self.kmerge_cost[(self.rows[self.indexes[this_index][int(i/self.chosen_k)]*self.chosen_k + i%self.chosen_k], self.cols[self.indexes[this_index][int(i/self.chosen_k)]*self.chosen_k + i%self.chosen_k])] = (float(val[0]), float(val[1]))
						i=i+1
						#print((rows[indexes[j][int(i/chosen_k)]*chosen_k + i%chosen_k], cols[indexes[j][int(i/chosen_k)]*chosen_k + i%chosen_k]))
	
	def shenzenMergeCosts(self, fname):
		files = glob.glob(fname)

		print(files, [id1[0] for id1 in self.indexes])
		
		for j in tqdm(range(len(files))):
			with open(files[j]) as f:
				this_index = int(files[j].split("/")[1].split(".")[0].split("_")[3][1:])
				start = [id1[0] for id1 in self.indexes]
				for i, st in enumerate(start):
		# 			print(i, st, this_index)
					if st==this_index:
						this_index=i
				print(this_index)
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
						self.kmerge_cost_shenzen[(self.rows[self.indexes[this_index][int(i/self.chosen_k)]*self.chosen_k + i%self.chosen_k], self.cols[self.indexes[this_index][int(i/self.chosen_k)]*self.chosen_k + i%self.chosen_k])] = (float(val[0]), float(val[1]))
						i=i+1
						#print((rows[indexes[j][int(i/chosen_k)]*chosen_k + i%chosen_k], cols[indexes[j][int(i/chosen_k)]*chosen_k + i%chosen_k]))


# In[85]:


# ec = extractCosts(100, 32)
# ec.testKNN('knn_matrix.pickle')
# ec.totalKmergeCosts('cost_kmerge/kmerge_cost_m*')
# ec.shenzenKNN('knn_matrix_shenzen100.pickle')
# ec.shenzenMergeCosts('cost_kmerge_shenzen/kmerge_cost_shenzen_m*')


# In[86]:


# ec_final = extractCosts(500, 1000)
# ec_final.shenzenKNN('knn_matrix_shenzen_new.pickle') #500
# ec_final.ShenzenFinalCosts("final5000costshenzennew/*")
# # ec_final.getSenegalFinalCosts("final5000cost/*")


# In[87]:


# ec_final = extractCosts(500, 1000)
# ec_final.takaKNN('knn_matrix_taka_new.pickle') #500
# ec_final.TakaFinalCosts("final5000costtakanew/*")
# # ec_final.getSenegalFinalCosts("final5000cost/*")


# In[88]:


# ec_final2 = extractCosts(500, 1000)
# ec_final2.testKNN('knn_matrix500.pickle')
# ec_final2.SenegalFinalCosts("final5000cost/*")


# In[89]:


# ec_final3 = extractCosts(500, 32)
# ec_final3.civKNN('knn_matrix_civ_new.pickle') #500
# ec_final3.CivFinalCosts("final5000costcivnew/*")


# In[90]:


# ec_final4 = extractCosts(500, 1000)
# ec_final4.millionKNN('knn_matrix_senegal_million500.pickle')
# ec_final4.MillionFinalCosts("cost_kmerge_senegal_million/*", True)


# In[91]:


# ec_final9 = extractCosts(500, 1000)
# ec_final9.millionKNN('knn_matrix_senegal_million_new.pickle')
# ec_final9.MillionFinalCosts("cost_kmerge_senegal_million_new/*", False)


# In[92]:


# st=2

# ec_final10 = extractCosts(100, 32)
# ec_final10.metroKNN('knn_matrix_metro_unique' + str(st) +  '.pickle')
# ec_final10.MetroFinalCosts("merge_cost_metro/*")


# In[2]:


# ec_final9 = extractCosts(500, 1000, True)
# ec_final9.millionKNN('knn_matrix_senegal_million_new.pickle')
# ec_final9.MillionFinalCosts("cost_kmerge_senegal_million_new/*", False)


# In[94]:


# with open("million_cost_list.pickle", 'wb') as handle:
# 	pickle.dump(ec_final4.getFinalMergeCostMillion(), handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[95]:


# t = ec_final4.getFinalMergeCostMillion()

# num_trajs=1008109
# nt=0
# K=500

# xt=[]

# for i in tqdm(range(num_trajs)):
# 	for j in range(K):
# 		if t[i*K+j]==0: #isinstance(t[i*K+j], tuple) or 
# 			nt=nt+1
# 			xt.append( (i, i*K+j) )

# print(nt)


# In[96]:


# # print(len(set([val[0] for val in xt])))

# with open('dz_fault.pickle', 'wb') as handle:
# 	pickle.dump(xt, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[97]:


# with open('dz_fault.pickle', 'rb') as handle:
# 	ft = pickle.load(handle)

# print(len(ft))


# In[98]:


# ston=set()
# for val in tqdm(ft):
# 	if val[0]!=val[1]:
# 		ston.add(val[0])

# print(len(ston))


# In[99]:


# with open('knn_matrix_senegal_million500.pickle', 'rb') as handle:
# 	knn_m = pickle.load(handle)

# rows, cols = knn_m.nonzero()

# xt=[]

# for i in tqdm(range(len(ft))):
# 	xt.append( (ft[i][0], cols[ft[i][1]]) )

# print(len(xt))


# In[100]:


# # print(xt[:100])

# with open('dz_fault.pickle', 'wb') as handle:
# 	pickle.dump(xt, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[101]:


# with open('dz_fault.pickle', 'wb') as handle:
# 	pickle.dump(xt, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[102]:


# with open("dz_fault.pickle", 'rb') as handle:
# 	dt=pickle.load(handle)


# In[103]:


# with open('fault_tajs_million.pickle', 'rb') as handle:
# 	ft = pickle.load(handle)

# for dval in tqdm(dt):
# 	if dval[0]!=dval[1] and dval not in ft:
# 		print(dval)


# In[104]:


# ec_final5 = extractCosts(500, 1000)
# ec_final5.testKNN('knn_matrix500.pickle')
# ec_final5.SenegalFinalCostsPartial("final5000cost/*", 0)


# In[105]:


# ec_final6 = extractCosts(500, 1000)
# ec_final6.MultSizeFinalCosts(200)


# In[106]:


# ec_final7 = extractCosts(500, 32)
# ec_final7.weekKNN('knn_matrix_week21.pickle')
# ec_final7.WeekFinalCosts("cost_week21/*")


# In[107]:


# sz=50 #1

# ec_final8 = extractCosts(500, 32)
# ec_final8.partialKNN('knn_matrix_senegal_partial_new' + str(sz*1000) +  '.pickle')
# ec_final8.PartialFinalCosts(sz)


# In[108]:


# def compareMetroMergeCostsToKNNDistance():
# 	ct = ec_final10.getFinalMergeCostMetro()
# 	val_array = shape_metro_trajs(tdata_full_metro_unique)
	
# # 	print(list(ct.keys()))

# 	stt=2
# 	dist1=[]
# 	dist2=[]
	
# 	for key in ct.keys():
# 		dist2.append( distance_metric_metro2(val_array[key[0]], val_array[key[1]]) )
		
# 		dist1.append( int(ct[key]) )
	
# 	plt.figure(figsize=(10, 10))
# 	axi = plt.gca()
# 	#fig, axi = plt.subplots(1, 1)

# 	# print(len(costs))

# 	plt.scatter(dist2, dist1, alpha=0.01) #alpha=0.01

# 	plt.ylabel("Merge Cost")
# 	plt.xlabel("Distance metric")

# # 	plt.ylim([0, 1e9])
# # 	# # plt.ylim([0, 0.15e12])
# # 	plt.xlim([0, 25])

# 	# plt.xlim([1, 60000])

# 	plt.savefig("plot_dist_metro"+str(stt)+".pdf")

# compareMetroMergeCostsToKNNDistance()


# In[109]:


# sp_accuracy=[]
# t_accuracy=[]

# with open('knn_matrix_metro_unique' + str(st) +  '.pickle', 'rb') as handle: #unique
# 	knn_m = pickle.load(handle)

# print("Making full K-merge matrix")

# rows, cols = knn_m.nonzero()

# def get_min_cost_metro(idx):
# 	ct = ec_final10.getFinalMergeCostMetro()
	
# 	vals = []
	
# 	for key in ct.keys():
# 		if key[0]==idx:
# 			vals.append( (key[1], ct[key]) )
	
# 	vals = sorted(vals, key=lambda x:x[1])
	
# 	for val in vals:
# 		if val[0]!=idx:
# 			return val[0]

# def get_kmerge_closest_pairs(indx):
# 	global tdata_full_metro, tdata_full_metro_kmerge, tdata_full_metro_unique, tdata_full_metro_unique_kmerge
	
# 	sp_accuracy_here=[]
# 	t_accuracy_here=[]
	
# 	chosen_k=100
# 	val=1

# 	file2 = open("generalised_metro/traj_closest" + str(indx[0]) + ".txt", "w")

# 	trace=""
	
# 	for i in indx:
# 		j = get_min_cost_metro(i)
# 		if i*chosen_k + j < len(cols):
# 			if i!=cols[i*chosen_k + j]:
# 				traj1=np.zeros((len(tdata_full_metro_unique_kmerge[rows[i*chosen_k + j]]), 9))
# 				traj2=np.zeros((len(tdata_full_metro_unique_kmerge[cols[i*chosen_k + j]]), 9))

# 				idd=0
# 				for k in range(len(traj1)):
# 					traj1[k] = np.append(tdata_full_metro_unique_kmerge[rows[i*chosen_k + j]][k], idd)

# 				for k in range(len(traj2)):
# 					traj2[k] = np.append(tdata_full_metro_unique_kmerge[cols[i*chosen_k + j]][k], idd+1)

# 	# 				tp.append( np.array([*traj1, *traj2]) )

# 				#tp.append( np.array([tdata_full[rows[i*chosen_k + j]], tdata_full[cols[i*chosen_k + j]]]) )
# 				result = get_node_trajs_metro_m(np.array([*traj1, *traj2]), 2)
# 				trace=trace + str(int(result[1][0])) + " "
# 	# 			trace=trace + str(result[0]) + '\n'

# 				for gsamp in result[0]:
# 					file2.write( str(gsamp[0]) + " " + str(gsamp[1]) + " " + str(gsamp[2]) + '\n' )
# 					sp_accuracy_here.append(len(gsamp[0]))
# 					t_accuracy_here.append(abs(gsamp[2] - gsamp[1]))
# 				file2.write('\n')

# 	file2.close()
# 	return (sp_accuracy_here, t_accuracy_here)

# # 	for idx in indx:
# # 		tp=[]
# # 		trajs_here=idx #i,j
# # 		print(tdata_full_metro_kmerge[trajs_here[0]], tdata_full_metro_kmerge[trajs_here[1]])
		
# # 		idd=0
# # 		for j in range(len(trajs_here)):
# # 			traj1=np.zeros((len(tdata_full_metro_kmerge[trajs_here[j]]), 9))

# # 			for k in range(len(traj1)):
# # 				traj1[k] = np.append(tdata_full_metro_kmerge[trajs_here[j]][k], idd)

# # 			tp.extend( [*traj1] )

# # 			idd = idd + 1
# # 		tp = np.array(tp)
# # # 					print(i, tp.shape[0], tp.shape[1], len(trajs), trajs)

# # # 				this_results = get_node_trajs_m(tp, len(trajs))

# # 		result = get_node_trajs_metro_m(tp, len(trajs_here)) #.append( get_node_trajs_m(tp, len(trajs_here)) ) #generalised_traj, cost = kmerge(trajs[[i,j]])
		
# # # 		print(result[0])

# indexes=[]
# num_cores=32
# num_trajs=len(tdata_full_metro_unique_kmerge)

# for i in range(num_cores):
# 	il = int(num_trajs/num_cores)
# 	indexes.append([j for j in range(i*il, i*il+il)])

# if indexes[-1][-1]!=num_trajs-1:
# 	indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])

# with Pool() as pool_per_key:
# 	results = pool_per_key.starmap( get_kmerge_closest_pairs, zip(indexes) )

# for res in results:
# 	sp_accuracy.extend(res[0])
# 	t_accuracy.extend(res[1])

# # get_kmerge_pairs_cost_metro(indexes[1])

# with open('spatial_accuracy_metro_unique.pickle', 'wb') as handle:
# 	pickle.dump(sp_accuracy, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('temporal_accuracy_metro_unique.pickle', 'wb') as handle:
# 	pickle.dump(t_accuracy, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[110]:


# ## fig, ax = plt.subplots()

# sns.ecdfplot(sp_accuracy) #Check the index consistency and condition while generalization above if needed!

# # plt.ylim([0, 2500])
# # plt.xlim([0, 2100])
# plt.xlabel("Num. of locations in generalised sample")
# plt.ylabel("CDF")

# # plt.savefig("knn-sizes_shenzen.pdf")
# plt.savefig("spatial_accuracy_metro.pdf")

# fig, ax = plt.subplots()

# sns.ecdfplot(t_accuracy)

# # plt.ylim([0, 2500])
# # plt.xlim([0, 2100])
# plt.xlabel("Time (units)")
# plt.ylabel("CDF")

# # plt.savefig("knn-sizes_shenzen.pdf")
# plt.savefig("temporal_accuracy_metro.pdf")


# In[111]:


# sp_accuracy_all=defaultdict(list)
# t_accuracy_all=defaultdict(list)

# def get_min_cost_metro_all(idx, this_K):
# 	ct = ec_final10.getFinalMergeCostMetro()
	
# 	vals = []
	
# 	for key in ct.keys():
# 		if key[0]==idx:
# 			vals.append( (key[1], ct[key]) )
	
# 	vals = sorted(vals, key=lambda x:x[1])
# 	close_neighbours=[]
	
# 	for val in vals:
# 		if val[0]!=idx and len(close_neighbours)<this_K-1:
# 			close_neighbours.append(val[0])
	
# 	return close_neighbours

# def get_kmerge_closest_pairs_all(indx):
# 	global tdata_full_metro, tdata_full_metro_kmerge, utrajs, all_to_one_map
	
# 	selected_K=30
	
# 	sp_accuracy_here=defaultdict(list)
# 	t_accuracy_here=defaultdict(list)
	
# 	chosen_k=100
# 	val=1

# # 	file2 = open("generalised_metro_all/traj_closest" + str(indx[0]) + ".txt", "w")

# # 	trace=""
	
# 	for i in indx:
# 		js = get_min_cost_metro_all(i, selected_K)

# 		js = js + [i]

# 		tp=[]
# 		idd=0
# 		for j in range(len(js)):
# # 					print(len(tdata_full_shenzen[trajs[j]]))
# 			traj1=np.zeros((len(tdata_full_metro_unique_kmerge[js[j]]), 9))

# 			for k in range(len(traj1)):
# 				traj1[k] = np.append(tdata_full_metro_unique_kmerge[js[j]][k], idd)

# 			tp.extend( [*traj1] )

# 			idd = idd + 1
# 		tp = np.array(tp)
# 		result = get_node_trajs_metro_m(tp, len(js))

# # 		trace=trace + str(int(result[1][0])) + " "

# 		for gsamp in result[0]:
# # 			file2.write( str(gsamp[0]) + " " + str(gsamp[1]) + " " + str(gsamp[2]) + '\n' )
# 			sp_accuracy_here[i].append(len(gsamp[0]))
# 			t_accuracy_here[i].append(abs(gsamp[2] - gsamp[1]))
# # 		file2.write('\n')
# # 		else:
# # 			for gsamp in tdata_full_metro[i]:
# # 				file2.write( "[" + str(gsamp[0]) + "]" + " " + str(gsamp[1]) + " " + str(gsamp[1]) + '\n' )
# # 				sp_accuracy_here.append(1)
# # 				t_accuracy_here.append(0)
# # 			file2.write('\n')

# # 	file2.close()
# 	return (sp_accuracy_here, t_accuracy_here)

# # 	for idx in indx:
# # 		tp=[]
# # 		trajs_here=idx #i,j
# # 		print(tdata_full_metro_kmerge[trajs_here[0]], tdata_full_metro_kmerge[trajs_here[1]])
		
# # 		idd=0
# # 		for j in range(len(trajs_here)):
# # 			traj1=np.zeros((len(tdata_full_metro_kmerge[trajs_here[j]]), 9))

# # 			for k in range(len(traj1)):
# # 				traj1[k] = np.append(tdata_full_metro_kmerge[trajs_here[j]][k], idd)

# # 			tp.extend( [*traj1] )

# # 			idd = idd + 1
# # 		tp = np.array(tp)
# # # 					print(i, tp.shape[0], tp.shape[1], len(trajs), trajs)

# # # 				this_results = get_node_trajs_m(tp, len(trajs))

# # 		result = get_node_trajs_metro_m(tp, len(trajs_here)) #.append( get_node_trajs_m(tp, len(trajs_here)) ) #generalised_traj, cost = kmerge(trajs[[i,j]])
		
# # # 		print(result[0])

# indexes=[]
# num_cores=32
# num_trajs=len(tdata_full_metro_unique_kmerge)

# for i in range(num_cores):
# 	il = int(num_trajs/num_cores)
# 	indexes.append([j for j in range(i*il, i*il+il)])

# if indexes[-1][-1]!=num_trajs-1:
# 	indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])

# with Pool() as pool_per_key:
# 	results = pool_per_key.starmap( get_kmerge_closest_pairs_all, zip(indexes) )

# for res in results:
# 	for key in res[0].keys():
# 		sp_accuracy_all[key] = res[0][key]
	
# 	for key in res[1].keys():
# 		t_accuracy_all[key] = res[1][key]

# selected_K=30
# val=defaultdict(int)

# alsp=[]
# alt=[]

# for i in range(len(tdata_full_metro)):
# 	if val[str(tdata_full_metro[i])]==0 and utrajs[str(tdata_full_metro[i])]<selected_K:
# 		alsp.extend(sp_accuracy_all[all_to_one_map[i]])
# 		alt.extend(t_accuracy_all[all_to_one_map[i]])
		
# 		val[str(tdata_full_metro[i])]=1
# 	else:
# 		for gsamp in tdata_full_metro[i]:
# 			alsp.append(1)
# 			alt.append(0)

# # get_kmerge_pairs_cost_metro(indexes[1])

# with open('spatial_accuracy_metro.pickle', 'wb') as handle:
# 	pickle.dump(alsp, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('temporal_accuracy_metro.pickle', 'wb') as handle:
# 	pickle.dump(alt, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[112]:


# ## fig, ax = plt.subplots()

# sns.ecdfplot(alsp, label="MilAnon (All trajs.)") #Check the index consistency and condition while generalization above if needed!
# sns.ecdfplot(sp_accuracy, label="Milanon (Unique trajs. only)")

# # plt.ylim([0, 2500])
# # plt.xlim([0, 2100])
# plt.xlabel("Num. of locations in generalised sample")
# plt.ylabel("CDF")

# # plt.savefig("knn-sizes_shenzen.pdf")
# plt.legend()
# plt.savefig("spatial_accuracy_metro_all.pdf")

# fig, ax = plt.subplots()

# sns.ecdfplot(alt, label="MilAnon (All trajs.)")
# sns.ecdfplot(t_accuracy, label="MilAnon (Unique trajs. only)")

# # plt.ylim([0, 2500])
# # plt.xlim([0, 2100])
# plt.xlabel("Time (units)")
# plt.ylabel("CDF")

# # plt.savefig("knn-sizes_shenzen.pdf")
# plt.legend()
# plt.savefig("temporal_accuracy_metro_all.pdf")


# In[113]:


def SenegalPlotCosts(cst):
	dt_cost=defaultdict(list)
	for key in cst.keys():
		dt_cost[key[0]].append(cst[key])
	
	arr_cost=[]
	for key in dt_cost.keys():
		arr_cost.append(np.mean(dt_cost[key]))
	
	fig, ax = plt.subplots()

	sns.ecdfplot(arr_cost)

	plt.savefig("sg_cost.pdf")


# In[114]:


def MillionPlotCosts(cst1, cst2):
	dt_cost=defaultdict(list)
	K=500
	
	thresh=[5000, 50000, 100000, 315000, 500000, 1000000]
	
	for j in range(1008109):
		if j>315000:
			break
		traj=j
		index0=traj*K
		for i in range(K):
			dt_cost[traj].append(cst1[traj*K+i])
	
	arr_cost=[]
	for key in dt_cost.keys():
		arr_cost.append(np.mean(dt_cost[key]))
	
	fig, ax = plt.subplots()

	sns.ecdfplot(arr_cost, label="Senegal")
	
	dt_cost=defaultdict(list)
	for key in cst2.keys():
		dt_cost[key[0]].append(cst2[key])
	
	arr_cost=[]
	for key in dt_cost.keys():
		arr_cost.append(np.mean(dt_cost[key]))

	sns.ecdfplot(arr_cost, label="Million")
	
	plt.xlim([0, 0.5e10])
	plt.xlabel("Average neighbour merge cost")
	
	plt.legend()

	plt.savefig("mill_cost.pdf")


# In[115]:


# ct = ec_final9.getFinalMergeCostMillion()


# In[116]:


# count = defaultdict(int)

# for key in tqdm(ct.keys()):
# 	if ct[key]==0:
# 		count[key[0]]=count[key[0]]+1


# In[117]:


# def getDist(p1, p2):
# 	return np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )

# cst=ec_final4.getFinalMergeCostMillion()
# knn_m_million=ec_final4.getMillionKNN()

# week=0
# with open('com_values_senegal_million.pickle', 'rb') as handle:
# 	rcm_array = pickle.load(handle)

# rows, cols = knn_m_million.nonzero()

# K=500
# num_million_trajs=int(len(cols)/K)
# print("The number of trajs. is: ", num_million_trajs)

# ptrajs=[]
# # gdtrajs=[]

# merge_ct = []
# costs = []
# for j in tqdm(range(num_million_trajs)):
# 	ct = 0
# 	index0=j*K
# 	for i in range(K):
# 		d=getDist(rcm_array[j], rcm_array[cols[index0+i]])
# 		ct=cst[index0+i]#*len(tdata_full[j])
# 		if j !=cols[index0+i]:
# 			costs.append( d )
# 			merge_ct.append( ct )#*len(tdata_full[j])
		
# 			if ct>0.3e11 and d < 500:
# 				ptrajs.append( (j, cols[index0+i]) )
# # 			if ct==0:#<1.0e1:#and d < 200
# # # 				print(ct)
# # 				gdtrajs.append( (j, cols[index0+i]) )
	
# # 	if j>201:
# # 		break


# In[118]:


# def getDist(p1, p2):
# 	return np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )

# cst=ec_final2.getFinalMergeCostSenegal() #ec_final4.getFinalMergeCostMillion()
# knn_m_million=ec_final2.getSenegalKNN()

# week=0
# with open('week_tests/com_values0.pickle', 'rb') as handle:
# 	rcm_array = pickle.load(handle)

# rows, cols = knn_m_million.nonzero()

# K=500
# num_million_trajs=int(len(cols)/K)
# print("The number of trajs. is: ", num_million_trajs)

# merge_ct = []
# costs = []
# for j in tqdm(range(num_million_trajs)):
# 	ct = 0
# 	index0=j*K
# 	for i in range(K):
# 		costs.append( getDist(rcm_array[j], rcm_array[cols[index0+i]]) )
# 		merge_ct.append( cst[(j, i)][0]*len(tdata_full[j]) )#*len(tdata_full[j])


# In[119]:


# rows, cols = aknn.nonzero()

# K_SIZE=500
# num=50000

# rcm_array = calculate_COM()

# with open('rg_array_senegal_partial' + str(50000) +  '.pickle', 'rb') as handle:
# 	rg_array = pickle.load(handle)

# for i in tqdm(range(len(rcm_array))):
# 	rcm_array[i].append(rg_array[i])

# min_d=[]
# max_d=[]

# for i in tqdm(range(len(rows))):
# 	if i%500==1:
# 		d=distance_metric10(rcm_array[rows[i]], rcm_array[cols[i]])
# 		min_d.append(d)
# 	if i%500==499:
# 		d=distance_metric10(rcm_array[rows[i]], rcm_array[cols[i]])
# 		max_d.append(d)
# # 		print(d) #rcm_array[rows[i]], rcm_array[cols[i]], 
	
# # 	if i > 10000:
# # 		break
	
# plt.figure(figsize=(12, 12))
# axi = plt.gca()
# #fig, axi = plt.subplots(1, 1)

# sns.ecdfplot(min_d, label="Min. neighbour distance")
# sns.ecdfplot(max_d, label="Max. neighbour distance")

# # plt.xlabel("Trajectory")
# plt.xlabel("Neighbour distance")
# plt.legend()

# # plt.ylim([0, 0.15e10])
# # plt.ylim([0, 0.15e12])
# plt.xlim([0, 1])

# plt.savefig("50k_distance_metric.pdf")


# In[120]:


# def getDist(p1, p2):
# 	return np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )

# cst=ec_final8.getFinalMergeCostPartial() #ec_final4.getFinalMergeCostMillion()
# knn_m_million=ec_final8.getPartialKNN()

# rcm_array = calculate_COM()

# with open('rg_array_senegal_partial' + str(50000) +  '.pickle', 'rb') as handle:
# 	rg_array = pickle.load(handle)

# for i in tqdm(range(len(rcm_array))):
# 	rcm_array[i].append(rg_array[i])

# rows, cols = knn_m_million.nonzero()

# K=500
# num_million_trajs=int(len(cols)/K)
# print("The number of trajs. is: ", num_million_trajs)

# count=0
# merge_ct = defaultdict(list)
# costs = defaultdict(list)

# for i in tqdm(range(len(rows))):
# 	d=distance_metric10(rcm_array[rows[i]], rcm_array[cols[i]])
	
# 	if cst[(rows[i], cols[i])]!=0:
# 		merge_ct[int(i/500)].append( cst[(rows[i], cols[i])] )#*len(tdata_full[j])
# 		costs[int(i/500)].append(d)
		
# # 		if type(cst[(j, cols[index0+i])]) is tuple:
# # 			print( cst[(j, cols[index0+i])] )

# # print(count)


# In[121]:


# for key in tqdm(cst.keys()):
# 	if type(cst[key]) is tuple:
# 		print(key)


# In[122]:


# from scipy import stats

# num_neighbours = [500] #[10, 20, 50, 500]

# co_relation=[]

# for j, val in enumerate(num_neighbours):
# 	co_relation.append([])
# 	for i in tqdm(range(num_million_trajs)):
# 		if len(costs[i])>0:
# 	# 		co_relation.append(stats.pearsonr(costs[i], merge_ct[i]).statistic)
# 			co_relation[j].append(stats.spearmanr(costs[i][:val], merge_ct[i][:val]).correlation)


# In[123]:


# # plt.figure(figsize=(8, 8))
# axi = plt.gca()
# #fig, axi = plt.subplots(1, 1)

# # print(len(costs))

# df=pd.DataFrame(co_relation).T
# # df.columns = num_neighbours

# sns.ecdfplot(df)

# plt.xlabel("Correlation coefficient")

# plt.xlim([0, 1])

# axi.get_legend().remove()
# plt.savefig("co_relation_spearman_all_final.pdf")


# In[124]:


# from sklearn.linear_model import LinearRegression

# linear_regressor = LinearRegression()  # create object for the class
# linear_regressor.fit(np.array(costs).reshape(-1, 1), np.array(merge_ct).reshape(-1, 1))  # perform linear regression


# In[125]:


# x_pred=np.array([random.random() for j in range(1000)]).reshape(-1, 1)
# y_pred = linear_regressor.predict(x_pred)


# In[126]:


# plt.figure(figsize=(12, 12))
# axi = plt.gca()
# #fig, axi = plt.subplots(1, 1)

# # print(len(costs))

# plt.plot(x_pred, y_pred)

# plt.xlabel("Distance metric")
# plt.ylabel("Merge Cost")

# # plt.ylim([0, 1e9])
# # # plt.ylim([0, 0.15e12])
# plt.xlim([0, 1])

# # plt.xlim([1, 60000])

# plt.savefig("regression_50k_0.pdf")


# In[127]:


# random.seed(10)

# plt.figure(figsize=(12, 12))
# axi = plt.gca()
# #fig, axi = plt.subplots(1, 1)

# # print(len(costs))

# plt.scatter(costs, merge_ct, alpha=0.01)

# plt.xlabel("Distance metric")
# plt.ylabel("Merge Cost")

# plt.ylim([0, 1e9])
# # # plt.ylim([0, 0.15e12])
# plt.xlim([0, 1])

# # plt.xlim([1, 60000])

# plt.savefig("plot_dist_partial0.pdf")


# In[128]:


# plt.figure(figsize=(12, 12))
# axi = plt.gca()
# #fig, axi = plt.subplots(1, 1)

# # print(len(costs))

# plt.scatter(costs[:10000000], merge_ct[:10000000], alpha=0.05)

# plt.xlabel("Distance metric")
# plt.ylabel("Merge Cost")

# plt.ylim([0, 1e9])
# # # plt.ylim([0, 0.15e12])
# # plt.xlim([0, 1])

# plt.xlim([1, 60000])

# plt.savefig("plot_dist_partial1.pdf")


# In[129]:


# del rows, cols

# # print(len(tdata_full))


# In[130]:


# print(len(ptrajs))

# # x=set()
# # for tr in gdtrajs:
# # 	x.add(tr[0])
# # # 	x.add(tr[1])
# # print(len(x))

# # x=set()
# # for tr in gdtrajs:
# # 	x.add(tr[1])
# # # 	x.add(tr[1])
# # print(len(x))

# # # with open('fault_tajs_million.pickle', 'wb') as handle:
# # # 	pickle.dump(gdtrajs, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[131]:


# with open('million_not_good_pairs.pickle', 'wb') as handle:
# 	pickle.dump(ptrajsrajs, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[132]:


# with open('million_not_good_pairs.pickle', 'rb') as handle:
# 	ti = pickle.load(handle)


# In[133]:


# import matplotlib.pyplot as plt
# from scipy import spatial
# from matplotlib.collections import PatchCollection
# from matplotlib.patches import Rectangle
# %matplotlib inline

# def plot_trajs(pair, val):
# 	nb_gen_traj_to_plot = 2

# 	fig, ax = plt.subplots(1,1,figsize=(15,20)) #2
# 	ax.set_prop_cycle(color=[
# 		'#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
# 		'#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
# 		'#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
# 		'#17becf', '#9edae5'])
	
# 	markers=['x','o']
# 	nb_ori_traj_plotted = 2
# 	for itraj in range(len(pair)):
# 		trajs=tdata_full[pair[itraj]]
# 		traj_ordered = trajs[trajs[:, 2].argsort()]  # sorting by 2nd col, tmin
# 		traj_ordered = traj_ordered[traj_ordered[:, 2]>0]  # get rid of points that were actually just padding
		
# # 		print(traj_ordered[:, 2])
		
# 		clrs = sns.color_palette("coolwarm", len(traj_ordered[:, 0]))
		
# 		ax.plot(traj_ordered[:, 0], 
# 				traj_ordered[:, 1],
# # 				traj_ordered[:, 2],
# # 				c=clrs[itraj],
# 				marker=markers[itraj], linewidth=1)
		
# # 		ax.scatter(traj_ordered[:, 0], 
# # 				traj_ordered[:, 1],
# # 				traj_ordered[:, 2])


# 	ax.spines.right.set_visible(False)
# 	ax.spines.top.set_visible(False)
# 	ax.spines.bottom.set_visible(False)
# 	ax.spines.left.set_visible(False)

# 	ax.set_title('{} individual trajectories'.format(nb_ori_traj_plotted))

# # 	ax = axes[1]

# # 	for itraj in np.arange(len(gdh.trajs))[sol_mask][:nb_gen_traj_to_plot]:
# # 		traj = gdh.trajs[itraj]

# # 		traj_nopad = traj[traj[:,0,2]>0]

# # 		traj_ordered = traj_nopad[traj_nopad[:, 0, 2].argsort()]

# # 		plottraj = ax.plot((traj_ordered[:,1,0]+traj_ordered[:,0,0])/2,  # middle of the box
# # 						   (traj_ordered[:,1,1]+traj_ordered[:,0,1])/2, 
# # 							linewidth=0.5)

# # 		# print generalized zone of each point
# # 		rectangles = []
# # 		for point in traj_nopad:
# # 			if (point[1]-point[0]).sum()>0:
# # 				rectangles += [Rectangle((point[0,0], point[0,1]), 
# # 										  point[1,0]-point[0,0], 
# # 										  point[1,1]-point[0,1])]
# # 		pc = PatchCollection(rectangles, facecolor=plottraj[-1].get_color(), alpha=0.5)
# # 		ax.add_collection(pc)

# # 		# print points of each original trajectory that make up the generalized trajectory
# # 		for ori_traj in trajs[gdh.original_content[itraj]]:
# # 			traj_ordered = ori_traj[ori_traj[:,0, 2].argsort()]  # sort by time
# # 			ax.plot(traj_ordered[:, 0, 0], 
# # 					traj_ordered[:, 0, 1], 
# # 					marker='o', linewidth=0, color = plottraj[-1].get_color())

# # 	ax.spines.right.set_visible(False)
# # 	ax.spines.top.set_visible(False)
# # 	ax.spines.bottom.set_visible(False)
# # 	ax.spines.left.set_visible(False)
# # 	ax.set_title('{} aggregated trajectories'.format(nb_gen_traj_to_plot))
# 	plt.savefig('weird_pairs'+str(val)+'.pdf')

# 	plt.show()


# In[134]:


# val=25

# plot_trajs(ti[val], val)


# In[135]:


# x=defaultdict(int)

# for tr in gdtrajs:
# 	x[tr[0]]=x[tr[0]]+1

# plt.figure(figsize=(12, 12))
# axi = plt.gca()
# #fig, axi = plt.subplots(1, 1)

# sns.histplot(list(x.values()))

# plt.xlabel("Trajectory")
# plt.ylabel("Number of zero merges")

# # plt.ylim([0, 0.15e10])
# # plt.ylim([0, 0.15e12])
# # plt.xlim([0, 1000])

# plt.savefig("plot_zero_merges.pdf")


# In[136]:


# x=defaultdict(int)

# for tr in gdtrajs:
# 	x[tr[1]]=x[tr[1]]+1

# plt.figure(figsize=(12, 12))
# axi = plt.gca()
# #fig, axi = plt.subplots(1, 1)

# sns.histplot(list(x.values()))

# plt.xlabel("Trajectory")
# plt.ylabel("Number of zero merges")

# # plt.ylim([0, 0.15e10])
# # plt.ylim([0, 0.15e12])
# # plt.xlim([0, 1000])

# plt.savefig("plot_zero_merges_vertical.pdf")


# In[137]:


# # print(len(set([val[0] for val in gdtrajs])))

# print(gdtrajs[0])

# print(tdata_full[0], len(tdata_full[0]))

# print(tdata_full[970233], len(tdata_full[970233]))


# In[138]:


# print(ptrajs[0])

# print(tdata_full[11], len(tdata_full[11]))

# print(tdata_full[640595], len(tdata_full[640595]))


# In[139]:


# idx1, idx2 = 0, 970233

# tp=[]
# for samp in tdata_full[idx1]:
# 	tp.append([*samp,0])

# for samp in tdata_full[idx2]:
# 	tp.append([*samp,1])

# this_results = int(get_node_pairs_m(np.vstack(np.array(tp)))[0]) #np.array([*traj1, *traj2])

# print(this_results)


# In[140]:


# plt.figure(figsize=(12, 12))
# axi = plt.gca()
# #fig, axi = plt.subplots(1, 1)

# plt.scatter(costs[:100000], merge_ct[:100000], alpha=0.05)

# plt.xlabel("Distance between CMs")
# plt.ylabel("Merge Cost")

# plt.ylim([0, 0.15e10])
# # plt.ylim([0, 0.15e12])
# plt.xlim([0, 5000])

# plt.savefig("plot_dist_million.pdf")


# In[141]:


# # with open("merge_ct_million.pickle", 'wb') as handle:
# # 	pickle.dump(merge_ct, handle, protocol=pickle.HIGHEST_PROTOCOL)

# del merge_ct

# with open("ct_million.pickle", 'wb') as handle:
# 	pickle.dump(costs, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[142]:


# plt.figure(figsize=(12, 12))
# axi = plt.gca()
# #fig, axi = plt.subplots(1, 1)

# plt.scatter(costs[:100000], merge_ct[:100000], alpha=0.05)

# plt.xlabel("Distance between CMs")
# plt.ylabel("Merge Cost")

# # plt.ylim([0, 0.15e10])
# plt.ylim([0, 0.15e12])

# # plt.ylim([0, 0.5e10])
# # plt.xlim([0, 1000])

# plt.savefig("plot_dist_senegal_without.pdf")


# In[143]:


# SenegalPlotCosts(ec_final2.getFinalMergeCostSenegal())


# In[144]:


# MillionPlotCosts(ec_final4.getFinalMergeCostMillion(), ec_final2.getFinalMergeCostSenegal())


# ## Trajectory Graph

# In[145]:


# from itertools import chain
# import networkx as nx


# In[146]:


class trajGraphs:
	def __init__(self, nt, K):
		self.num_trajs=nt
		self.chosen_k=K
		
		self.dig_shenzen=0
		self.dig_senegal=0
		self.dig_civ=0
		self.dig_million=0
		
		self.g_shenzen=0
		self.g_senegal=0
		self.g_civ=0
		self.g_million=0
		self.g_partial=0
		self.g_week=0
		
		self.knn_m=0
		self.knn_m_senegal=0
		self.knn_m_civ=0
		self.knn_m_million=0
		self.knn_m_partial=0
		self.knn_m_week=0
		
		self.faulty_civ=[43066, 43067, 43068, 43069, 43070, 43071, 43072, 43073, 43074, 43075, 43076, 43077, 43078, 43079, 43080, 43081, 43082, 43083, 43084, 43085, 43086, 43087, 43088, 43089, 43090, 43091, 43092, 43093, 43094, 43095, 43096, 43097, 43098, 43099, 43100, 43101, 43102, 43103, 43104, 43105, 43106, 43107, 43108, 43109, 43110, 43111, 43112, 43113, 43114, 43115, 43116, 43117, 43118, 43119, 43120, 43121, 43122, 43123, 43124, 43125, 43126, 43127, 43128, 43129, 43130, 43131, 43132, 43133, 43134, 43135, 43136, 43137, 43138, 43139, 43140, 43141, 43142, 43143, 43144, 43145, 43146, 43147]
	
	def setKNNSenegal(self, iknn):
		self.knn_m_senegal=iknn
	
	def setKNNShenzen(self, iknn):
		self.knn_m=iknn
	
	def setKNNCiv(self, iknn):
		self.knn_m_civ=iknn
	
	def setKNNMillion(self, iknn):
		self.knn_m_million=iknn
	
	def setKNNPartial(self, iknn):
		self.knn_m_partial=iknn
	
	def setKNNWeek(self, iknn):
		self.knn_m_week=iknn
	
	def getGraphShenzen(self):
		return self.g_shenzen
	
	def getGraphSenegal(self):
		return self.g_senegal
	
	def getGraphCiv(self):
		return self.g_civ
	
	def getGraphMillion(self):
		return self.g_million
	
	def getGraphPartial(self):
		return self.g_partial
	
	def getGraphWeek(self):
		return self.g_week
	
	def enumerate_digraphs(self, scenario, kmerge_cost):
		g = nx.DiGraph()
		g.add_nodes_from([i for i in range(self.num_trajs)])
		
		if scenario=="Shenzen":
			rows, cols = self.knn_m.nonzero()
		elif scenario=="Senegal":
			rows, cols = self.knn_m_senegal.nonzero()

		# create a list of possible edges
		empty_i=set()
		for i in tqdm(range(self.num_trajs)):
			edges = []
			costs={}
			for j in range(self.chosen_k):
				l, m = rows[i*self.chosen_k + j], cols[i*self.chosen_k + j]
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
	
	def enumerate_graphs(self, scenario, kmerge_cost):
		if scenario=="Shenzen":
			rows, cols = self.knn_m.nonzero()
		elif scenario=="Senegal":
			rows, cols = self.knn_m_senegal.nonzero()
		elif scenario=="Partial":
			rows, cols = self.knn_m_partial.nonzero()

		# create a list of possible edges
		#print(num_trajs)
# 		empty_i=set()
# 		for i in tqdm(range(self.num_trajs)):
# 			edges = []
# 			costs={}
# 			for j in range(self.chosen_k):
# 				l, m = rows[i*self.chosen_k + j], cols[i*self.chosen_k + j]
# 				if len(kmerge_cost[(l, m)])==0:
# 					empty_i.add(l)
# 				else:
# 					self.g_senegal.add_edge(l, m, weight=kmerge_cost[(l, m)][0])
# 	# 				#print(kmerge_cost[(l, m)])
# 	# 				costs[(l, m)] = kmerge_cost[(l, m)][0]
# 	# 				edge1 = (l, m)
# 	# 				edges.append(edge1)
# 	# 		print(costs.items())
# 	# 		nx.set_edge_attributes(g, costs)
# 	# 		g.add_edges_from(edges)
# 			#break
		count=0
		if scenario=="Partial":
			self.g_partial = nx.Graph()
			self.g_partial.add_nodes_from([i for i in range(len(tdata_full))])
			
# 			with open('common_costs_size_test.pickle', 'rb') as handle:
# 				common_costs = pickle.load(handle)

			with open('senegal_data_rand_'+str(self.num_trajs)+'.pickle', 'rb') as handle:
				rand_samples = pickle.load(handle)
			
			tr = list(rand_samples)
			
# 			traj_dict=defaultdict(int)
# 			for i, val in enumerate(rand_samples):
# # 				traj_dict[val]=i
			
# 			for key in kmerge_cost.keys():
# # 				print(kmerge_cost[key])
# 				if type(kmerge_cost[key]) is not tuple:
# 					if kmerge_cost[key]!=0:
# 						self.g_partial.add_edge(traj_dict[key[0]], traj_dict[key[1]], weight=kmerge_cost[key])
# 					else:
# 						count=count+1
# # 						print(traj_dict[key[0]], traj_dict[key[1]])
# 				else:
# 					if len(kmerge_cost[key])!=0:
# 						self.g_partial.add_edge(traj_dict[key[0]], traj_dict[key[1]], weight=kmerge_cost[key][0])
# 					else:
# 						count=count+1
# 	# 					print(key, traj_dict[key[0]], traj_dict[key[1]], common_costs[key])
# 			print("There are: ", count, " zero cost pairs!")

			for j in tqdm(range(len(tdata_full))):
				ct = 0
				index0=j*self.chosen_k
				for i in range(self.chosen_k):
# 					key=(tr[j], tr[cols[index0+i]])

					self.g_partial.add_edge(j, cols[index0+i], weight=kmerge_cost[(j, cols[index0+i])])
			print(self.g_partial)
		
		elif scenario=="Week":
			self.g_week = nx.Graph()
			self.g_week.add_nodes_from([i for i in range(self.num_trajs)])
			
			for key in kmerge_cost.keys():
				if kmerge_cost[key]!=0:
					self.g_week.add_edge(key[0], key[1], weight=kmerge_cost[key])
				else:
					count=count+1
			print("There are: ", count, " zero cost pairs!")
			print(self.g_week)
		
		else:
			self.g_senegal = nx.Graph()
			self.g_senegal.add_nodes_from([i for i in range(self.num_trajs)])
			
			for key in kmerge_cost.keys():
				if kmerge_cost[key]!=0:
					self.g_senegal.add_edge(key[0], key[1], weight=kmerge_cost[key][0])
				else:
					count=count+1
			print("There are: ", count, " zero cost pairs!")
			print(self.g_senegal)
	
	def enumerate_graphs_million(self, scenario, kmerge_cost):
		files=glob.glob(kmerge_cost[0]+"*")
		self.g_million = nx.Graph()
		self.g_million.add_nodes_from([i for i in range(self.num_trajs)])

# 		rows, cols = self.knn_m_million.nonzero()
		count=0
		
		for file in tqdm(files):
			with open(file, 'rb') as handle:
				this_cost = pickle.load(handle)
			edge_count=defaultdict(int)
			for key in this_cost.keys():
				edge_count[key[0]]=edge_count[key[0]]+1
				if this_cost[key]!=0 and edge_count[key[0]]<=600:
					self.g_million.add_edge(key[0], key[1], weight=this_cost[key])
				else:
					count=count+1
		print("There are: ", count, " zero cost pairs!")
		print(self.g_million)
	
	def enumerate_graphs_million2(self, scenario, kmerge_cost):
		self.g_million = nx.Graph()
		self.g_million.add_nodes_from([i for i in range(self.num_trajs)])
		
		rows, cols = self.knn_m_million.nonzero()

		i=0
		K=500
		edge_count=defaultdict(int)
		for val in kmerge_cost:
			edge_count[int(i/K)]=edge_count[int(i/K)]+1
# 			print(int(i/K), cols[int(i/K)*K+i%K])
			if edge_count[int(i/K)]<=250 and val!=0:
				self.g_million.add_edge(int(i/K), cols[int(i/K)*K+i%K], weight=val)
			i=i+1
		print(self.g_million)
		
		del kmerge_cost
		
# 		print(self.g_million)
		print("Graphs are created")
	
	def enumerate_graphs_shenzen(self, scenario, kmerge_cost_shenzen):
		self.g_shenzen = nx.Graph()
		self.g_shenzen.add_nodes_from([i for i in range(self.num_trajs)])
		
		if scenario=="Shenzen":
			rows, cols = self.knn_m.nonzero()
		elif scenario=="Senegal":
			rows, cols = self.knn_m_senegal.nonzero()

		# create a list of possible edges
		#print(num_trajs)
# 		empty_i=set()
# 		for i in tqdm(range(self.num_trajs)):
# 			edges = []
# 			costs={}
# 			for j in range(self.chosen_k):
# 				l, m = rows[i*self.chosen_k + j], cols[i*self.chosen_k + j]
# 				if len(kmerge_cost_shenzen[(l, m)])==0:
# 					empty_i.add(l)
# 				else:
# 					self.g_shenzen.add_edge(l, m, weight=kmerge_cost_shenzen[(l, m)][0])
# 	# 				#print(kmerge_cost[(l, m)])
# 	# 				costs[(l, m)] = kmerge_cost[(l, m)][0]
# 	# 				edge1 = (l, m)
# 	# 				edges.append(edge1)
# 	# 		print(costs.items())
# 	# 		nx.set_edge_attributes(g, costs)
# 	# 		g.add_edges_from(edges)
# 			#break
		count=0
		for key in kmerge_cost_shenzen.keys():
			if len(kmerge_cost_shenzen[key])!=0:
				self.g_shenzen.add_edge(key[0], key[1], weight=kmerge_cost_shenzen[key][0])
			else:
				count=count+1
		print("There are: ", count, " zero cost pairs!")
		print(self.g_shenzen)
	
	def enumerate_graphs_civ(self, scenario, kmerge_cost_civ):
		self.g_civ = nx.Graph()
		nodes=[i for i in range(self.faulty_civ[0])]+[i for i in range(self.faulty_civ[-1]+1, self.num_trajs)]
		
		self.g_civ.add_nodes_from(nodes)
		
# 		with open('knn_matrix_civ100.pickle', 'rb') as handle:
# 			self.knn_m_civ = pickle.load(handle)
		
		if scenario=="Shenzen":
			rows, cols = self.knn_m.nonzero()
		elif scenario=="Senegal":
			rows, cols = self.knn_m_senegal.nonzero()
		elif scenario=="Civ":
			rows, cols = self.knn_m_civ.nonzero()

		# create a list of possible edges
		#print(num_trajs)
# 		empty_i=set()
# 		for i in tqdm(range(self.num_trajs)):
# 			edges = []
# 			costs={}
# 			for j in range(self.chosen_k):
# 				l, m = rows[i*self.chosen_k + j], cols[i*self.chosen_k + j]
# 				if len(kmerge_cost_shenzen[(l, m)])==0:
# 					empty_i.add(l)
# 				else:
# 					self.g_shenzen.add_edge(l, m, weight=kmerge_cost_shenzen[(l, m)][0])
# 	# 				#print(kmerge_cost[(l, m)])
# 	# 				costs[(l, m)] = kmerge_cost[(l, m)][0]
# 	# 				edge1 = (l, m)
# 	# 				edges.append(edge1)
# 	# 		print(costs.items())
# 	# 		nx.set_edge_attributes(g, costs)
# 	# 		g.add_edges_from(edges)
# 			#break
		count=0
		for key in kmerge_cost_civ.keys():
			if kmerge_cost_civ[key]!=0 and key[0] not in self.faulty_civ and key[1]<101:
				self.g_civ.add_edge(key[0], key[1], weight=kmerge_cost_civ[key])
# 			else:
# 				count=count+1
# 		print("There are: ", count, " zero cost pairs!")
		print(self.g_civ)
	
	def find_num_disconnected_components():
		cluster_ids=set()
		#nodes = digraphs.nodes()

		print("Number of strongly connected components are: ", nx.number_strongly_connected_components(self.dig_senegal))

		c_comp = nx.kosaraju_strongly_connected_components(self.dig_senegal)

		l_comp = [len(comp) for comp in c_comp]

	# 	for comp in c_comp:
	# 		print(comp)

		fig, ax = plt.subplots()

		df = pd.Series(l_comp)
		sns.boxplot(df)
		plt.xlabel("Len. of strongly connected comps.")

		plt.savefig("sconnected_comps.pdf")

		#find_num_disconnected_components()
	
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
	
	def get_graph_info(g_):
		print(nx.maximal_independent_set(digraphs))

		# get_graph_info()
	
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


# In[147]:


# tg = trajGraphs(ec.getNumShenzenTrajs(), ec.getK())
# tg.setKNNShenzen(ec.getShenzenKNN())
# # # # print(ec.getMergeCostShenzen())
# # # tg.enumerate_digraphs("Shenzen", ec.getMergeCostShenzen())
# # print(digraphs)
# tg.enumerate_graphs_shenzen("Shenzen", ec.getMergeCostShenzen())


# In[148]:


# tg_final = trajGraphs(ec_final.getNumShenzenTrajs(), ec_final.getK())
# tg_final.setKNNShenzen(ec_final.getShenzenKNN())
# # print(list(ec_final.getFinalMergeCostShenzen().keys())[1])
# # # tg.enumerate_digraphs("Shenzen", ec.getMergeCostShenzen())
# # print(digraphs)
# # tg_final.enumerate_graphs_shenzen("Shenzen", ec_final.getFinalMergeCostShenzen())


# In[149]:


# tg_final2 = trajGraphs(ec_final2.getNumSenegalTrajs(), ec_final2.getK())
# tg_final2.setKNNSenegal(ec_final2.getSenegalKNN())
# # # print(list(ec_final.getFinalMergeCostShenzen().keys())[1])
# # # # tg.enumerate_digraphs("Shenzen", ec.getMergeCostShenzen())
# # # print(digraphs)
# # tg_final2.enumerate_graphs("Senegal", ec_final2.getFinalMergeCostSenegal())


# In[150]:


# tg_final3 = trajGraphs(ec_final3.getNumCivTrajs(), ec_final3.getK())
# tg_final3.setKNNCiv(ec_final3.getCivKNN())
# # print(list(ec_final.getFinalMergeCostShenzen().keys())[1])
# # tg.enumerate_digraphs("Shenzen", ec.getMergeCostShenzen())
# # print(digraphs)
# tg_final3.enumerate_graphs_civ("Civ", ec_final3.getFinalMergeCostCiv())


# In[151]:


# tg_final4 = trajGraphs(ec_final4.getNumMillionTrajs(), ec_final4.getK())
# tg_final4.setKNNMillion(ec_final4.getMillionKNN())
# # print(list(ec_final.getFinalMergeCostShenzen().keys())[1])
# # tg.enumerate_digraphs("Shenzen", ec.getMergeCostShenzen())
# # print(digraphs)
# tg_final4.enumerate_graphs_million("Million", ec_final4.getFinalMergeCostMillion())


# In[152]:


# j=0
# tg_final5 = trajGraphs(sub_sampled_datasets[j]*1000, ec_final5.getK())
# tg_final5.setKNNSenegal(ec_final5.getSenegalKNN())
# # # print(list(ec_final.getFinalMergeCostShenzen().keys())[1])
# # # # tg.enumerate_digraphs("Shenzen", ec.getMergeCostShenzen())
# # # print(digraphs)
# tg_final5.enumerate_graphs("Senegal", ec_final5.getFinalMergeCostSenegal())


# In[153]:


# tg_final6 = trajGraphs(ec_final6.getNumPartialTrajs(), ec_final6.getK())
# tg_final6.setKNNPartial(ec_final6.getPartialKNN())
# tg_final6.enumerate_graphs("Partial", ec_final6.getFinalMergeCostPartial())


# In[154]:


# tg_final7 = trajGraphs(ec_final7.getNumWeekTrajs(), ec_final7.getK())
# tg_final7.setKNNWeek(ec_final7.getWeekKNN())
# tg_final7.enumerate_graphs("Week", ec_final7.getFinalMergeCostWeek())


# In[155]:


# tg_final8 = trajGraphs(ec_final8.getNumPartialTrajs(), ec_final8.getK())
# tg_final8.setKNNPartial(ec_final8.getPartialKNN())
# tg_final8.enumerate_graphs("Partial", ec_final8.getFinalMergeCostPartial())


# In[156]:


# tg_final9 = trajGraphs(ec_final9.getNumMillionTrajs(), ec_final9.getK())
# tg_final9.setKNNMillion(ec_final9.getMillionKNN())
# tg_final9.enumerate_graphs_million2("Million", ec_final9.getFinalMergeCostMillion())


# ## Two Optimal Showcase

# In[157]:


from multiprocessing import Process
from multiprocessing import Pool, freeze_support
from community import community_louvain
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.optimize import linear_sum_assignment
from itertools import repeat
import cProfile
import math

class twoOptimalShowcase:
	def __init__(self):
		self.knn_m=[]
		self.ck_vals=[10, 20, 30, 40, 50, 100, 250, 500, 1000, 2000] #2, 5,
		self.knn_m_random=[]
		
		self.partion_there=True #True
		self.partion_there_shenzen=True #True
		self.partion_there_civ=True #True
		self.partion_there_million=True #True
		self.partion_there_partial=True #True
		self.partion_there_week=True #True
		
		self.partition=0
		
		self.com_id_size = [67, 37, 10, 54] #size variation
		self.com_id_size_shenzen = [19, 9, 18, 5] #size variation shenzen
		self.com_id_density = [39.0, 7.0, 57.0, 65.0] #density variation ids
		self.com_density = [1.230683e+03, 1.772690e+04, 5.349870e+05, 4.385194e+06] #corresponding densities

		self.com_id_density_shenzen = [18.0, 12.0, 27.0, 23.0] #density variation shenzen
		self.com_density_shenzen = [1582.148117, 5217.418016, 10864.524687, 105912.595293] #corresponding densities
		self.com_size = ["1000-2000", "2000-3000", "3000-4000", "4000-5000"]
	
	def getShenzenCommunities(self):
		return self.partition_shenzen
	
	def getSenegalCommunities(self):
		return self.partition
	
	def getCivCommunities(self):
		return self.partition_civ
	
	def getMillionCommunities(self):
		return self.partition_million
	
	def getPartialCommunities(self):
		return self.partition_partial
	
	def getWeekCommunities(self):
		return self.partition_week
	
	def obtainKNN(self):
		for K_SIZE in tqdm(self.ck_vals):
			with open('knn_matrix_dakar' + str(K_SIZE) +  '.pickle', 'rb') as handle:
				self.knn_m.append(pickle.load(handle))
		
		for K_SIZE in tqdm(self.ck_vals):
			with open('knn_matrix_random' + str(K_SIZE) +  '.pickle', 'rb') as handle:
				self.knn_m_random.append(pickle.load(handle))
	
	def senegalCommunity(self):
		if not self.partion_there:
			#first compute the best partition
			self.partition = community_louvain.best_partition(graphs)#, weight='kmerge'
			print("Partioned...")

			#pickling partitions
			with open('community_partitions.pickle', 'wb') as handle:
				pickle.dump(self.partition, handle, protocol=pickle.HIGHEST_PROTOCOL)
			print("Pickled...")
		else:
			with open('community_partitions.pickle', 'rb') as handle:
				self.partition = pickle.load(handle)
	
	def partialCommunity(self, num_trajs, graphs=0):
		if not self.partion_there_partial:
			#first compute the best partition
			print("Partitioning...")
			
			self.partition_partial = community_louvain.best_partition(graphs)#, weight='kmerge'
			print("Partioned...")
			
			self.num_trajs=num_trajs

			#pickling partitions
			with open('community_partitions_partial'+str(num_trajs)+'.pickle', 'wb') as handle:
				pickle.dump(self.partition_partial, handle, protocol=pickle.HIGHEST_PROTOCOL)
			print("Pickled...")
		else:
			with open('community_partitions_partial'+str(num_trajs)+'.pickle', 'rb') as handle:
				self.partition_partial = pickle.load(handle)
	
	def weekCommunity(self, num_trajs, graphs=0):
		if not self.partion_there_week:
			#first compute the best partition
			self.partition_week = community_louvain.best_partition(graphs)#, weight='kmerge'
			print("Partioned...")
			
			self.num_trajs=num_trajs

			#pickling partitions
			with open('community_partitions_week'+str(num_trajs)+'.pickle', 'wb') as handle:
				pickle.dump(self.partition_week, handle, protocol=pickle.HIGHEST_PROTOCOL)
			print("Pickled...")
		else:
			with open('community_partitions_week'+str(num_trajs)+'.pickle', 'rb') as handle:
				self.partition_week = pickle.load(handle)
	
	def shenzenCommunity(self):
		if not self.partion_there_shenzen:
			#first compute the best partition
			self.partition_shenzen = community_louvain.best_partition(graphs)#, weight='kmerge'
			print("Partioned...")

			#pickling partitions
			with open('community_partitions_shenzen.pickle', 'wb') as handle:
				pickle.dump(self.partition_shenzen, handle, protocol=pickle.HIGHEST_PROTOCOL)
			print("Pickled...")
		else:
			with open('community_partitions_shenzen.pickle', 'rb') as handle:
				self.partition_shenzen = pickle.load(handle)
	
	def civCommunity(self, graphs=0):
		if not self.partion_there_civ:
			#first compute the best partition
			self.partition_civ = community_louvain.best_partition(graphs)#, weight='kmerge'
			print("Partioned...")

			#pickling partitions
			with open('community_partitions_civ.pickle', 'wb') as handle:
				pickle.dump(self.partition_civ, handle, protocol=pickle.HIGHEST_PROTOCOL)
			print("Pickled...")
		else:
			with open('community_partitions_civ.pickle', 'rb') as handle:
				self.partition_civ = pickle.load(handle)
	
	def millionCommunity(self, graphs=0):
		if not self.partion_there_million:
			#first compute the best partition
			print("Partitioning...")
			
			self.partition_million = community_louvain.best_partition(graphs)#, weight='kmerge'
			print("Partioned...")

			#pickling partitions
			with open('community_partitions_million.pickle', 'wb') as handle:
				pickle.dump(self.partition_million, handle, protocol=pickle.HIGHEST_PROTOCOL)
			print("Pickled...")
		else:
			with open('community_partitions_million.pickle', 'rb') as handle:
				self.partition_million = pickle.load(handle)
	
	def senegalFinalCommunity(self, graphs):
		if not self.partion_there:
			#first compute the best partition
			self.partition_final = community_louvain.best_partition(graphs)#, weight='kmerge'
			print("Partioned...")

			#pickling partitions
			with open('community_partitions_final.pickle', 'wb') as handle:
				pickle.dump(self.partition_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
			print("Pickled...")
		else:
			with open('community_partitions_final.pickle', 'rb') as handle:
				self.partition_final = pickle.load(handle)
	
	def shenzenFinalCommunity(self, graphs):
		if not self.partion_there_shenzen:
			#first compute the best partition
			self.partition_shenzen_final = community_louvain.best_partition(graphs)#, weight='kmerge'
			print("Partioned...")

			#pickling partitions
			with open('community_partitions_shenzen_final.pickle', 'wb') as handle:
				pickle.dump(self.partition_shenzen_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
			print("Pickled...")
		else:
			with open('community_partitions_shenzen.pickle', 'rb') as handle: #'community_partitions_shenzen_final.pickle'
				self.partition_shenzen_final = pickle.load(handle)
	
	def odDict(self):
		with open('od_values.pickle', 'rb') as handle:
			self.od_dict = pickle.load(handle)
	
	def plot_dist(self, dt, xlab, name, ylab=""):
		fig, axi = plt.subplots(1, 1)

		df = pd.DataFrame(dt)
		df.columns=["size"]

# 		print(df)
		sns.ecdfplot(data=df)

		plt.xlabel(xlab)
		if len(ylab):
			plt.ylabel(ylab)
		#plt.legend()
	# 	if xlab[-6:]=="random":
	# 		plt.xlim([0, 200])
	# 	else:
	# 		plt.xlim([0, 5])
		axi.get_legend().remove()
# 		plt.savefig(str(name) + '.pdf')
	
	def communitySizes(self):
# 		self.count=defaultdict(int)
# 		for com in set(self.partition.values()):
# 			list_nodes = [nodes for nodes in self.partition.keys() if self.partition[nodes] == com]
# 			#print(len(list_nodes))
# 			self.count[com]=len(list_nodes)

# # 		print(list(count.values()))
# 		self.plot_dist(list(self.count.values()), "community sizes", "community_sizes")
		
# 		self.count2=defaultdict(int)
# 		for com in set(self.partition_shenzen.values()):
# 			list_nodes = [nodes for nodes in self.partition_shenzen.keys() if self.partition_shenzen[nodes] == com]
# 			#print(len(list_nodes))
# 			self.count2[com]=len(list_nodes)

# # 		print(self.count2)
# 		self.plot_dist(list(self.count2.values()), "community sizes", "community_sizes_shenzen")
		
# 		self.count3=defaultdict(int)
# 		for com in set(self.partition_civ.values()):
# 			list_nodes = [nodes for nodes in self.partition_civ.keys() if self.partition_civ[nodes] == com]
# # 			print(len(list_nodes))
# 			self.count3[com]=len(list_nodes)

# 		print(self.count3)
# 		self.plot_dist(list(self.count3.values()), "community sizes", "community_sizes_civ")
		
		self.count4=defaultdict(int)
		for nodes in tqdm(self.partition_million.keys()):
# 			list_nodes = [nodes for nodes in self.partition_million.keys() if self.partition_million[nodes] == com]
# 			print(len(list_nodes))
# 			if len(list_nodes)>1:
# 				self.count4[com]=len(list_nodes)
			self.count4[self.partition_million[nodes]]=self.count4[self.partition_million[nodes]]+1

		print(self.count4)
		self.plot_dist(list(self.count4.values()), "community sizes", "community_sizes_million")

# 		self.count5=defaultdict(int)
# 		for com in set(self.partition_partial.values()):
# 			list_nodes = [nodes for nodes in self.partition_partial.keys() if self.partition_partial[nodes] == com]
# # 			print(len(list_nodes))
# 			if len(list_nodes)>1:
# 				self.count5[com]=len(list_nodes)

# 		print(self.count5)
# 		self.plot_dist(list(self.count5.values()), "community sizes", "community_sizes_partial"+str(self.num_trajs))

# 		self.count6=defaultdict(int)
# 		for com in set(self.partition_week.values()):
# 			list_nodes = [nodes for nodes in self.partition_week.keys() if self.partition_week[nodes] == com]
# # 			print(len(list_nodes))
# 			if len(list_nodes)>1:
# 				self.count6[com]=len(list_nodes)

# 		print(self.count6)
# 		self.plot_dist(list(self.count6.values()), "community sizes", "community_sizes_week")
	
	def visualiseCommunities(self):
# 		self.df2=pd.DataFrame(list(self.partition_million.items()))
# 		del self.df2[0]
# 		self.df2.columns=["Community ID"]

# 		print(df2, self.count2)
		
		self.df=pd.DataFrame(list(self.partition.items()))
		del self.df[0]
		self.df.columns=["Community ID"]

# 		print(df)
		
		with open('com_values.pickle', 'rb') as handle:
			self.rcm_array = pickle.load(handle)

# 		with open('com_values_senegal_million.pickle', 'rb') as handle:
# 			self.rcm_array_million = pickle.load(handle)

		# with open('od_values_shenzen.pickle', 'rb') as handle:
		# 	od_dict2=pickle.load(handle)
		#print(rcm_array)

# 		self.df["x"]=[val[0] for val in self.rcm_array_million]
# 		self.df["y"]=[val[1] for val in self.rcm_array_million]
# 		self.df["origin_x"]=[self.od_dict[key][0][0] for key in self.od_dict.keys()]
# 		self.df["origin_y"]=[self.od_dict[key][0][1] for key in self.od_dict.keys()]
# 		self.df["destination_x"]=[self.od_dict[key][1][0] for key in self.od_dict.keys()]
# 		self.df["destination_y"]=[self.od_dict[key][1][1] for key in self.od_dict.keys()]

		self.df["x"]=[val[0] for val in self.rcm_array]
		self.df["y"]=[val[1] for val in self.rcm_array]
# 		self.df["origin_x"]=[self.od_dict[key][0][0] for key in self.od_dict.keys()]
# 		self.df["origin_y"]=[self.od_dict[key][0][1] for key in self.od_dict.keys()]
# 		self.df["destination_x"]=[self.od_dict[key][1][0] for key in self.od_dict.keys()]
# 		self.df["destination_y"]=[self.od_dict[key][1][1] for key in self.od_dict.keys()]
		
# 		print(df)
		
# 		self.df2["x"]=[val[0] for val in self.rcm_array_million]
# 		self.df2["y"]=[val[1] for val in self.rcm_array_million]
		
# 		ct=[]
# 		for com in tqdm(set(self.partition_million.values())):
# 			list_nodes = [nodes for nodes in self.partition_million.keys() if self.partition_million[nodes] == com]
# 			if len(list_nodes)>1:
# 				ct.append(com)

# 		self.df2=self.df2[self.df2["Community ID"].isin(ct)]
# 		print(self.df2)
		
		plt.figure(figsize=(12, 12))
		axi = plt.gca()
		#fig, axi = plt.subplots(1, 1)

		g = sns.scatterplot(x="x", y="y", data=self.df, c=self.df["Community ID"], cmap= "Spectral", s=12) #, hue="Community ID"
# 		#g.fig.set_figwidth(8.27)
# 		#g.fig.set_figheight(11.7)

# 		#axi.legend(loc='upper center', bbox_to_anchor=(1.1, 1.4), ncol=1, fancybox=True, shadow=True)

# 		# #hide x-axis
# 		# axi.get_xaxis().set_visible(False)

# 		# #hide y-axis 
# 		# axi.get_yaxis().set_visible(False)
		plt.savefig("plot_communities_senegal.pdf")
	
	def analyseSpatialCosts(self):
		self.st_devs=defaultdict(list)
		self.commune_com=defaultdict(list)

		for index, row in self.df.iterrows():
			self.st_devs[row["Community ID"]].append( (row["x"], row["y"]) )
			self.commune_com[row["Community ID"]].append( (row["origin_x"], row["origin_y"]) )
		
		for key in self.commune_com.keys():
			com_x, com_y = 0, 0
			for traj in self.commune_com[key]:
				com_x = com_x + traj[0]
				com_y = com_y + traj[1]
			self.commune_com[key]=(com_x/self.count[key], com_y/self.count[key])
		
		self.df_com = pd.DataFrame.from_dict(self.commune_com).T
		self.df_com["Size"] = [self.count[val] for val in list(self.df_com.index)]
		self.df_com.columns = ["X", "Y", "Size"]
# 		print(self.df_com, self.count, self.commune_com, self.df_com[self.df_com["X"]<300000], np.sum(self.df_com[self.df_com["X"]<300000]["Size"]))
		
		plt.figure(figsize=(10, 6))
		axi = plt.gca()
		#fig, axi = plt.subplots(1, 1)

		self.df_dakar=self.df_com[self.df_com["X"]<260000]
# 		print(self.df_com[self.df_com["X"]<260000])

		g = sns.scatterplot(x="X", y="Y", data=self.df_com[self.df_com["X"]<260000], c=self.df_com[self.df_com["X"]<260000].index, cmap= "Spectral", s=60, hue="Size") #, hue="Community ID"
		#g.fig.set_figwidth(8.27)
		#g.fig.set_figheight(11.7)

		#axi.legend(loc='upper center', bbox_to_anchor=(1.1, 1.4), ncol=1, fancybox=True, shadow=True)

		# #hide x-axis
		# axi.get_xaxis().set_visible(False)

		# #hide y-axis 
		# axi.get_yaxis().set_visible(False)
		plt.savefig("plot_communities_com3.pdf")
		
		plt.figure(figsize=(10, 6))
		axi = plt.gca()
		#fig, axi = plt.subplots(1, 1)

		g = sns.scatterplot(x="X", y="Y", data=self.df_com[self.df_com.index.isin(self.com_id_size)], s=100, hue='Size') #data=df_com[df_com.index.isin(com_id_size)]["Size"], c=df_com[df_com.index.isin(com_id_size)]["Size"], cmap= "Spectral", hue="Community ID

		axi.set_xlim([200000, 900000])
		axi.set_ylim([1300000, 1900000])
		plt.savefig("plot_communities_com.pdf")
		
		plt.figure(figsize=(10, 6))
		axi = plt.gca()
		#fig, axi = plt.subplots(1, 1)

		g = sns.scatterplot(x="X", y="Y", data=self.df_com[self.df_com.index.isin(self.com_id_density)], s=100, hue=self.df_com[self.df_com.index.isin(self.com_id_density)].index) #data=df_com[df_com.index.isin(com_id_size)]["Size"], c=df_com[df_com.index.isin(com_id_size)]["Size"], cmap= "Spectral", hue="Community ID"

		axi.set_xlim([200000, 900000])
		axi.set_ylim([1300000, 1900000])
		plt.savefig("plot_communities_com2.pdf")

		#com_id_density = [39.0, 7.0, 57.0, 65.0] #density variation ids
		
		for key in self.st_devs.keys():
			self.st_devs[key]=(np.std([point[0] for point in self.st_devs[key]])/1000.0, np.std([point[1] for point in self.st_devs[key]])/1000.0)

# 		print(self.st_devs)
		
		self.plot_dist([val[0] for val in self.st_devs.values()], "Std. Dev in X-Coord(kms)", "spatial_varriation_x")

		self.plot_dist([val[1] for val in self.st_devs.values()], "Std. Dev in Y-Coord(kms)", "spatial_varriation_y")
	
	def sizeDensityRelation(self):
		plt.figure(figsize=(10, 6))
		self.coms_com=defaultdict(list)

		for index, row in self.df.iterrows():
			self.coms_com[row["Community ID"]].append( (row["x"], row["y"]) )
		
		self.hdensity=defaultdict(tuple)

		for key in self.coms_com.keys():
			if len(self.coms_com[key])>1:
				hull = ConvexHull(self.coms_com[key])
				self.hdensity[key]=(len(self.coms_com[key]), hull.volume/len(self.coms_com[key]))

		# 	else:
		# 		hdensity[key]=(1, 1)

		self.hdensity = pd.DataFrame(self.hdensity).T
		self.hdensity.columns=["size", "density"]
		
		g = sns.histplot(self.hdensity["density"])
		
		plt.figure(figsize=(10, 6))
		g = sns.histplot(self.hdensity["size"], bins=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000])
		
		g = sns.relplot(data=self.hdensity, x='size', y='density')
	
	def shenzenSpatialCosts(self):
		st_devs=defaultdict(list)
		commune_com=defaultdict(list)

		for index, row in self.df2.iterrows():
			st_devs[row["Community ID"]].append( (row["x"], row["y"]) )
			commune_com[row["Community ID"]].append( (row["x"], row["y"]) )

		for key in commune_com.keys():
			com_x, com_y = 0, 0
			for traj in commune_com[key]:
				com_x = com_x + traj[0]
				com_y = com_y + traj[1]
			commune_com[key]=(com_x/self.count2[key], com_y/self.count2[key])

		df_com2 = pd.DataFrame.from_dict(commune_com).T
		df_com2["Size"] = [self.count2[val] for val in list(df_com2.index)]
		df_com2.columns = ["X", "Y", "Size"]
		
		plt.figure(figsize=(10, 6))
		axi = plt.gca()
		#fig, axi = plt.subplots(1, 1)

		g = sns.scatterplot(x="X", y="Y", data=df_com2, c=df_com2["Size"], cmap= "Spectral", s=100) #, hue="Community ID"
		plt.savefig("plot_communities_com_shenzen.pdf")
		
		coms_com=defaultdict(list)

		for index, row in self.df2.iterrows():
			coms_com[row["Community ID"]].append( (row["x"], row["y"]) )

		hdensity2=defaultdict(tuple)

		for key in coms_com.keys():
			if len(coms_com[key])>1:
				hull = ConvexHull(coms_com[key])
				hdensity2[key]=(len(coms_com[key]), hull.volume/len(coms_com[key]))

		# 	else:
		# 		hdensity[key]=(1, 1)

		hdensity2 = pd.DataFrame(hdensity2).T
		hdensity2.columns=["size", "density"]
		
		plt.figure(figsize=(10, 6))
		sns.histplot(hdensity2["density"])
		
		plt.figure(figsize=(10, 6))
		sns.histplot(hdensity2["size"], bins=[0, 1000, 2000, 3000, 4000, 5000, 6000])
		
		g = sns.relplot(data=hdensity2, x='size', y='density')
		
		size_min, size_max = 3000, 4000

		hd = self.hdensity[self.hdensity["size"]>size_min]
		hd = hd[hd["size"]<size_max]
		print(hd)

		hd2 = hdensity2[hdensity2["size"]>size_min]
		hd2 = hd2[hd2["size"]<size_max]
		print(hd2)
		
		size_min, size_max = 1000, 2000

		hd = self.hdensity[self.hdensity["size"]>size_min]
		hd = hd[hd["size"]<size_max]
		print(hd)

		hd2 = hdensity2[hdensity2["size"]>size_min]
		hd2 = hd2[hd2["size"]<size_max]
		print(hd2)
		
		size_min, size_max = 2000, 3000

		hd = self.hdensity[self.hdensity["size"]>size_min]
		hd = hd[hd["size"]<size_max]
		print(hd)

		hd2 = hdensity2[hdensity2["size"]>size_min]
		hd2 = hd2[hd2["size"]<size_max]
		print(hd2)
		
		size_min, size_max = 4000, 5000

		hd = self.hdensity[self.hdensity["size"]>size_min]
		hd = hd[hd["size"]<size_max]
		print(hd)

		hd2 = hdensity2[hdensity2["size"]>size_min]
		hd2 = hd2[hd2["size"]<size_max]
		print(hd2)
		
		ls_trajs=defaultdict(list)
		ls_trajs_shenzen=defaultdict(list)

		for index, row in self.df.iterrows():
			ls_trajs[row["Community ID"]].append( index )

		for index, row in self.df2.iterrows():
			ls_trajs_shenzen[row["Community ID"]].append( index )
		
		print(len(ls_trajs_shenzen[19]), len(ls_trajs[67]))
		
		for j in range(4):
			with open('com_trajectories_size' + str(self.com_id_size[j]) + '.pickle', 'wb') as handle:
				pickle.dump(ls_trajs[self.com_id_size[j]], handle, protocol=pickle.HIGHEST_PROTOCOL)

			with open('com_trajectories_shenzen_size' + str(self.com_id_size_shenzen[j]) + '.pickle', 'wb') as handle:
				pickle.dump(ls_trajs_shenzen[self.com_id_size_shenzen[j]], handle, protocol=pickle.HIGHEST_PROTOCOL)

			with open('com_trajectories_density' + str(self.com_id_density[j]) + '.pickle', 'wb') as handle:
				pickle.dump(ls_trajs[self.com_id_density[j]], handle, protocol=pickle.HIGHEST_PROTOCOL)

			with open('com_trajectories_shenzen_density' + str(self.com_id_density_shenzen[j]) + '.pickle', 'wb') as handle:
				pickle.dump(ls_trajs_shenzen[self.com_id_density_shenzen[j]], handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	def trajsDakarRandom(self):
		self.trajs_dakar = [key for key in self.partition.keys() if self.partition[key]==42][:5000]
# 		print(self.trajs_dakar)

		#pickling investigated trajs. of Dakar
		with open('dakar_trajectories.pickle', 'wb') as handle:
			pickle.dump(self.trajs_dakar, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		indexes=[]
		self.num_trajs_dakar=5000
		num_cores = 32

		for i in range(num_cores):
			il = int(len(self.trajs_dakar)/num_cores)
			indexes.append([j for j in range(i*il, i*il+il)])
		
		with open('kmerge_costs_dakar_trajs.pickle', 'rb') as handle:
			self.kmerge_cost_dakar = pickle.load(handle)

		with open('senegal_trajs_random_5000.pickle', 'rb') as handle:
			self.trajs_senegal_random = pickle.load(handle)
		
		indexes_random=[]
		self.num_trajs_random=5000

		for i in range(num_cores):
			il = int(len(self.trajs_senegal_random)/num_cores)
			indexes_random.append([j for j in range(i*il, i*il+il)])
		
		with open('kmerge_costs_random_trajs.pickle', 'rb') as handle:
			self.kmerge_cost_senegal_random = pickle.load(handle)
		
# 		index_count=defaultdict(int)

# 		for key in self.kmerge_cost_dakar.keys():
# 			index_count[key[0]]=index_count[key[0]]+1

# 		a=set(list(index_count.keys()))
# 		b=set([i for i in range(5000)])
# 		print(b-a)
		
# 		index_count=defaultdict(int)

# 		for key in self.kmerge_cost_senegal_random.keys():
# 			index_count[key[0]]=index_count[key[0]]+1

# 		a=set(list(index_count.keys()))
# 		b=set([i for i in range(5000)])
# 		print(b-a)
	
	def trajMaps(self):
		self.dk_trajs_dict=defaultdict(int)
		self.dk_trajs_reverse=defaultdict(int)

		for i, traj in enumerate(self.trajs_dakar):
			self.dk_trajs_dict[traj]=i
			self.dk_trajs_reverse[i]=traj
		
		self.dk_trajs_dict_random=defaultdict(int)
		self.dk_trajs_reverse_random=defaultdict(int)

		for i, traj in enumerate(self.trajs_senegal_random):
			self.dk_trajs_dict_random[traj]=i
			self.dk_trajs_reverse_random[i]=traj
		
		self.arr_kmerge_cost = np.zeros( (self.num_trajs_dakar, self.num_trajs_dakar) )
		count1, count2 = 0, 0

		for key in self.kmerge_cost_dakar.keys():
			count2=count2+1
			if self.kmerge_cost_dakar[key][0]==0:
				count1=count1+1
		# 	if key[0] < num_trajs_dakar and key[1] < num_trajs_dakar:
			if key[0]==key[1]:
				self.arr_kmerge_cost[key[0]][key[1]]=math.inf
			else:
				self.arr_kmerge_cost[key[0]][key[1]]=self.kmerge_cost_dakar[key][0]

# 		print(self.arr_kmerge_cost, (count1/count2)*100, " percent of k-merges have zero cost!", count1, count2)
		
		self.arr_kmerge_cost_random = np.zeros( (self.num_trajs_dakar, self.num_trajs_dakar) )
		count1, count2 = 0, 0

		for key in self.kmerge_cost_senegal_random.keys():
			count2=count2+1
			if self.kmerge_cost_senegal_random[key][0]==0:
				count1=count1+1
		# 	if key[0] < num_trajs_dakar and key[1] < num_trajs_dakar:
			if key[0]==key[1]:
				self.arr_kmerge_cost_random[key[0]][key[1]]=math.inf
			else:
				self.arr_kmerge_cost_random[key[0]][key[1]]=self.kmerge_cost_senegal_random[key][0]

# 		print(self.arr_kmerge_cost_random, (count1/count2)*100, " percent of k-merges have zero cost!", count1, count2)
	
	def comdist(self, p1, p2):
		return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)/1000
	
	def gtAssignments(self):
		self.row_ind, self.col_ind = linear_sum_assignment(self.arr_kmerge_cost)
		self.row_ind_random, self.col_ind_random = linear_sum_assignment(self.arr_kmerge_cost_random)
		
		print(self.col_ind, self.row_ind, self.col_ind_random[:100], self.row_ind_random[:100])
		print(self.arr_kmerge_cost[self.row_ind, self.col_ind].sum())

		print(self.arr_kmerge_cost_random[self.row_ind_random, self.col_ind_random].sum())

		self.gt_knn=self.arr_kmerge_cost[self.row_ind, self.col_ind].sum()
		self.gt_knn_random=self.arr_kmerge_cost_random[self.row_ind_random, self.col_ind_random].sum()
		
		dcom=[]
		dcom_rand=[]

		for i in range(len(self.col_ind)):
			dcom.append(self.comdist(self.rcm_array[self.trajs_dakar[self.row_ind[i]]], self.rcm_array[self.trajs_dakar[self.col_ind[i]]]))

			dcom_rand.append(self.comdist(self.rcm_array[self.trajs_senegal_random[self.row_ind_random[i]]], self.rcm_array[self.trajs_senegal_random[self.col_ind_random[i]]]))

		self.plot_dist(dcom, "Dist. between CM(kms) in comm.", "dcom")
		self.plot_dist(dcom_rand, "Dist. between CM(kms) random", "dcom_rand")
	
	def knn_inclusiveness(self, nn_dakar):
		inclusiveness=0
		# print(dk_trajs_reverse)
		# need a reverse dictionary

		for i, tr in enumerate(self.row_ind):
			if self.col_ind[i] in nn_dakar[i]:
		# 		print(nn_dakar[i])
				inclusiveness = inclusiveness + 1

		print("The inclusiveness is: ", (inclusiveness/len(self.row_ind))*100, " percent with: ", inclusiveness, " trajs. inside the nearest neighbours.")
		return (inclusiveness/len(self.row_ind))*100
	
	def knn_inclusivenss_random(self, nn_random):
		inclusiveness=0
		# print(dk_trajs_reverse)
		# need a reverse dictionary

		for i, tr in enumerate(self.row_ind_random):
			if self.col_ind_random[i] in nn_random[i]:
		# 		print(nn_dakar[i])
				inclusiveness = inclusiveness + 1

		print("The inclusiveness(random trajs.) is: ", (inclusiveness/len(self.row_ind_random))*100, " with: ", inclusiveness, " trajs. inside the nearest neighbours.")
		return (inclusiveness/len(self.row_ind_random))*100 
	
	def get_knn_cost(self, K, idex):
		rows, cols = self.knn_m[idex].nonzero()
	# 	print(max(cols))

		arr_kmerge_cost_knn = np.full( (self.num_trajs_dakar, self.num_trajs_dakar), math.inf ) #(num_trajs_dakar, K)
		nn_dakar=defaultdict(list)

		for j in range(self.num_trajs_dakar):
		# 	print(key)
			index0=j*K
			traj=j
			for i in range(K):
	# 			print((traj, cols[index0+i])) #dk_trajs_dict[cols[index0+i]]
	# 			if len(kmerge_cost_dakar[(traj, cols[index0+i])]):
				if traj==cols[index0+i]:
					arr_kmerge_cost_knn[traj][cols[index0+i]]=math.inf
				else:
					arr_kmerge_cost_knn[traj][cols[index0+i]]=self.kmerge_cost_dakar[(traj, cols[index0+i])][0]
	# 			else:
	# # 				arr_kmerge_cost_knn[dk_trajs_dict[traj]][dk_trajs_dict[cols[index0+i]]]=math.inf #np.mean(arr_kmerge_cost_knn[dk_trajs_dict[traj]])
	# 				print("Non-existing edge!")
				nn_dakar[traj].append(cols[index0+i])

	# 	print(arr_kmerge_cost_knn)
		return arr_kmerge_cost_knn, nn_dakar
	
	def get_knn_cost_random(self, K, idex):
		rows, cols = self.knn_m_random[idex].nonzero()

		arr_kmerge_cost_knn_random = np.full( (self.num_trajs_dakar, self.num_trajs_dakar), math.inf )
		nn_random=defaultdict(list)

		for j in range(self.num_trajs_dakar):
		# 	print(key)
			traj=j
			index0=j*K
			for i in range(K):
		# 		print(traj, index0+i, cols[index0+i]) #dk_trajs_dict[cols[index0+i]]
	# 			if len(kmerge_cost_senegal_random[(traj, cols[index0+i])]):
				if traj==cols[index0+i]:
					arr_kmerge_cost_knn_random[traj][cols[index0+i]]=math.inf
				else:
					arr_kmerge_cost_knn_random[traj][cols[index0+i]]=self.kmerge_cost_senegal_random[(traj, cols[index0+i])][0]
	# 			else:
	# 				arr_kmerge_cost_knn_random[dk_trajs_dict_random[traj]][dk_trajs_dict_random[cols[index0+i]]]=math.inf #np.mean(arr_kmerge_cost_knn_random[dk_trajs_dict_random[traj]])
		# 			print("Non-existing edge!")
				nn_random[traj].append(cols[index0+i])

	# 	print(arr_kmerge_cost_knn_random)
		return arr_kmerge_cost_knn_random, nn_random
	
	def lassignment(self, cost_m, K, strategy):
		count=0
		cost=0
		ca=cost_m

		if strategy == 1:
			while(count<self.num_trajs_dakar):
				ri, ci = linear_sum_assignment(ca) #, maximize=True
				this_cost=ca[ri, ci].sum()
		# 		print(this_cost)

				ca = np.delete(ca, ri, axis=0)

				count=count+K
				cost=cost+this_cost

		elif strategy == 2:
			num_it = int(self.num_trajs_dakar/K)
			for i in range(num_it):
				ri, ci = linear_sum_assignment(ca[i*K : (i+1)*K]) #, maximize=True
				this_cost=ca[i*K : (i+1)*K][ri, ci].sum()
		# 		print(this_cost)

				cost=cost+this_cost

		elif strategy == 3:
			num_it = int(self.num_trajs_dakar/K)
			flags = [1 for i in range(self.num_trajs_dakar)]
			idx_random = []

			for i in range(num_it):
				this_rand=set()
				while len(this_rand)<K:
					idx = random.randint(0, self.num_trajs_dakar-1)
					if flags[idx]==1:
						this_rand.add(idx)
						flags[idx]=0
				idx_random.append(list(this_rand))

			for i in range(num_it):
				ri, ci = linear_sum_assignment(ca[idx_random[i]]) #, maximize=True
				this_cost=ca[idx_random[i]][ri, ci].sum()

				cost=cost+this_cost

		elif strategy == 4:
			ri, ci = linear_sum_assignment(ca) #, maximize=True
			cost=ca[ri, ci].sum()
		return cost, (ri, ci)
	
	def analyseOverallKNN(self):
		strategy=4

		self.cost_knn=[]
		self.cost_knn_random=[]
		self.inclusiveness_knn=[]
		self.inclusiveness_knn_random=[]
		self.idss_array=[]
		self.idss_array_rand=[]

		for idex, K_SIZE in enumerate(self.ck_vals):
			aknn, nn_dakar=self.get_knn_cost(K_SIZE, idex)
			aknn_random, nn_random=self.get_knn_cost_random(K_SIZE, idex)

			min_vals1 = [math.inf for i in range(self.num_trajs_dakar)]
			min_vals2 = [math.inf for i in range(self.num_trajs_dakar)]

			for row in aknn:
				for i, val in enumerate(row):
					if val < min_vals1[i]:
						min_vals1[i]=val

			for row in aknn_random:
				for i, val in enumerate(row):
					if val < min_vals2[i]:
						min_vals2[i]=val

			count1, count2=0,0
			inf_cols1, inf_cols2=[],[]
			for i, val in enumerate(min_vals1):
				if val == math.inf:
					count1=count1+1
					inf_cols1.append(i)
			for i, val in enumerate(min_vals2):
				if val == math.inf:
					count2=count2+1
					inf_cols2.append(i)

			print(count1, " and ", count2, " number of cols. have just inf. values in community and random respectively!")
			if count1>0:
				aknn = np.delete(aknn, inf_cols1, axis=0)
				aknn = np.delete(aknn, inf_cols1, axis=1)
			if count2>0:
				aknn_random = np.delete(aknn_random, inf_cols2, axis=0)
				aknn_random = np.delete(aknn_random, inf_cols2, axis=1)

		# 	print(aknn.shape, aknn_random.shape)
			thiss_cost1, idss1 = self.lassignment(aknn, K_SIZE, strategy)
			self.cost_knn.append(thiss_cost1-self.gt_knn)
			self.idss_array_rand.append(idss1)
			try:
				thiss_cost, idss = self.lassignment(aknn_random, K_SIZE, strategy)
				self.idss_array.append(idss)
				self.cost_knn_random.append(thiss_cost-self.gt_knn_random)
			except:
				self.cost_knn_random.append(math.inf)

			self.inclusiveness_knn.append(self.knn_inclusiveness(nn_dakar))
			self.inclusiveness_knn_random.append(self.knn_inclusivenss_random(nn_random))
	
	def plotOverallKNN(self):
		# ck_vals = ck_vals[:-2]
		# cost_knn = cost_knn[:-2]
		# cost_knn_random = cost_knn_random[:-2]

		fig, ax = plt.subplots(5, 1, figsize=(10, 10))

		ax[0].plot(self.ck_vals, [(cst+self.gt_knn) for cst in self.cost_knn], 'g-', label="community(cost)")
		ax[1].plot(self.ck_vals, [(cst+self.gt_knn_random) for cst in self.cost_knn_random], 'r-', label="random(cost)")
		ax[2].plot(self.ck_vals, [(cst+self.gt_knn)/self.gt_knn for cst in self.cost_knn], 'g-', label="community(cost)")
		ax[3].plot(self.ck_vals, [(cst+self.gt_knn_random)/self.gt_knn_random for cst in self.cost_knn_random], 'r-', label="random(cost)")
		ax[4].plot(self.ck_vals, [(cst+self.gt_knn_random)/self.gt_knn for cst in self.cost_knn_random], 'r-', label="random(cost)")


		ax2 = ax[0].twinx()
		ax2.plot(self.ck_vals, self.inclusiveness_knn, 'b-', label="community")
		ax2.set_ylabel('Inclusiveness(%)', color='b')#, color='b'
		ax2.set_ylim([0, 100])

		ax2 = ax[1].twinx()
		ax2.plot(self.ck_vals, self.inclusiveness_knn_random, 'y-', label="random")
		ax2.set_ylabel('Inclusiveness(%)', color='y')#, color='b'
		ax2.set_ylim([0, 100])

		ax2 = ax[2].twinx()
		ax2.plot(self.ck_vals, self.inclusiveness_knn, 'b-', label="community")
		ax2.set_ylabel('Inclusiveness(%)', color='b')#, color='b'
		ax2.set_ylim([0, 100])

		ax2 = ax[3].twinx()
		ax2.plot(self.ck_vals, self.inclusiveness_knn_random, 'y-', label="random")
		ax2.set_ylim([0, 100])

		ax[2].set_xlabel('K (Num. of nearest neighbours)')
		ax[0].set_ylabel('Total merge cost', color='g')#, color='g'
		ax[1].set_ylabel('Total merge cost', color='r')#, color='g'
		ax[2].set_ylabel('Frac. opt. cost', color='g')#, color='g'
		ax[3].set_ylabel('Frac. opt. cost', color='r')#, color='g'
		ax[4].set_ylabel('Frac. opt. cost', color='r')#, color='g'
		ax2.set_ylabel('Inclusiveness(%)', color='y')#, color='b'

		ax2 = ax[4].twinx()
		ax2.plot(self.ck_vals, self.inclusiveness_knn_random, 'y-', label="random")
		ax2.set_ylabel('Inclusiveness(%)', color='y')#, color='b'
		ax2.set_ylim([0, 100])

		# ax[2].set_ylim([1, 2])
		# ax[3].set_ylim([1, 2])
		ax[4].set_ylim([1, 15])

		# ax[0].set_ylim([1, 2])
		# ax[1].set_ylim([1, 2])

		# plt.xlabel("Num. of nearest neighbours")
		# plt.ylabel("Time to construct KNN (s)")

		ax[0].set_title("Community trajs.")
		ax[1].set_title("Random trajs.")
		ax[2].set_title("Community trajs.")
		ax[3].set_title("Random trajs.")
		ax[4].set_title("Random trajs.(wrt. community optimal cost)")

	# 	plt.savefig("delta_cost_inclusiveness" + str(strategy) + ".pdf")
		print([(cst+self.gt_knn)/self.gt_knn for cst in self.cost_knn], [(cst+self.gt_knn_random)/self.gt_knn_random for cst in self.cost_knn_random])
	
	def kLawCost(self):
		self.frac_cost_size_shenzen=defaultdict(list)
		self.frac_cost_density_shenzen=defaultdict(list)

		self.gt_cost_size_shenzen=defaultdict(list)
		self.gt_cost_density_shenzen=defaultdict(list)

		self.total_merge_cost_size_shenzen=defaultdict(list)
		self.total_merge_cost_density_shenzen=defaultdict(list)

		self.knn_m_size_shenzen=defaultdict(list)
		self.knn_m_density_shenzen=defaultdict(list)
		
		self.files_density_shenzen = ["com_trajectories_shenzen_density12.0.pickle", "com_trajectories_shenzen_density18.0.pickle", "com_trajectories_shenzen_density23.0.pickle", "com_trajectories_shenzen_density27.0.pickle"] 

		self.files_size_shenzen = ["com_trajectories_shenzen_size18.pickle", "com_trajectories_shenzen_size19.pickle", "com_trajectories_shenzen_size5.pickle", "com_trajectories_shenzen_size9.pickle"]
		
		self.ck_vals=[10, 20, 30, 40, 50, 100, 250, 500, 1000, 2000] #2, 5, 

		for file in self.files_size_shenzen:
			for K_SIZE in tqdm(self.ck_vals):
				try:
					with open('size_knn_shenzen/knn_matrix_shenzen_' + file.split(".")[0].split("_")[3] + "_" + str(K_SIZE) +  '.pickle', 'rb') as handle:
						self.knn_m_size_shenzen[file.split(".")[0].split("_")[3]].append(pickle.load(handle))
				except:
					print("The KNN size does not exist!")

		for file in self.files_density_shenzen:
			for K_SIZE in tqdm(self.ck_vals):
				try:
					with open('density_knn_shenzen/knn_matrix_shenzen_' + file.split(".")[0].split("_")[3] + "_" + str(K_SIZE) +  '.pickle', 'rb') as handle:
						self.knn_m_density_shenzen[file.split(".")[0].split("_")[3]].append(pickle.load(handle))
				except:
					print("The KNN size does not exist!")
		
		print(self.knn_m_density_shenzen.keys(), self.knn_m_size_shenzen.keys())
		
		self.trajs_size_shenzen=defaultdict(list)
		self.trajs_density_shenzen=defaultdict(list)
		
		self.gt_indices_size_shenzen = defaultdict(tuple)
		self.gt_indices_density_shenzen = defaultdict(tuple)

		for file in tqdm(self.files_size_shenzen):
			case=file.split(".")[0].split("_")[3]
			with open(file, 'rb') as handle:
				trajs_this = pickle.load(handle)

			with open('kmerge_costs_shenzen_' + case + '.pickle', 'rb') as handle:
				this_cost = pickle.load(handle)

			num_trajs = len(trajs_this)
			arr_kmerge_cost = np.zeros( (num_trajs, num_trajs) )

			for key in this_cost.keys():
				if key[0]==key[1]:
					arr_kmerge_cost[key[0]][key[1]]=math.inf
				else:
					arr_kmerge_cost[key[0]][key[1]]=this_cost[key][0]

			row_ind, col_ind = linear_sum_assignment(arr_kmerge_cost)
			self.gt_indices_size_shenzen[case]=(row_ind, col_ind)
			self.gt_cost_size_shenzen[case]=arr_kmerge_cost[row_ind, col_ind].sum()

		for file in tqdm(self.files_density_shenzen):
			case=file.split(".")[0].split("_")[3]
			with open(file, 'rb') as handle:
				trajs_this = pickle.load(handle)

			with open('kmerge_costs_shenzen_' + case + '.pickle', 'rb') as handle:
				this_cost = pickle.load(handle)

			num_trajs = len(trajs_this)
			arr_kmerge_cost = np.zeros( (num_trajs, num_trajs) )

			for key in this_cost.keys():
				if key[0]==key[1]:
					arr_kmerge_cost[key[0]][key[1]]=math.inf
				else:
					arr_kmerge_cost[key[0]][key[1]]=this_cost[key][0]

			row_ind, col_ind = linear_sum_assignment(arr_kmerge_cost)
			self.gt_indices_density_shenzen[case]=(row_ind, col_ind)
			self.gt_cost_density_shenzen[case]=arr_kmerge_cost[row_ind, col_ind].sum()
	
	def get_knn_cost_specific(self, K, knn_m, num_trajs, this_cost):
		rows, cols = knn_m.nonzero()
	# 	print(len(rows), len(cols))

		arr_kmerge_cost_knn = np.full( (num_trajs, num_trajs), math.inf ) #(num_trajs_dakar, K)
		nn_this=defaultdict(list)

		for j in range(num_trajs):
		# 	print(key)
			index0=j*K
			traj=j
			for i in range(K):
	# 			print((traj, cols[index0+i])) #dk_trajs_dict[cols[index0+i]]
	# 			if len(kmerge_cost_dakar[(traj, cols[index0+i])]):
				if traj==cols[index0+i]:
					arr_kmerge_cost_knn[traj][cols[index0+i]]=math.inf
				else:
					arr_kmerge_cost_knn[traj][cols[index0+i]]=this_cost[(traj, cols[index0+i])][0]
	# 			else:
	# # 				arr_kmerge_cost_knn[dk_trajs_dict[traj]][dk_trajs_dict[cols[index0+i]]]=math.inf #np.mean(arr_kmerge_cost_knn[dk_trajs_dict[traj]])
	# 				print("Non-existing edge!")
				nn_this[traj].append(cols[index0+i])

	# 	print(arr_kmerge_cost_knn)
		return arr_kmerge_cost_knn, nn_this
	
	def lassignment_specific(self, cost_m, K):
		ca=cost_m

		ri, ci = linear_sum_assignment(ca) #, maximize=True
		cost=ca[ri, ci].sum()
		return cost, (ri, ci)
	
	def knn_inclusiveness_specific(self, nn_this, row_ind, col_ind):
		inclusiveness=0

		for i, tr in enumerate(row_ind):
			if col_ind[i] in nn_this[i]:
		# 		print(nn_dakar[i])
				inclusiveness = inclusiveness + 1

		print("The inclusiveness is: ", (inclusiveness/len(row_ind))*100, " percent with: ", inclusiveness, " trajs. inside the nearest neighbours.")
		return (inclusiveness/len(row_ind))*100
	
	def analyseShenzenKNNSize(self):
		self.inclusiveness_size_shenzen=defaultdict(list)
		self.inclusiveness_density_shenzen=defaultdict(list)
		
		self.ck_vals=[10, 20, 30, 40, 50, 100, 250, 500, 1000, 2000] #2, 5,

		for file in tqdm(self.files_size_shenzen):
			case=file.split(".")[0].split("_")[3]
			with open(file, 'rb') as handle:
				trajs_this = pickle.load(handle)

			with open('kmerge_costs_shenzen_' + case + '.pickle', 'rb') as handle:
				this_cost = pickle.load(handle)

			num_trajs = len(trajs_this)

			for idex, K_SIZE in enumerate(self.ck_vals):
				if num_trajs > K_SIZE:
					aknn, nn_this=self.get_knn_cost_specific(K_SIZE, self.knn_m_size_shenzen[case][idex], num_trajs, this_cost)

					min_vals1 = [math.inf for i in range(num_trajs)]

					for row in aknn:
						for i, val in enumerate(row):
							if val < min_vals1[i]:
								min_vals1[i]=val

					count1 = 0
					inf_cols1 = []
					for i, val in enumerate(min_vals1):
						if val == math.inf:
							count1=count1+1
							inf_cols1.append(i)

					print(count1, " number of cols. have just inf. values in community!")
					if count1>0:
						aknn = np.delete(aknn, inf_cols1, axis=0)
						aknn = np.delete(aknn, inf_cols1, axis=1)

				# 	print(aknn.shape, aknn_random.shape)
					thiss_cost1, idss1 = self.lassignment_specific(aknn, K_SIZE)
					self.total_merge_cost_size_shenzen[case].append(thiss_cost1-self.gt_cost_size_shenzen[case])

					self.inclusiveness_size_shenzen[case].append(self.knn_inclusiveness_specific(nn_this, self.gt_indices_size_shenzen[case][0], self.gt_indices_size_shenzen[case][1]))
	
	def analyseShenzenKNNDensity(self):
		for file in tqdm(self.files_density_shenzen):
			case=file.split(".")[0].split("_")[3]
			with open(file, 'rb') as handle:
				trajs_this = pickle.load(handle)

			with open('kmerge_costs_shenzen_' + case + '.pickle', 'rb') as handle:
				this_cost = pickle.load(handle)

			num_trajs = len(trajs_this)

			for idex, K_SIZE in enumerate(self.ck_vals):
				if num_trajs > K_SIZE:
					aknn, nn_this=self.get_knn_cost_specific(K_SIZE, self.knn_m_density_shenzen[case][idex], num_trajs, this_cost)

					min_vals1 = [math.inf for i in range(num_trajs)]

					for row in aknn:
						for i, val in enumerate(row):
							if val < min_vals1[i]:
								min_vals1[i]=val

					count1 = 0
					inf_cols1 = []
					for i, val in enumerate(min_vals1):
						if val == math.inf:
							count1=count1+1
							inf_cols1.append(i)

					print(count1, " number of cols. have just inf. values in community!")
					if count1>0:
						aknn = np.delete(aknn, inf_cols1, axis=0)
						aknn = np.delete(aknn, inf_cols1, axis=1)

				# 	print(aknn.shape, aknn_random.shape)
					thiss_cost1, idss1 = self.lassignment_specific(aknn, K_SIZE)
					self.total_merge_cost_density_shenzen[case].append(thiss_cost1-self.gt_cost_density_shenzen[case])

					self.inclusiveness_density_shenzen[case].append(self.knn_inclusiveness_specific(nn_this, self.gt_indices_density_shenzen[case][0], self.gt_indices_density_shenzen[case][1]))
		
		print(self.total_merge_cost_density_shenzen, self.total_merge_cost_size_shenzen)

		print(self.gt_cost_size_shenzen, self.gt_cost_density_shenzen)

		# print(gt_indices_size_shenzen, gt_indices_density_shenzen)
	
	def plotShenzenCosts(self):
		# ck_vals = ck_vals[:-2]
		# cost_knn = cost_knn[:-2]
		# cost_knn_random = cost_knn_random[:-2]

		fig, ax = plt.subplots(2, 1, figsize=(10, 10))

		j=0
		ax2 = ax[0].twinx()
		for idd in self.com_id_size_shenzen:
			val1 = []
			val2 = []
			count=0
			for i in range(len(self.ck_vals)):
				try:
					val1.append( (self.total_merge_cost_size_shenzen['size'+str(int(idd))][i] + self.gt_cost_size_shenzen['size'+str(int(idd))])/self.gt_cost_size_shenzen['size'+str(int(idd))] )
					val2.append( self.inclusiveness_size_shenzen['size'+str(int(idd))][i] )
					count=i
				except:
					pass
			ax[0].plot(self.ck_vals[ : count+1], val1, label=self.com_size[j])
			ax2.plot(self.ck_vals[ : count+1], val2, label=self.com_size[j])
			j=j+1

		j=0
		ax3 = ax[1].twinx()
		for idd in self.com_id_density_shenzen:
			val1 = []
			val2 = []
			count=0
			for i in range(len(self.ck_vals)):
				try:
					val1.append( (self.total_merge_cost_density_shenzen['density'+str(int(idd))][i] + self.gt_cost_density_shenzen['density'+str(int(idd))])/self.gt_cost_density_shenzen['density'+str(int(idd))] )
					val2.append( self.inclusiveness_density_shenzen['density'+str(int(idd))][i] )
					count=i
				except:
					pass
			if count!=0:
				ax[1].plot(self.ck_vals[ : count+1], val1, label=self.com_density_shenzen[j])
				ax3.plot(self.ck_vals[ : count+1], val2, label=self.com_density_shenzen[j])

				j=j+1

		ax2.set_ylabel('Inclusiveness(%)', color='b')#, color='b'
		ax3.set_ylabel('Inclusiveness(%)', color='y')#, color='b'

		ax2.set_ylim([0, 100])
		ax3.set_ylim([0, 100])

		ax[1].set_xlabel('K (Num. of nearest neighbours)')
		ax[0].set_ylabel('Frac. opt. cost', color='g')#, color='g'
		ax[1].set_ylabel('Frac. opt. cost', color='r')#, color='g'

		ax[0].set_ylim([1, 1.5])
		ax[1].set_ylim([1, 1.5])

		# plt.xlabel("Num. of nearest neighbours")
		# plt.ylabel("Time to construct KNN (s)")

		ax[0].legend(title="Community size", bbox_to_anchor=(1.30, 1.05),
				  ncol=1, fancybox=True, shadow=True)
		ax[1].legend(title="Community's density", loc="best", bbox_to_anchor=(1.30, 1.05),
				  ncol=1, fancybox=True, shadow=True)

		ax[0].set_title("Size variation")
		ax[1].set_title("Density variation")

# 		plt.savefig("delta_cost_shenzen.pdf")

		self.ck_vals=[500, 1000, 2000]

		plt.figure(figsize=(10, 6))
		val1 = defaultdict(list)
		for idd in self.com_id_size_shenzen:
			for i in range(len(self.ck_vals)):
				v = (self.total_merge_cost_size_shenzen['size'+str(int(idd))][i-3] + self.gt_cost_size_shenzen['size'+str(int(idd))])/self.gt_cost_size_shenzen['size'+str(int(idd))]
				val1[self.ck_vals[i]].append( v )
		df = pd.DataFrame.from_dict(val1)
		df["Community size"]=self.com_size
		df = df.melt(id_vars="Community size")
		df.columns=["Community size", "K (Num. of nearest neighbours)", "Frac. opt. cost"]

		g = sns.catplot( data=df, kind="bar", x="K (Num. of nearest neighbours)", y="Frac. opt. cost", hue="Community size", palette="dark", alpha=.6, height=6, legend_out=False)
		plt.ylim([1.0, 1.1])
# 		plt.savefig("size_kvars_shenzen.pdf")

		plt.figure(figsize=(10, 6))
		val1 = defaultdict(list)
		for idd in self.com_id_density_shenzen:
			for i in range(len(self.ck_vals)):
				v = (self.total_merge_cost_density_shenzen['density'+str(int(idd))][i-3] + self.gt_cost_density_shenzen['density'+str(int(idd))])/self.gt_cost_density_shenzen['density'+str(int(idd))]
				val1[self.ck_vals[i]].append( v )

		df = pd.DataFrame.from_dict(val1)
		df["Community's density"]=self.com_density_shenzen
		df = df.melt(id_vars="Community's density")
		df.columns=["Community's density", "K (Num. of nearest neighbours)", "Frac. opt. cost"]

		g = sns.catplot( data=df, kind="bar", x="K (Num. of nearest neighbours)", y="Frac. opt. cost", hue="Community's density", palette="dark", alpha=.6, height=6, legend_out=False)
		plt.ylim([1.0, 1.1])
# 		plt.savefig("density_kvars_shenzen.pdf")

		self.ck_vals=[10, 20, 30, 40, 50, 100, 250, 500, 1000, 2000]

		fig, ax = plt.subplots(2, 1, figsize=(10, 10))

		j=0
		val1=defaultdict(list)
		for idd in self.com_id_size_shenzen:
			count=0
			for i in range(len(self.ck_vals)):
				try:
					val1[self.ck_vals[i]].append( (self.total_merge_cost_size_shenzen['size'+str(int(idd))][i] + self.gt_cost_size_shenzen['size'+str(int(idd))])/self.gt_cost_size_shenzen['size'+str(int(idd))] )
					count=i
				except:
					pass
			j=j+1
		for key in val1.keys():
			ax[0].plot(self.com_size[len(self.com_size) - len(val1[key]): ], val1[key], label=key, marker="x")

		j=0
		val1=defaultdict(list)
		for idd in self.com_id_density_shenzen:
			count=0
			for i in range(len(self.ck_vals)):
				try:
					val1[self.ck_vals[i]].append( (self.total_merge_cost_density_shenzen['density'+str(int(idd))][i] + self.gt_cost_density_shenzen['density'+str(int(idd))])/self.gt_cost_density_shenzen['density'+str(int(idd))] )
					count=i
				except:
					pass
			if count!=0:
				j=j+1
		for key in val1.keys():
			ax[1].plot([math.log10(cm) for cm in self.com_density_shenzen], val1[key], label=key, marker="x")

		ax[0].set_xlabel("Community size")#'K (Num. of nearest neighbours)'
		ax[1].set_xlabel("Community's density (log10)")

		ax[0].set_ylabel('Frac. opt. cost', color='g')#, color='g'
		ax[1].set_ylabel('Frac. opt. cost', color='r')#, color='g'

		# ax[0].set_ylim([1, 2.0])
		# ax[1].set_ylim([1, 2.0])

		# plt.xlabel("Num. of nearest neighbours")
		# plt.ylabel("Time to construct KNN (s)")

		ax[0].legend(title="K (Num. of nearest neighbours)")
		ax[1].legend(title="K (Num. of nearest neighbours)")

		ax[0].set_title("Size variation")
		ax[1].set_title("Density variation")

# 		plt.savefig("delta_cost_shenzen_size.pdf")
	
	def kLawCostSenegal(self):
		self.frac_cost_size=defaultdict(list)
		self.frac_cost_density=defaultdict(list)

		self.gt_cost_size=defaultdict(list)
		self.gt_cost_density=defaultdict(list)

		self.total_merge_cost_size=defaultdict(list)
		self.total_merge_cost_density=defaultdict(list)

		self.knn_m_size=defaultdict(list)
		self.knn_m_density=defaultdict(list)
		
		self.files_size = ["com_trajectories_size10.pickle", "com_trajectories_size37.pickle", "com_trajectories_size54.pickle", "com_trajectories_size67.pickle"]

		self.files_density = ["com_trajectories_density39.0.pickle", "com_trajectories_density57.0.pickle", "com_trajectories_density65.0.pickle", "com_trajectories_density7.0.pickle"]
		
		self.ck_vals=[10, 20, 30, 40, 50, 100, 250, 500, 1000, 2000] #2, 5, 

		for file in self.files_size:
			for K_SIZE in tqdm(self.ck_vals):
				try:
					with open('size_knn/knn_matrix_' + file.split(".")[0].split("_")[2] + "_" + str(K_SIZE) +  '.pickle', 'rb') as handle:
						self.knn_m_size[file.split(".")[0].split("_")[2]].append(pickle.load(handle))
				except:
					print("The KNN size does not exist!")

		for file in self.files_density:
			for K_SIZE in tqdm(self.ck_vals):
				try:
					with open('density_knn/knn_matrix_' + file.split(".")[0].split("_")[2] + "_" + str(K_SIZE) +  '.pickle', 'rb') as handle:
						self.knn_m_density[file.split(".")[0].split("_")[2]].append(pickle.load(handle))
				except:
					print("The KNN size does not exist!")
		
		print(self.knn_m_density.keys(), self.knn_m_size.keys())
		
		self.trajs_size=defaultdict(list)
		self.trajs_density=defaultdict(list)
		
		self.gt_indices_size = defaultdict(tuple)
		self.gt_indices_density = defaultdict(tuple)

		for file in tqdm(self.files_size):
			case=file.split(".")[0].split("_")[2]
			with open(file, 'rb') as handle:
				trajs_this = pickle.load(handle)

			with open('kmerge_costs_' + case + '.pickle', 'rb') as handle:
				this_cost = pickle.load(handle)

			num_trajs = len(trajs_this)
			arr_kmerge_cost = np.zeros( (num_trajs, num_trajs) )

			for key in this_cost.keys():
				if key[0]==key[1]:
					arr_kmerge_cost[key[0]][key[1]]=math.inf
				else:
					arr_kmerge_cost[key[0]][key[1]]=this_cost[key][0]

			row_ind, col_ind = linear_sum_assignment(arr_kmerge_cost)
			self.gt_indices_size[case]=(row_ind, col_ind)
			self.gt_cost_size[case]=arr_kmerge_cost[row_ind, col_ind].sum()

		for file in tqdm(self.files_density):
			case=file.split(".")[0].split("_")[2]
			with open(file, 'rb') as handle:
				trajs_this = pickle.load(handle)

			with open('kmerge_costs_' + case + '.pickle', 'rb') as handle:
				this_cost = pickle.load(handle)

			num_trajs = len(trajs_this)
			arr_kmerge_cost = np.zeros( (num_trajs, num_trajs) )

			for key in this_cost.keys():
				if key[0]==key[1]:
					arr_kmerge_cost[key[0]][key[1]]=math.inf
				else:
					arr_kmerge_cost[key[0]][key[1]]=this_cost[key][0]

			row_ind, col_ind = linear_sum_assignment(arr_kmerge_cost)
			self.gt_indices_density[case]=(row_ind, col_ind)
			self.gt_cost_density[case]=arr_kmerge_cost[row_ind, col_ind].sum()
		
		self.inclusiveness_size=defaultdict(list)
		self.inclusiveness_density=defaultdict(list)
	
	def analyseSenegalKNNSize(self):
		self.ck_vals=[10, 20, 30, 40, 50, 100, 250, 500, 1000, 2000] #2, 5,

		for file in tqdm(self.files_size):
			case=file.split(".")[0].split("_")[2]
			with open(file, 'rb') as handle:
				trajs_this = pickle.load(handle)

			with open('kmerge_costs_' + case + '.pickle', 'rb') as handle:
				this_cost = pickle.load(handle)

			num_trajs = len(trajs_this)
		# 	print(num_trajs, case)

			for idex, K_SIZE in enumerate(self.ck_vals):
				if num_trajs > K_SIZE:
					try:
						aknn, nn_this=self.get_knn_cost_specific(K_SIZE, self.knn_m_size[case][idex], num_trajs, this_cost)

						min_vals1 = [math.inf for i in range(num_trajs)]

						for row in aknn:
							for i, val in enumerate(row):
								if val < min_vals1[i]:
									min_vals1[i]=val

						count1 = 0
						inf_cols1 = []
						for i, val in enumerate(min_vals1):
							if val == math.inf:
								count1=count1+1
								inf_cols1.append(i)

						print(count1, " number of cols. have just inf. values in community!")
						if count1>0:
							aknn = np.delete(aknn, inf_cols1, axis=0)
							aknn = np.delete(aknn, inf_cols1, axis=1)

					# 	print(aknn.shape, aknn_random.shape)
						thiss_cost1, idss1 = self.lassignment_specific(aknn, K_SIZE)
						self.total_merge_cost_size[case].append(thiss_cost1-self.gt_cost_size[case])

						self.inclusiveness_size[case].append(self.knn_inclusiveness_specific(nn_this, self.gt_indices_size[case][0], self.gt_indices_size[case][1]))
					except:
						print("cost matrix is infeasible")

						self.total_merge_cost_size[case].append(-1)
						self.inclusiveness_size[case].append(-1)
	
	def analyseSenegalKNNDensity(self):
		for file in tqdm(self.files_density):
			case=file.split(".")[0].split("_")[2]
			with open(file, 'rb') as handle:
				trajs_this = pickle.load(handle)

			with open('kmerge_costs_' + case + '.pickle', 'rb') as handle:
				this_cost = pickle.load(handle)

			num_trajs = len(trajs_this)
			print(case)

			for idex, K_SIZE in enumerate(self.ck_vals):
				if num_trajs > K_SIZE:
					try:
						aknn, nn_this=self.get_knn_cost_specific(K_SIZE, self.knn_m_density[case][idex], num_trajs, this_cost)

						min_vals1 = [math.inf for i in range(num_trajs)]

						for row in aknn:
							for i, val in enumerate(row):
								if val < min_vals1[i]:
									min_vals1[i]=val

						count1 = 0
						inf_cols1 = []
						for i, val in enumerate(min_vals1):
							if val == math.inf:
								count1=count1+1
								inf_cols1.append(i)

						print(count1, " number of cols. have just inf. values in community!")
						if count1>0:
							aknn = np.delete(aknn, inf_cols1, axis=0)
							aknn = np.delete(aknn, inf_cols1, axis=1)

					# 	print(aknn.shape, aknn_random.shape)
						thiss_cost1, idss1 = self.lassignment_specific(aknn, K_SIZE)
						self.total_merge_cost_density[case].append(thiss_cost1-self.gt_cost_density[case])

						self.inclusiveness_density[case].append(self.knn_inclusiveness_specific(nn_this, self.gt_indices_density[case][0], self.gt_indices_density[case][1]))
					except:
						print("cost matrix is infeasible")

						self.total_merge_cost_density[case].append(-1)
						self.inclusiveness_density[case].append(-1)
		
		print(self.total_merge_cost_density, self.total_merge_cost_size)

		print(self.gt_cost_size, self.gt_cost_density)

		# print(gt_indices_size, gt_indices_density)
	
	def plotSenegalCosts(self):
		# ck_vals = ck_vals[:-2]
		# cost_knn = cost_knn[:-2]
		# cost_knn_random = cost_knn_random[:-2]

		self.ck_vals=[10, 20, 30, 40, 50, 100, 250, 500, 1000, 2000] #2, 5,

		fig, ax = plt.subplots(2, 1, figsize=(10, 10))

		j=0
		ax2 = ax[0].twinx()
		for idd in self.com_id_size:
			val1 = []
			val2 = []
			count=0
			for i in range(len(self.ck_vals)):
				try:
					v = (self.total_merge_cost_size['size'+str(int(idd))][i] + self.gt_cost_size['size'+str(int(idd))])/self.gt_cost_size['size'+str(int(idd))]
					if self.total_merge_cost_size['size'+str(int(idd))][i]!=-1:
						val1.append( v )
						count=count+1
						val2.append( self.inclusiveness_size['size'+str(int(idd))][i] )
				except:
					pass
			ax[0].plot(self.ck_vals[len(self.ck_vals) - count: ], val1, label=self.com_size[j])
			ax2.plot(self.ck_vals[len(self.ck_vals) - count: ], val2, label=self.com_size[j])
			j=j+1

		j=0
		ax3 = ax[1].twinx()
		for idd in self.com_id_density:
			val1 = []
			val2 = []
			count=0
			for i in range(len(self.ck_vals)):
				try:
					v = (self.total_merge_cost_density['density'+str(int(idd))][i] + self.gt_cost_density['density'+str(int(idd))])/self.gt_cost_density['density'+str(int(idd))]
					if self.total_merge_cost_density['density'+str(int(idd))][i]!=-1:
						val1.append( v )
						count=count+1
						val2.append( self.inclusiveness_density['density'+str(int(idd))][i] )
				except:
					pass
			if count!=0:
				ax[1].plot(self.ck_vals[len(self.ck_vals) - count: ], val1, label=self.com_density[j])
				ax3.plot(self.ck_vals[len(self.ck_vals) - count: ], val2, label=self.com_density[j])
				j=j+1

		ax2.set_ylabel('Inclusiveness(%)', color='b')#, color='b'
		ax3.set_ylabel('Inclusiveness(%)', color='y')#, color='b'

		ax2.set_ylim([0, 100])
		ax3.set_ylim([0, 100])

		ax[1].set_xlabel('K (Num. of nearest neighbours)')
		ax[0].set_ylabel('Frac. opt. cost', color='g')#, color='g'
		ax[1].set_ylabel('Frac. opt. cost', color='r')#, color='g'

		ax[0].set_ylim([1, 1.8])
		ax[1].set_ylim([1, 1.8])

		ax[0].set_xlim([0, 2100])
		ax[1].set_xlim([1, 2100])

		# plt.xlabel("Num. of nearest neighbours")
		# plt.ylabel("Time to construct KNN (s)")

		ax[0].legend(title="Community size")
		ax[1].legend(title="Community's density")

		ax[0].set_title("Size variation")
		ax[1].set_title("Density variation")

# 		plt.savefig("delta_cost.pdf")
		
		self.ck_vals=[500, 1000, 2000]
		
		plt.figure(figsize=(10, 6))
		val1 = defaultdict(list)
		for idd in self.com_id_size:
			for i in range(len(self.ck_vals)):
				v = (self.total_merge_cost_size['size'+str(int(idd))][i-3] + self.gt_cost_size['size'+str(int(idd))])/self.gt_cost_size['size'+str(int(idd))]
				val1[self.ck_vals[i]].append( v )
		df = pd.DataFrame.from_dict(val1)
		df["Community size"]=self.com_size
		df = df.melt(id_vars="Community size")
		df.columns=["Community size", "K (Num. of nearest neighbours)", "Frac. opt. cost"]

		g = sns.catplot( data=df, kind="bar", x="K (Num. of nearest neighbours)", y="Frac. opt. cost", hue="Community size", palette="dark", alpha=.6, height=6, legend_out=False)
		plt.ylim([1.0, 1.3])
# 		plt.savefig("size_kvars.pdf")
		
		plt.figure(figsize=(10, 6))
		val1 = defaultdict(list)
		for idd in self.com_id_density:
			for i in range(len(self.ck_vals)):
				v = (self.total_merge_cost_density['density'+str(int(idd))][i-3] + self.gt_cost_density['density'+str(int(idd))])/self.gt_cost_density['density'+str(int(idd))]
				val1[self.ck_vals[i]].append( v )

		df = pd.DataFrame.from_dict(val1)
		df["Community's density"]=self.com_density
		df = df.melt(id_vars="Community's density")
		df.columns=["Community's density", "K (Num. of nearest neighbours)", "Frac. opt. cost"]

		g = sns.catplot( data=df, kind="bar", x="K (Num. of nearest neighbours)", y="Frac. opt. cost", hue="Community's density", palette="dark", alpha=.6, height=6, legend_out=False)
		plt.ylim([1.0, 1.3])
# 		plt.savefig("density_kvars.pdf")

		self.ck_vals=[10, 20, 30, 40, 50, 100, 250, 500, 1000, 2000]

		fig, ax = plt.subplots(2, 1, figsize=(10, 10))

		j=0

		val1=defaultdict(list)

		for idd in self.com_id_size:
			count=0
			for i in range(len(self.ck_vals[-3:])):
				try:
					val1[self.ck_vals[i-3]].append( (self.total_merge_cost_size['size'+str(int(idd))][i-3] + self.gt_cost_size['size'+str(int(idd))])/self.gt_cost_size['size'+str(int(idd))] )
					count=i
				except:
					pass
			j=j+1
		for key in val1.keys():
			ax[0].plot(self.com_size, val1[key], label=key, marker="x")

		j=0
		ax3 = ax[0].twinx()

		val1=defaultdict(list)

		for idd in self.com_id_density:
			count=0
			for i in range(len(self.ck_vals[-5:])):
				try:
					val1[self.ck_vals[i-5]].append( (self.total_merge_cost_density['density'+str(int(idd))][i-5] + self.gt_cost_density['density'+str(int(idd))])/self.gt_cost_density['density'+str(int(idd))] )
					count=i
				except:
					pass
			if count!=0:
				j=j+1
		for key in val1.keys():
			ax[1].plot([math.log10(cm) for cm in self.com_density], val1[key], label=key, marker="x")

		ax[0].set_xlabel("Community size")#'K (Num. of nearest neighbours)'
		ax[1].set_xlabel("Community's density (log10)")

		ax[0].set_ylabel('Frac. opt. cost', color='g')#, color='g'
		ax[1].set_ylabel('Frac. opt. cost', color='r')#, color='g'

		ax[0].set_ylim([1, 2.0])
		ax[1].set_ylim([1, 2.0])

		# plt.xlabel("Num. of nearest neighbours")
		# plt.ylabel("Time to construct KNN (s)")

		ax[0].legend(title="K (Num. of nearest neighbours)")
		ax[1].legend(title="K (Num. of nearest neighbours)")

		ax[0].set_title("Size variation")
		ax[1].set_title("Density variation")

# 		plt.savefig("delta_cost_size_senegal.pdf")


# In[158]:


# optshow = twoOptimalShowcase()

# optshow.obtainKNN()
# optshow.senegalCommunity()
# optshow.shenzenCommunity()
# optshow.odDict()

# optshow.communitySizes()

# optshow.visualiseCommunities()


# In[159]:


# optshow_final = twoOptimalShowcase()
# # # # optshow_final.senegalCommunity()
# # # # optshow_final.shenzenCommunity()
# # # # # # optshow_final.civCommunity(tg_final3.getGraphCiv())#tg_final3.getGraphCiv()
# # # # # optshow_final.shenzenCommunity()#tg_final.getGraphShenzen()
# # # # # optshow_final.communitySizes()

# # optshow_final.millionCommunity()#tg_final4.getGraphMillion()

# optshow_final.partialCommunity(50)#tg_final6.getGraphPartial(), , tg_final8.getGraphPartial()
# # optshow_final.communitySizes()

# # optshow_final.visualiseCommunities()


# In[160]:


optshow_final = twoOptimalShowcase()
# # optshow_final.senegalCommunity()
# # # optshow_final.shenzenCommunity()
# # # # # optshow_final.civCommunity(tg_final3.getGraphCiv())#tg_final3.getGraphCiv()
# # # # optshow_final.shenzenCommunity()#tg_final.getGraphShenzen()
# # # # optshow_final.communitySizes()

optshow_final.millionCommunity()#tg_final4.getGraphMillion() tg_final9.getGraphMillion()

# optshow_final.weekCommunity(49878, tg_final7.getGraphWeek())#tg_final6.getGraphPartial()
# optshow_final.communitySizes()

# optshow_final.visualiseCommunities()


# ## K-Anonymisation Final

# In[161]:


# trajs_here = [i for i in range(319366)]
# with open('senegal_data_rand_'+str(319366)+'.pickle', 'wb') as handle:
# 	pickle.dump(trajs_here, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[162]:


from scipy.sparse import csr_array
from multiprocessing import Pool, Value
from ctypes import c_int
import sys
import copy

class KAnonFinal(twoOptimalShowcase):
	def __init__(self, scenario):
		super().__init__()
		self.genralised_shenzen=0
		self.genralised_taka=0
		self.genralised_senegal=0
		self.genralised_civ=0
		self.genralised_million=0
		self.genralised_partial=0
		self.genralised_week=0
		
		self.cost_shenzen=0
		self.cost_taka=0
		self.cost_senegal=0
		self.cost_civ=0
		self.cost_million=0
		self.cost_partial=0
		self.cost_week=0
		
		self.num_shenzen_trajs=100000
		self.num_taka_trajs=100000
		self.num_senegal_trajs=319366
		self.num_civ_trajs=49324#49242
		self.num_million_trajs=1008169 #1008109
		self.num_partial_trajs=1000 #319366 #50000
		self.num_week_trajs=49878
		
		self.faulty_civ=[43066, 43067, 43068, 43069, 43070, 43071, 43072, 43073, 43074, 43075, 43076, 43077, 43078, 43079, 43080, 43081, 43082, 43083, 43084, 43085, 43086, 43087, 43088, 43089, 43090, 43091, 43092, 43093, 43094, 43095, 43096, 43097, 43098, 43099, 43100, 43101, 43102, 43103, 43104, 43105, 43106, 43107, 43108, 43109, 43110, 43111, 43112, 43113, 43114, 43115, 43116, 43117, 43118, 43119, 43120, 43121, 43122, 43123, 43124, 43125, 43126, 43127, 43128, 43129, 43130, 43131, 43132, 43133, 43134, 43135, 43136, 43137, 43138, 43139, 43140, 43141, 43142, 43143, 43144, 43145, 43146, 43147]
		self.K=500
		self.scenario=scenario
		self.k=10
	
	def init_pool_processes(self, knn, costs):
		global iknn, icosts
		iknn, icosts = knn, costs
	
	def plot_dist(self, dt, xlab, name, ylab=""):
		fig, axi = plt.subplots(1, 1)

		df = pd.DataFrame(dt)
		df.columns=["size"]

# 		print(df)
		sns.histplot(data=df, bins=[1*i for i in range(10)])

		plt.xlabel(xlab)
		if len(ylab):
			plt.ylabel(ylab)
		#plt.legend()
	# 	if xlab[-6:]=="random":
	# 		plt.xlim([0, 200])
	# 	else:
	# 		plt.xlim([0, 5])
		axi.get_legend().remove()
		plt.savefig(str(name) + '.pdf')
	
	def get_union_neighbours(self, trajs, knn):
		rows, cols = knn.nonzero()
		num_trajs=len(trajs)
		neibs = set()
		
# 		print(trajs)

		if self.scenario=="Partial":
			with open('senegal_data_rand_'+str(self.num_partial_trajs)+'.pickle', 'rb') as handle:
				rand_samples = pickle.load(handle)

			tr=list(rand_samples)
			
			for j in range(num_trajs):
				index0=trajs[j]*self.K
				neibs.add(trajs[j])
				for i in range(self.K):
					neibs.add(cols[index0+i])
		else:
			for j in range(num_trajs):
				index0=trajs[j]*self.K
				neibs.add(trajs[j])
				for i in range(self.K):
					neibs.add(cols[index0+i])

		return list(neibs)
	
	def get_final_knn_cost(self, knn, costs, trajs):
		rows, cols = knn.nonzero()
		num_trajs=len(trajs)
		
		neibs=self.get_union_neighbours(trajs, knn)
# 		print(len(neibs), num_trajs)
# 		print(cols)
		
		arr_kmerge_cost_knn = np.full( (num_trajs, len(neibs)), math.inf ) #(num_trajs_dakar, K)
		
		neighdict = defaultdict(int)
		for i, tr in enumerate(neibs):
			neighdict[tr]=i
		
		with open('senegal_data_rand_'+str(self.num_partial_trajs)+'.pickle', 'rb') as handle:
			rand_samples = pickle.load(handle)
		
		tr=list(rand_samples)

		for j in range(num_trajs):
			traj=j
			index0=trajs[j]*self.K
			for i in range(self.K):
				if trajs[j]==cols[index0+i]:
					arr_kmerge_cost_knn[traj][neighdict[cols[index0+i]]]=math.inf
				else:
# 					print(cols[index0+i], neighdict[cols[index0+i]])
					if self.scenario=="Civ":
						arr_kmerge_cost_knn[traj][neighdict[cols[index0+i]]]=costs[(trajs[j], i)]
					elif self.scenario=="Partial":
# 						print(tr[trajs[j]], tr[cols[index0+i]], costs[(tr[trajs[j]], tr[cols[index0+i]])])
						arr_kmerge_cost_knn[traj][neighdict[cols[index0+i]]]=costs[(trajs[j], cols[index0+i])] #tr[trajs[j]], tr[cols[index0+i]]
					elif self.scenario=="Week":
# 						print(tr[trajs[j]], tr[cols[index0+i]], costs[(tr[trajs[j]], tr[cols[index0+i]])])
						arr_kmerge_cost_knn[traj][neighdict[cols[index0+i]]]=costs[(trajs[j], cols[index0+i])]
					elif self.scenario=="Million":
						arr_kmerge_cost_knn[traj][neighdict[cols[index0+i]]]=costs[trajs[j]*self.K+i]
					else:
						arr_kmerge_cost_knn[traj][neighdict[cols[index0+i]]]=costs[(trajs[j], i)][0] #cols[index0+i]

		return arr_kmerge_cost_knn, neibs
	
	def scmUnique(self, l_nodes):
		global iknn, icosts
		
		assigned_rows, assigned_cols=[],[]
		
		aknn, neibs = self.get_final_knn_cost(iknn, icosts, l_nodes)

		row_ind, col_ind = linear_sum_assignment(aknn)
# 			print(row_ind, col_ind)

		assigned_rows.extend([l_nodes[j] for j in row_ind])
		assigned_cols.extend([neibs[j] for j in col_ind])
		
		return (assigned_rows, assigned_cols)
	
	def sparseCostMatrix(self, comms, knn, costs):
		assigned_rows, assigned_cols=[],[]
		for com in tqdm(set(comms.values())):
			list_nodes = [nodes for nodes in comms.keys() if comms[nodes] == com]
			
# 			print(list_nodes)
			
			if self.scenario=="Million":
				if len(list_nodes)==1:
					assigned_rows.extend(list_nodes)
					assigned_cols.extend(list_nodes)
					continue

			aknn, neibs = self.get_final_knn_cost(knn, costs, list_nodes)

			row_ind, col_ind = linear_sum_assignment(aknn)
# 			print(row_ind, col_ind)
			
			assigned_rows.extend([list_nodes[j] for j in row_ind])
			assigned_cols.extend([neibs[j] for j in col_ind])

# 		list_nodes=[]
# 		for com in tqdm(set(comms.values())):
# 			list_nodes.append( [nodes for nodes in comms.keys() if comms[nodes] == com] )
			
# # # 			if self.scenario=="Million":
# # # 				if len(list_nodes)==1:
# # # 					continue

# # 			aknn, neibs = self.get_final_knn_cost(knn, costs, list_nodes)

# # 			row_ind, col_ind = linear_sum_assignment(aknn)
# # # 			print(row_ind, col_ind)
			
# # 			assigned_rows.extend([list_nodes[j] for j in row_ind])
# # 			assigned_cols.extend([neibs[j] for j in col_ind])
		
# 		with Pool(initializer=self.init_pool_processes, initargs=(knn, costs)) as pool_per_key:
# 			results = pool_per_key.starmap( self.scmUnique, zip(list_nodes) )
		
# 		for j in range(len(list_nodes)):
# 			assigned_rows.extend(results[j][0])
# 			assigned_cols.extend(results[j][1])
		
		data=[1 for val in assigned_rows]
		self.scm = csr_array((data, (assigned_rows, assigned_cols)), shape=(knn.shape[0], knn.shape[0]))
# 		print(self.scm)
		
		if self.scenario=="Shenzen":
			with open('shenzen_scm.pickle', 'wb') as handle:
				pickle.dump(self.scm, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		if self.scenario=="Taka":
			with open('taka_scm.pickle', 'wb') as handle:
				pickle.dump(self.scm, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		elif self.scenario=="Senegal":
			with open('senegal_scm.pickle', 'wb') as handle:
				pickle.dump(self.scm, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		elif self.scenario=="Civ":
			with open('civ_scm.pickle', 'wb') as handle:
				pickle.dump(self.scm, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		elif self.scenario=="Million":
			with open('million_scm.pickle', 'wb') as handle:
				pickle.dump(self.scm, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		elif self.scenario=="Partial":
			with open('partial_scm'+str(self.num_partial_trajs)+'.pickle', 'wb') as handle:
				pickle.dump(self.scm, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		elif self.scenario=="Week":
			with open('week_scm'+str(self.num_week_trajs)+'.pickle', 'wb') as handle:
				pickle.dump(self.scm, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	def commonCostMatrix(self, costs, knn):
		if self.scenario=="Shenzen":
			with open('shenzen_scm.pickle', 'rb') as handle:
				self.scm = pickle.load(handle)
# 		elif self.scenario=="Taka":
# 			with open('taka_scm.pickle', 'rb') as handle:
# 				self.scm = pickle.load(handle)
		elif self.scenario=="Senegal":
			with open('senegal_scm.pickle', 'rb') as handle:
				self.scm = pickle.load(handle)
		elif self.scenario=="Civ":
			with open('civ_scm.pickle', 'rb') as handle:
				self.scm = pickle.load(handle)
# 		elif self.scenario=="Million":
# 			with open('million_scm.pickle', 'rb') as handle:
# 				self.scm = pickle.load(handle)
# 		elif self.scenario=="Partial":
# 			with open('partial_scm'+str(self.num_partial_trajs)+'.pickle', 'rb') as handle:
# 				self.scm = pickle.load(handle)
		elif self.scenario=="Week":
			with open('week_scm'+str(self.num_week_trajs)+'.pickle', 'rb') as handle:
				self.scm = pickle.load(handle)
		
# 		rows, cols = self.scm.nonzero()
		rows, cols = self.num_million_trajs, self.num_million_trajs

		rows1, cols1 = knn.nonzero()
		
		this_cost = defaultdict(tuple)
		self.row_min = defaultdict(int)
		
		with open('senegal_data_rand_'+str(self.num_partial_trajs)+'.pickle', 'rb') as handle:
			rand_samples = pickle.load(handle)
		
		tr=list(rand_samples)
		
		if self.scenario=="Civ":
			for i in range(self.num_civ_trajs): #len(rows)
				self.row_min[i]=(-1, math.inf) #rows[i]
# 				pass
		elif self.scenario=="Partial":
			for i in range(self.num_partial_trajs):
				self.row_min[i]=(-1, math.inf)
		else:
			for i in range(rows): #len(rows)
				self.row_min[i]=(-1, math.inf)
		
# 		print(rows, len(rows))
		
		if self.scenario=="Civ":
			for key in costs.keys():
				if key[0] == cols1[key[0]*self.K+key[1]]:
	# 				print(key[0], cols1[key[0]*self.K+key[1]])
					this_cost[(key[0], cols1[key[0]*self.K+key[1]])]=math.inf
				else:
					this_cost[(key[0], cols1[key[0]*self.K+key[1]])]=costs[key]

			for key in costs.keys():
				if costs[key]!=0:
					if this_cost[(key[0], cols1[key[0]*self.K+key[1]])]<self.row_min[key[0]][1]:
		# 				print(this_cost[(key[0], cols1[key[0]*self.K+key[1]])][0], self.row_min[key[0]][1])
						self.row_min[key[0]]=(cols1[key[0]*self.K+key[1]], this_cost[(key[0], cols1[key[0]*self.K+key[1]])])
	# 				else:
# 					print(key[0])
		elif self.scenario=="Partial":
			for key in costs.keys():
				if key[0] == key[1]:
	# 				print(key[0], cols1[key[0]*self.K+key[1]])
					this_cost[key]=math.inf
				else:
					this_cost[key]=costs[key]

			for key in costs.keys():
				if this_cost[key]<self.row_min[key[0]][1]:
	# 				print(this_cost[(key[0], cols1[key[0]*self.K+key[1]])][0], self.row_min[key[0]][1])
					self.row_min[key[0]]=(key[1], this_cost[key])
		
		elif self.scenario=="Week":
			for key in costs.keys():
				if key[0] == key[1]:
	# 				print(key[0], cols1[key[0]*self.K+key[1]])
					this_cost[key]=math.inf
				else:
					this_cost[key]=costs[key]

			for key in costs.keys():
				if this_cost[key]<self.row_min[key[0]][1]:
	# 				print(this_cost[(key[0], cols1[key[0]*self.K+key[1]])][0], self.row_min[key[0]][1])
					self.row_min[key[0]]=(key[1], this_cost[key])
		
		elif self.scenario=="Million":
# 			print(len(cols))
			for j in range(self.num_million_trajs):
				traj=j
				index0=traj*self.K
				for i in range(self.K):
					if traj==cols1[index0+i]:
						this_cost[(traj, cols1[index0+i])]=math.inf
					else:
						this_cost[(traj, cols1[index0+i])]=costs[traj*self.K+i]
			
			for j in range(self.num_million_trajs):
				traj=j
				index0=traj*self.K
				for i in range(self.K):
					if self.row_min[traj]!=0:
						if this_cost[(traj, cols1[index0+i])]<self.row_min[traj][1]:
			# 				print(this_cost[(key[0], cols1[key[0]*self.K+key[1]])][0], self.row_min[key[0]][1])
							self.row_min[traj]=(cols1[index0+i], this_cost[(traj, cols1[index0+i])])
					else:
						print(traj)
		else:
			for key in costs.keys():
				if key[0] == cols1[key[0]*self.K+key[1]]:
	# 				print(key[0], cols1[key[0]*self.K+key[1]])
					this_cost[(key[0], cols1[key[0]*self.K+key[1]])]=(math.inf, math.inf)
				else:
					this_cost[(key[0], cols1[key[0]*self.K+key[1]])]=costs[key]

			for key in costs.keys():
				if this_cost[(key[0], cols1[key[0]*self.K+key[1]])][0]<self.row_min[key[0]][1]:
	# 				print(this_cost[(key[0], cols1[key[0]*self.K+key[1]])][0], self.row_min[key[0]][1])
					self.row_min[key[0]]=(cols1[key[0]*self.K+key[1]], this_cost[(key[0], cols1[key[0]*self.K+key[1]])][0])
		
# 		print(list(self.row_min.items())[:1000])
		
# 		for i in range(len(rows)):
# 			this_cost[(rows[i], cols[i])]=(sys.maxsize, sys.maxsize)

		del rows1, cols1, costs
		
		traj_dict=defaultdict(int)
		for i, val in enumerate(rand_samples):
			traj_dict[val]=i
		
		if self.scenario=="Civ":
			tot_rows, tot_cols, data=[],[],[]
			for key in this_cost.keys():
				tot_rows.append(key[0])
				tot_cols.append(key[1])
				data.append(this_cost[key])
		elif self.scenario=="Million":
			tot_rows, tot_cols, data=[],[],[]
			for key in this_cost.keys():
				tot_rows.append(key[0])
				tot_cols.append(key[1])
				data.append(this_cost[key])
		elif self.scenario=="Partial":
			tot_rows, tot_cols, data=[],[],[]
			for key in this_cost.keys():
				tot_rows.append(traj_dict[key[0]])
				tot_cols.append(traj_dict[key[1]])
				data.append(this_cost[key])
		elif self.scenario=="Week":
			tot_rows, tot_cols, data=[],[],[]
			for key in this_cost.keys():
				tot_rows.append(traj_dict[key[0]])
				tot_cols.append(traj_dict[key[1]])
				data.append(this_cost[key])
		else:
			tot_rows, tot_cols, data=[],[],[]
			for key in this_cost.keys():
				tot_rows.append(key[0])
				tot_cols.append(key[1])
				data.append(this_cost[key][0])
		
		del this_cost
		
# 		print(len(rows), len(cols), len(tot_rows), rows[-1], max(tot_rows), max(tot_cols))
		if self.scenario=="Civ":
			self.ccm = csr_array((data, (tot_rows, tot_cols)), shape=(self.num_civ_trajs, self.num_civ_trajs)) #rows[-1]+1, rows[-1]+1)
# 			pass
		else:
			self.ccm = csr_array((data, (tot_rows, tot_cols)), shape=(rows, cols)) #len(rows), len(cols)
	
		if self.scenario=="Shenzen":
			with open('shenzen_ccm.pickle', 'wb') as handle:
				pickle.dump(self.ccm, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
			with open('shenzen_row_min.pickle', 'wb') as handle:
				pickle.dump(self.row_min, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		elif self.scenario=="Taka":
			with open('taka_ccm.pickle', 'wb') as handle:
				pickle.dump(self.ccm, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
			with open('taka_row_min.pickle', 'wb') as handle:
				pickle.dump(self.row_min, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		elif self.scenario=="Senegal":
			with open('senegal_ccm.pickle', 'wb') as handle:
				pickle.dump(self.ccm, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
			with open('senegal_row_min.pickle', 'wb') as handle:
				pickle.dump(self.row_min, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		elif self.scenario=="Partial":
			with open('partial_ccm'+str(self.num_partial_trajs)+'.pickle', 'wb') as handle:
				pickle.dump(self.ccm, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
			with open('partial_row_min'+str(self.num_partial_trajs)+'.pickle', 'wb') as handle:
				pickle.dump(self.row_min, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		elif self.scenario=="Week":
			with open('week_ccm'+str(self.num_week_trajs)+'.pickle', 'wb') as handle:
				pickle.dump(self.ccm, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
			with open('week_row_min'+str(self.num_week_trajs)+'.pickle', 'wb') as handle:
				pickle.dump(self.row_min, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		elif self.scenario=="Civ":
			with open('civ_ccm.pickle', 'wb') as handle:
				pickle.dump(self.ccm, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
			with open('civ_row_min.pickle', 'wb') as handle:
				pickle.dump(self.row_min, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		elif self.scenario=="Million":
			with open('million_ccm.pickle', 'wb') as handle:
				pickle.dump(self.ccm, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
			del self.ccm
			
			with open('million_row_min.pickle', 'wb') as handle:
				pickle.dump(self.row_min, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
			del self.row_min
	
	def kAnonFinal(self, scenario):
# 		print(self.ccm)
		ccm_there=True
		if scenario=="Shenzen":
			if not ccm_there:
				with open('shenzen_ccm.pickle', 'wb') as handle:
					pickle.dump(self.ccm, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
			else:
				with open('shenzen_ccm.pickle', 'rb') as handle:
					self.ccm = pickle.load(handle)
				
				with open('shenzen_scm.pickle', 'rb') as handle:
					self.scm = pickle.load(handle)
				
				with open('shenzen_row_min.pickle', 'rb') as handle:
					self.row_min = pickle.load(handle)
				
				rows, cols = self.ccm.nonzero()
				
				picks = []
				for i in range(self.num_shenzen_trajs):
					picks.append([])
				
				dt = self.ccm.data
# 				for val in dt:
# 					if val==sys.maxsize:
# 						print(val)
				num_pts=len(rows)
				for j in range(num_pts):
					picks[cols[j]].append( (rows[j], dt[j]) )
				
				if self.k>2:
					for i in range(self.num_shenzen_trajs):
						srt_t = sorted(picks[i], key=lambda x:x[1])
	# 					print(srt_t)
						picks[i]=[val[0] for val in srt_t[:self.k-2]]

	# 					print(srt_t, picks[i], sys.maxsize)
	# 					break
				
				rows1, cols1 = self.scm.nonzero()
				merge_list=[]
				for i in range(self.num_shenzen_trajs):
					merge_list.append([i])
				
				if self.k>2:
					for i in range(self.num_shenzen_trajs):
						for j,val in enumerate(picks[i]):
							merge_list[val].append( i )
				
				for i in range(self.num_shenzen_trajs):
					merge_list[cols1[i]].append( rows1[i] )
				
				for i in range(len(merge_list)):
					if len(merge_list[i])==1:
						merge_list[i].append(self.row_min[i][0])
				
				num_picks=[len(ml)-1 for ml in merge_list]
# 		print(len(set(rows1)), len(set(cols1)))
				
				x=defaultdict(int)
				for i,npk in enumerate(num_picks):
					x[npk]=x[npk]+1
				x=dict(sorted(x.items(), key = lambda y:y[0]))
				print(x)
				
				tot=0
				for key in x.keys():
					tot=tot+key*x[key]
				print("The total number of picks is: ", tot)
				
# 				print(merge_list[:1000])
				
				ks=list(x.keys())
				for j in range(1, len(ks)):
					x[ks[j]] = x[ks[j-1]] + x[ks[j]]
				print(x)

# 				print(picks[:100]) #merge_list[:100]
				
				
# 				self.plot_dist(num_picks, "Num. of picks", "num_picks_shenzen")
				
				
				with open('shenzen_merge_list.pickle', 'wb') as handle:
					pickle.dump(merge_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		elif scenario=="Senegal":
			if not ccm_there:
				with open('senegal_ccm.pickle', 'wb') as handle:
					pickle.dump(self.ccm, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
			else:
				with open('senegal_ccm.pickle', 'rb') as handle:
					self.ccm = pickle.load(handle)
				
				with open('senegal_scm.pickle', 'rb') as handle:
					self.scm = pickle.load(handle)
				
				with open('senegal_row_min.pickle', 'rb') as handle:
					self.row_min = pickle.load(handle)
				
				rows, cols = self.ccm.nonzero()
				
				picks = []
				for i in range(self.num_senegal_trajs):
					picks.append([])
				
				dt = self.ccm.data
# 				for val in dt:
# 					if val==sys.maxsize:
# 						print(val)
				num_pts=len(rows)
				for j in range(num_pts):
					picks[cols[j]].append( (rows[j], dt[j]) )
				
				if self.k>2:
					for i in range(self.num_senegal_trajs):
						srt_t = sorted(picks[i], key=lambda x:x[1])
	# 					print(srt_t)
						picks[i]=[val[0] for val in srt_t[:self.k-2]]

	# 					print(srt_t, picks[i], sys.maxsize)
	# 					break
				
				rows1, cols1 = self.scm.nonzero()
				merge_list=[]
				for i in range(self.num_senegal_trajs):
					merge_list.append([i])
				
				if self.k>2:
					for i in range(self.num_senegal_trajs):
						for j,val in enumerate(picks[i]):
							merge_list[val].append( i )
				
				for i in range(self.num_senegal_trajs):
					merge_list[cols1[i]].append( rows1[i] )
				
				for i in range(len(merge_list)):
					if len(merge_list[i])==1:
						merge_list[i].append(self.row_min[i][0])
				
				num_picks=[len(ml)-1 for ml in merge_list]
# 		print(len(set(rows1)), len(set(cols1)))
				
				x=defaultdict(int)
				for i,npk in enumerate(num_picks):
					x[npk]=x[npk]+1
				x=dict(sorted(x.items(), key = lambda y:y[0]))
				print(x)
				
				tot=0
				for key in x.keys():
					tot=tot+key*x[key]
				print("The total number of picks is: ", tot)
				
# 				print(merge_list[:1000])
				
				ks=list(x.keys())
				for j in range(1, len(ks)):
					x[ks[j]] = x[ks[j-1]] + x[ks[j]]
				print(x)

# 				print(picks[:100]) #merge_list[:100]
				
				
# 				self.plot_dist(num_picks, "Num. of picks", "num_picks_shenzen")
				
				
				with open('senegal_merge_list.pickle', 'wb') as handle:
					pickle.dump(merge_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		elif scenario=="Partial":
			if not ccm_there:
				with open('partial_ccm'+str(self.num_partial_trajs)+'.pickle', 'wb') as handle:
					pickle.dump(self.ccm, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
			else:
				with open('partial_ccm'+str(self.num_partial_trajs)+'.pickle', 'rb') as handle:
					self.ccm = pickle.load(handle)
				
# 				with open('partial_scm'+str(self.num_partial_trajs)+'.pickle', 'rb') as handle:
# 					self.scm = pickle.load(handle)
				
				with open('partial_row_min'+str(self.num_partial_trajs)+'.pickle', 'rb') as handle:
					self.row_min = pickle.load(handle)
				
				rows, cols = self.ccm.nonzero()
				
				with open('senegal_data_rand_'+str(self.num_partial_trajs)+'.pickle', 'rb') as handle:
					rand_samples = pickle.load(handle)

				self.num_partial_trajs = len(tdata_full)
				
				tr=list(rand_samples)
				
				traj_dict=defaultdict(int)
				for i, val in enumerate(rand_samples):
					traj_dict[val]=i
				
				picks = []
				for i in range(self.num_partial_trajs):
					picks.append([])
				
				dt = self.ccm.data
# 				for val in dt:
# 					if val==sys.maxsize:
# 						print(val)
				num_pts=len(rows)
				for j in range(num_pts):
					picks[cols[j]].append( (rows[j], dt[j]) )
				
				if self.k>1: #2
					for i in range(self.num_partial_trajs):
# 						srt_t = sorted(picks[i], key=lambda x:x[1]) #i
# 	# 					print(srt_t)
# 						picks[i]=[val[0] for val in srt_t[:self.k-2]] #i 2

# 	# 					print(srt_t, picks[i], sys.maxsize)
# 	# 					break

						srt_t = sorted(picks[i], key=lambda x:x[1])
# 						print(srt_t[:self.k-1])
# 						picks[i]=[val[0] for val in srt_t[:self.k-1]]#k-2
						
						xp=[]
						for val in srt_t:
# 							if val[1]>0 and mark_singleton[val[0]]:
# 								continue
# 							else:
							xp.append(val[0])
							if len(xp)==self.k-1:
								break
						picks[i]=[val for val in xp]

# 						print(srt_t, picks[i], sys.maxsize)
	# 					break
				
# 				rows1, cols1 = self.scm.nonzero()
				merge_list=[]
				for i in range(self.num_partial_trajs):
					merge_list.append([i]) #i
				
				if self.k>1: #2
					for i in range(self.num_partial_trajs):
						for j,val in enumerate(picks[i]): #i
							merge_list[val].append( i ) #i
				
# 				for i in range(self.num_partial_trajs):
# 					merge_list[cols1[i]].append( rows1[i] )#rows1[i] #col1, row1
				
				for i in range(len(merge_list)):
					if len(merge_list[i])==1:
						merge_list[i].append(self.row_min[i][0])#i traj_dict[self.row_min[tr[i]][0]
				
				num_picks=[len(ml)-1 for ml in merge_list]
# 		print(len(set(rows1)), len(set(cols1)))
				
				x=defaultdict(int)
				for i,npk in enumerate(num_picks):
					x[npk]=x[npk]+1
				x=dict(sorted(x.items(), key = lambda y:y[0]))
				print(x)
				
				tot=0
				for key in x.keys():
					tot=tot+key*x[key]
				print("The total number of picks is: ", tot)
				
# 				print(merge_list[:1000])
				
				ks=list(x.keys())
				for j in range(1, len(ks)):
					x[ks[j]] = x[ks[j-1]] + x[ks[j]]
				print(x)

# 				print(picks[:100]) #merge_list[:100]
				
				
# 				self.plot_dist(num_picks, "Num. of picks", "num_picks_shenzen")
				
				
				with open('partial_merge_list'+str(self.num_partial_trajs)+'.pickle', 'wb') as handle:
					pickle.dump(merge_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		elif scenario=="Week":
			if not ccm_there:
				with open('week_ccm'+str(self.num_week_trajs)+'.pickle', 'wb') as handle:
					pickle.dump(self.ccm, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
			else:
				with open('week_ccm'+str(self.num_week_trajs)+'.pickle', 'rb') as handle:
					self.ccm = pickle.load(handle)
				
				with open('week_scm'+str(self.num_week_trajs)+'.pickle', 'rb') as handle:
					self.scm = pickle.load(handle)
				
				with open('week_row_min'+str(self.num_week_trajs)+'.pickle', 'rb') as handle:
					self.row_min = pickle.load(handle)
				
				rows, cols = self.ccm.nonzero()
				
				picks = []
				for i in range(self.num_week_trajs):
					picks.append([])
				
				dt = self.ccm.data
# 				for val in dt:
# 					if val==sys.maxsize:
# 						print(val)
				num_pts=len(rows)
				for j in range(num_pts):
					picks[cols[j]].append( (rows[j], dt[j]) )
				
				if self.k>2:
					for i in range(self.num_week_trajs):
						srt_t = sorted(picks[i], key=lambda x:x[1])
	# 					print(srt_t)
						picks[i]=[val[0] for val in srt_t[:self.k-2]]

	# 					print(srt_t, picks[i], sys.maxsize)
	# 					break
				
				rows1, cols1 = self.scm.nonzero()
				merge_list=[]
				for i in range(self.num_week_trajs):
					merge_list.append([i])
				
				if self.k>2:
					for i in range(self.num_week_trajs):
						for j,val in enumerate(picks[i]):
							merge_list[val].append( i )
				
				for i in range(self.num_week_trajs):
					merge_list[cols1[i]].append( rows1[i] )
				
				for i in range(len(merge_list)):
					if len(merge_list[i])==1:
						merge_list[i].append(self.row_min[i][0])
				
				num_picks=[len(ml)-1 for ml in merge_list]
# 		print(len(set(rows1)), len(set(cols1)))
				
				x=defaultdict(int)
				for i,npk in enumerate(num_picks):
					x[npk]=x[npk]+1
				x=dict(sorted(x.items(), key = lambda y:y[0]))
				print(x)
				
				tot=0
				for key in x.keys():
					tot=tot+key*x[key]
				print("The total number of picks is: ", tot)
				
# 				print(merge_list[:1000])
				
				ks=list(x.keys())
				for j in range(1, len(ks)):
					x[ks[j]] = x[ks[j-1]] + x[ks[j]]
				print(x)

# 				print(picks[:100]) #merge_list[:100]
				
				
# 				self.plot_dist(num_picks, "Num. of picks", "num_picks_shenzen")
				
				
				with open('week_merge_list.pickle', 'wb') as handle:
					pickle.dump(merge_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		elif scenario=="Civ":
			if not ccm_there:
				with open('civ_ccm.pickle', 'wb') as handle:
					pickle.dump(self.ccm, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
			else:
				with open('civ_ccm.pickle', 'rb') as handle:
					self.ccm = pickle.load(handle)
				
				with open('civ_scm.pickle', 'rb') as handle:
					self.scm = pickle.load(handle)
				
				with open('civ_row_min.pickle', 'rb') as handle:
					self.row_min = pickle.load(handle)
				
				rows, cols = self.ccm.nonzero()
				
				picks = []
				for i in range(self.num_civ_trajs):
					picks.append([])
				
				dt = self.ccm.data
# 				for val in dt:
# 					if val==sys.maxsize:
# 						print(val)
				num_pts=len(rows)
				for j in range(num_pts):
					picks[cols[j]].append( (rows[j], dt[j]) )
				
				if self.k>2:
					for i in range(self.num_civ_trajs):
						srt_t = sorted(picks[i], key=lambda x:x[1])
	# 					print(srt_t)
						picks[i]=[val[0] for val in srt_t[:self.k-2]]

	# 					print(srt_t, picks[i], sys.maxsize)
	# 					break
				
				rows1, cols1 = self.scm.nonzero()
				merge_list=[]
				for i in range(self.num_civ_trajs):
					merge_list.append([i])
				
				if self.k>2:
					for i in range(self.num_civ_trajs):
						for j,val in enumerate(picks[i]):
							merge_list[val].append( i )
				
				for i in range(len(rows1)):#self.num_civ_trajs
					merge_list[cols1[i]].append( rows1[i] )
				
				for i in range(len(merge_list)):
					if len(merge_list[i])==1:
						if self.row_min[i]!=0:
							merge_list[i].append(self.row_min[i][0])
						else:
							merge_list[i].append(-1)
				
				num_picks=[len(ml)-1 for ml in merge_list]
# 		print(len(set(rows1)), len(set(cols1)))
				
				x=defaultdict(int)
				for i,npk in enumerate(num_picks):
					x[npk]=x[npk]+1
				x=dict(sorted(x.items(), key = lambda y:y[0]))
				print(x)
				
				tot=0
				for key in x.keys():
					tot=tot+key*x[key]
				print("The total number of picks is: ", tot)
				
# 				print(merge_list[:1000])
				
				ks=list(x.keys())
				for j in range(1, len(ks)):
					x[ks[j]] = x[ks[j-1]] + x[ks[j]]
				print(x)

# 				print(picks[:100]) #merge_list[:100]
				
				
# 				self.plot_dist(num_picks, "Num. of picks", "num_picks_shenzen")
				
				
				with open('civ_merge_list.pickle', 'wb') as handle:
					pickle.dump(merge_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		elif scenario=="Million":
			if not ccm_there:
				with open('million_ccm.pickle', 'wb') as handle:
					pickle.dump(self.ccm, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
			else:
				with open('million_ccm.pickle', 'rb') as handle:
					self.ccm = pickle.load(handle)
				
# 				with open('million_scm.pickle', 'rb') as handle:
# 					self.scm = pickle.load(handle)
				
				with open('million_row_min.pickle', 'rb') as handle:
					self.row_min = pickle.load(handle)
				
# 				with open('singletons_million.pickle', 'rb') as handle:
# 					singletons = pickle.load(handle)
				
				mark_singleton=[]
				
				rows, cols = self.ccm.nonzero()
				
				picks = []
				for i in range(self.num_million_trajs):
					picks.append([])
# 					mark_singleton.append(False)
				
# 				for val in singletons:
# 					mark_singleton[val]=True
				
				dt = self.ccm.data
# 				for val in dt:
# 					if val==sys.maxsize:
# 						print(val)
				num_pts=len(rows)
				for j in range(num_pts):
# 					if dt[j]==0:
# 						picks[cols[j]].append( (rows[j], sys.maxsize) )
# 					else:
					picks[cols[j]].append( (rows[j], dt[j]) )
	
				st_picks = picks.copy()
				
				if self.k>1:#2
					for i in range(self.num_million_trajs):
						srt_t = sorted(picks[i], key=lambda x:x[1])
# 						print(srt_t[:self.k-1])
# 						picks[i]=[val[0] for val in srt_t[:self.k-1]]#k-2
						
						xp=[]
						for val in srt_t:
# 							if val[1]>0 and mark_singleton[val[0]]:
# 								continue
# 							else:
							xp.append(val[0])
							if len(xp)==self.k-1:
								break
						picks[i]=[val for val in xp]

# 						print(srt_t, picks[i], sys.maxsize)
	# 					break
				
# 				rows1, cols1 = self.scm.nonzero()
				merge_list=[]
				for i in range(self.num_million_trajs):
					merge_list.append([i])
				
				if self.k>1:#2
					for i in range(self.num_million_trajs):
						for j,val in enumerate(picks[i]):
							merge_list[val].append( i )
				
# 				for i in range(self.num_million_trajs):
# 					merge_list[cols1[i]].append( rows1[i] )#col1, row1
				
				count2=0
				for i in range(len(merge_list)):
					if len(merge_list[i])==1:
# 						srt_t = sorted(st_picks[i], key=lambda x:x[1])
						
# 						for val in srt_t:
# 							if val[1]>0 and mark_singleton[val[0]]:
# 								continue
# 							else:
# 								merge_list[i].append(val[0])
# 								break
						merge_list[i].append(self.row_min[i][0])
# 						if self.row_min[i][0]==0:
# 							count2=count2+1
				
# 				print("There are: ", count2, "trajectories with zero picks.")
				
				num_picks=[len(ml)-1 for ml in merge_list]
# 		print(len(set(rows1)), len(set(cols1)))
				
				x=defaultdict(int)
				for i,npk in enumerate(num_picks):
					x[npk]=x[npk]+1
				x=dict(sorted(x.items(), key = lambda y:y[0]))
				print(x)
				
				tot=0
				for key in x.keys():
					tot=tot+key*x[key]
				print("The total number of picks is: ", tot)
				
# 				print(merge_list[:1000])
				
				ks=list(x.keys())
				for j in range(1, len(ks)):
					x[ks[j]] = x[ks[j-1]] + x[ks[j]]
				print(x)

# 				print(picks[:100]) #merge_list[:100]
				
				
# 				self.plot_dist(num_picks, "Num. of picks", "num_picks_shenzen")
				
				
				with open('million_merge_list.pickle', 'wb') as handle:
					pickle.dump(merge_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		elif scenario=="Taka":
			if not ccm_there:
				with open('taka_ccm.pickle', 'wb') as handle:
					pickle.dump(self.ccm, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
			else:
				with open('taka_ccm.pickle', 'rb') as handle:
					self.ccm = pickle.load(handle)
				
# 				with open('million_scm.pickle', 'rb') as handle:
# 					self.scm = pickle.load(handle)
				
				with open('taka_row_min.pickle', 'rb') as handle:
					self.row_min = pickle.load(handle)
				
# 				with open('singletons_million.pickle', 'rb') as handle:
# 					singletons = pickle.load(handle)
				
				mark_singleton=[]
				
				rows, cols = self.ccm.nonzero()
				
				picks = []
				for i in range(self.num_taka_trajs):
					picks.append([])
# 					mark_singleton.append(False)
				
# 				for val in singletons:
# 					mark_singleton[val]=True
				
				dt = self.ccm.data
# 				for val in dt:
# 					if val==sys.maxsize:
# 						print(val)
				num_pts=len(rows)
				for j in range(num_pts):
# 					if dt[j]==0:
# 						picks[cols[j]].append( (rows[j], sys.maxsize) )
# 					else:
					picks[cols[j]].append( (rows[j], dt[j]) )
	
				st_picks = picks.copy()
				
				if self.k>1:#2
					for i in range(self.num_taka_trajs):
						srt_t = sorted(picks[i], key=lambda x:x[1])
# 						print(srt_t[:self.k-1])
# 						picks[i]=[val[0] for val in srt_t[:self.k-1]]#k-2
						
						xp=[]
						for val in srt_t:
# 							if val[1]>0 and mark_singleton[val[0]]:
# 								continue
# 							else:
							xp.append(val[0])
							if len(xp)==self.k-1:
								break
						picks[i]=[val for val in xp]

# 						print(srt_t, picks[i], sys.maxsize)
	# 					break
				
# 				rows1, cols1 = self.scm.nonzero()
				merge_list=[]
				for i in range(self.num_taka_trajs):
					merge_list.append([i])
				
				if self.k>1:#2
					for i in range(self.num_taka_trajs):
						for j,val in enumerate(picks[i]):
							merge_list[val].append( i )
				
# 				for i in range(self.num_million_trajs):
# 					merge_list[cols1[i]].append( rows1[i] )#col1, row1
				
				count2=0
				for i in range(len(merge_list)):
					if len(merge_list[i])==1:
# 						srt_t = sorted(st_picks[i], key=lambda x:x[1])
						
# 						for val in srt_t:
# 							if val[1]>0 and mark_singleton[val[0]]:
# 								continue
# 							else:
# 								merge_list[i].append(val[0])
# 								break
						merge_list[i].append(self.row_min[i][0])
# 						if self.row_min[i][0]==0:
# 							count2=count2+1
				
# 				print("There are: ", count2, "trajectories with zero picks.")
				
				num_picks=[len(ml)-1 for ml in merge_list]
# 		print(len(set(rows1)), len(set(cols1)))
				
				x=defaultdict(int)
				for i,npk in enumerate(num_picks):
					x[npk]=x[npk]+1
				x=dict(sorted(x.items(), key = lambda y:y[0]))
				print(x)
				
				tot=0
				for key in x.keys():
					tot=tot+key*x[key]
				print("The total number of picks is: ", tot)
				
# 				print(merge_list[:1000])
				
				ks=list(x.keys())
				for j in range(1, len(ks)):
					x[ks[j]] = x[ks[j-1]] + x[ks[j]]
				print(x)

# 				print(picks[:100]) #merge_list[:100]
				
				
# 				self.plot_dist(num_picks, "Num. of picks", "num_picks_shenzen")
				
				
				with open('taka_merge_list.pickle', 'wb') as handle:
					pickle.dump(merge_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def get_kmerge_trajs_cost(self, indexes):
		if self.scenario=="Shenzen":
			file = open("generalised_shenzennew/kmerge_cost_shenzen_m"+ str(indexes[0]) + ".txt", "w")
			
			for i in indexes:
				tp=[]
				trajs=self.merge_list[i]
				idd=0
				if len(trajs) <= 10:
					for j in range(len(trajs)):
	# 					print(len(tdata_full_shenzen[trajs[j]]))
						traj1=np.zeros((len(tdata_full_shenzen[trajs[j]]), 9))

						for k in range(len(traj1)):
							traj1[k] = np.append(tdata_full_shenzen[trajs[j]][k], idd)

						tp.extend( [*traj1] )

						idd = idd + 1
					tp = np.array(tp)
# 					print(i, tp.shape[0], tp.shape[1], len(trajs), trajs)
				
					this_results = get_node_trajs_m(tp, len(trajs))
				else:
					gtraz=[]
					count=0
					for l in range(int(len(trajs)/self.k)+1):
						chosen_trajs=trajs[10*l:10*(l+1)]
						if len(chosen_trajs)>0:
							tp=[]
							idd=0
							for j in range(len(chosen_trajs)):
			# 					print(len(tdata_full_shenzen[trajs[j]]))
								traj1=np.zeros((len(tdata_full_shenzen[chosen_trajs[j]]), 9))

								for k in range(len(traj1)):
									traj1[k] = np.append(tdata_full_shenzen[chosen_trajs[j]][k], idd)

								tp.extend( [*traj1] )

								idd = idd + 1
							tp = np.array(tp)
	# 						print(i, tp, tp.shape[0], tp.shape[1], len(chosen_trajs), chosen_trajs)

# 							print(i, tp, chosen_trajs)
							trs=get_node_trajs_m(tp, len(chosen_trajs))
							trs = trs.reshape(trs.shape[0], 8)

							t=np.zeros( (trs.shape[0], 9) )

							for k in range(trs.shape[0]):
								t[k] = np.append(trs[k], l)
	# 						print(trs)
	
							gtraz.extend( t )
		
							count = count + 1
					gtraz=np.array(gtraz)
# 					print(gtraz.shape)
# 					gtraz = gtraz.reshape(gtraz.shape[0], 8)
# 					print(i, gtraz)
					this_results = get_node_trajs_m(gtraz, count)
	
				trace=""
				for res in this_results:
					if res[0][0]!=0 or res[1][0]!=0:
						trace=trace + ' '.join(str(int(x)) for x in res[0][:3]) + ' ' + ' '.join(str(int(x)) for x in res[0][4:])
						trace=trace + ' '.join(str(int(x)) for x in res[1][:3]) + ' ' + ' '.join(str(int(x)) for x in res[1][4:]) + '\n'
				trace=trace+'\n'

				file.write(trace)#this_results
		
		elif self.scenario=="Taka":
			file = open("generalised_takanew/kmerge_cost_taka_m"+ str(indexes[0]) + ".txt", "w")
			
			for i in indexes:
				tp=[]
				trajs=self.merge_list[i]
				idd=0
				if len(trajs) <= 10:
					for j in range(len(trajs)):
	# 					print(len(tdata_full_shenzen[trajs[j]]))
						traj1=np.zeros((len(tdata_full_taka[trajs[j]]), 9))

						for k in range(len(traj1)):
							traj1[k] = np.append(tdata_full_taka[trajs[j]][k], idd)

						tp.extend( [*traj1] )

						idd = idd + 1
					tp = np.array(tp)
# 					print(i, tp.shape[0], tp.shape[1], len(trajs), trajs)
				
					this_results = get_node_trajs_m(tp, len(trajs))
				else:
					gtraz=[]
					count=0
					for l in range(int(len(trajs)/self.k)+1):
						chosen_trajs=trajs[10*l:10*(l+1)]
						if len(chosen_trajs)>0:
							tp=[]
							idd=0
							for j in range(len(chosen_trajs)):
			# 					print(len(tdata_full_shenzen[trajs[j]]))
								traj1=np.zeros((len(tdata_full_taka[chosen_trajs[j]]), 9))

								for k in range(len(traj1)):
									traj1[k] = np.append(tdata_full_taka[chosen_trajs[j]][k], idd)

								tp.extend( [*traj1] )

								idd = idd + 1
							tp = np.array(tp)
	# 						print(i, tp, tp.shape[0], tp.shape[1], len(chosen_trajs), chosen_trajs)

# 							print(i, tp, chosen_trajs)
							trs=get_node_trajs_m(tp, len(chosen_trajs))
							trs = trs.reshape(trs.shape[0], 8)

							t=np.zeros( (trs.shape[0], 9) )

							for k in range(trs.shape[0]):
								t[k] = np.append(trs[k], l)
	# 						print(trs)
	
							gtraz.extend( t )
		
							count = count + 1
					gtraz=np.array(gtraz)
# 					print(gtraz.shape)
# 					gtraz = gtraz.reshape(gtraz.shape[0], 8)
# 					print(i, gtraz)
					this_results = get_node_trajs_m(gtraz, count)
	
				trace=""
				for res in this_results:
					if res[0][0]!=0 or res[1][0]!=0:
						trace=trace + ' '.join(str(int(x)) for x in res[0][:3]) + ' ' + ' '.join(str(int(x)) for x in res[0][4:])
						trace=trace + ' '.join(str(int(x)) for x in res[1][:3]) + ' ' + ' '.join(str(int(x)) for x in res[1][4:]) + '\n'
				trace=trace+'\n'

				file.write(trace)#this_results
		
		elif self.scenario=="Senegal":
			file = open("generalised_senegal2/kmerge_cost_senegal_m"+ str(indexes[0]) + ".txt", "w")
			
			for i in indexes:
				tp=[]
				trajs=self.merge_list[i]
				idd=0
				if len(trajs) <= 10:
					for j in range(len(trajs)):
	# 					print(len(tdata_full_shenzen[trajs[j]]))
						traj1=np.zeros((len(tdata_full[trajs[j]]), 9))

						for k in range(len(traj1)):
							traj1[k] = np.append(tdata_full[trajs[j]][k], idd)

						tp.extend( [*traj1] )

						idd = idd + 1
					tp = np.array(tp)
# 					print(i, tp.shape[0], tp.shape[1], len(trajs), trajs)
				
					this_results = get_node_trajs_m(tp, len(trajs))
				else:
					gtraz=[]
					count=0
					for l in range(int(len(trajs)/self.k)+1):
						chosen_trajs=trajs[10*l:10*(l+1)]
						if len(chosen_trajs)>0:
							tp=[]
							idd=0
							for j in range(len(chosen_trajs)):
			# 					print(len(tdata_full_shenzen[trajs[j]]))
								traj1=np.zeros((len(tdata_full[chosen_trajs[j]]), 9))

								for k in range(len(traj1)):
									traj1[k] = np.append(tdata_full[chosen_trajs[j]][k], idd)

								tp.extend( [*traj1] )

								idd = idd + 1
							tp = np.array(tp)
	# 						print(i, tp, tp.shape[0], tp.shape[1], len(chosen_trajs), chosen_trajs)

# 							print(i, tp, chosen_trajs)
							trs=get_node_trajs_m(tp, len(chosen_trajs))
							trs = trs.reshape(trs.shape[0], 8)

							t=np.zeros( (trs.shape[0], 9) )

							for k in range(trs.shape[0]):
								t[k] = np.append(trs[k], l)
	# 						print(trs)
	
							gtraz.extend( t )
		
							count = count + 1
					gtraz=np.array(gtraz)
# 					print(gtraz.shape)
# 					gtraz = gtraz.reshape(gtraz.shape[0], 8)
# 					print(i, gtraz)
					this_results = get_node_trajs_m(gtraz, count)
	
				trace=""
				for res in this_results:
					if res[0][0]!=0 or res[1][0]!=0:
						trace=trace + ' '.join(str(int(x)) for x in res[0][:3]) + ' ' + ' '.join(str(int(x)) for x in res[0][4:])
						trace=trace + ' '.join(str(int(x)) for x in res[1][:3]) + ' ' + ' '.join(str(int(x)) for x in res[1][4:]) + '\n'
				trace=trace+'\n'

				file.write(trace)#this_results
		
		elif self.scenario=="Week":
			file = open("generalised_week/kmerge_cost_senegal_m"+ str(indexes[0]) + ".txt", "w")
			
			for i in indexes:
				tp=[]
				trajs=self.merge_list[i]
				idd=0
				if len(trajs) <= 10:
					for j in range(len(trajs)):
	# 					print(len(tdata_full_shenzen[trajs[j]]))
						traj1=np.zeros((len(tdata_full[trajs[j]]), 9))

						for k in range(len(traj1)):
							traj1[k] = np.append(tdata_full[trajs[j]][k], idd)

						tp.extend( [*traj1] )

						idd = idd + 1
					tp = np.array(tp)
# 					print(i, tp.shape[0], tp.shape[1], len(trajs), trajs)
				
					this_results = get_node_trajs_m(tp, len(trajs))
				else:
					gtraz=[]
					count=0
					for l in range(int(len(trajs)/self.k)+1):
						chosen_trajs=trajs[10*l:10*(l+1)]
						if len(chosen_trajs)>0:
							tp=[]
							idd=0
							for j in range(len(chosen_trajs)):
			# 					print(len(tdata_full_shenzen[trajs[j]]))
								traj1=np.zeros((len(tdata_full[chosen_trajs[j]]), 9))

								for k in range(len(traj1)):
									traj1[k] = np.append(tdata_full[chosen_trajs[j]][k], idd)

								tp.extend( [*traj1] )

								idd = idd + 1
							tp = np.array(tp)
	# 						print(i, tp, tp.shape[0], tp.shape[1], len(chosen_trajs), chosen_trajs)

# 							print(i, tp, chosen_trajs)
							trs=get_node_trajs_m(tp, len(chosen_trajs))
							trs = trs.reshape(trs.shape[0], 8)

							t=np.zeros( (trs.shape[0], 9) )

							for k in range(trs.shape[0]):
								t[k] = np.append(trs[k], l)
	# 						print(trs)
	
							gtraz.extend( t )
		
							count = count + 1
					gtraz=np.array(gtraz)
# 					print(gtraz.shape)
# 					gtraz = gtraz.reshape(gtraz.shape[0], 8)
# 					print(i, gtraz)
					this_results = get_node_trajs_m(gtraz, count)
	
				trace=""
				for res in this_results:
					if res[0][0]!=0 or res[1][0]!=0:
						trace=trace + ' '.join(str(int(x)) for x in res[0][:3]) + ' ' + ' '.join(str(int(x)) for x in res[0][4:])
						trace=trace + ' '.join(str(int(x)) for x in res[1][:3]) + ' ' + ' '.join(str(int(x)) for x in res[1][4:]) + '\n'
				trace=trace+'\n'

				file.write(trace)#this_results
		
		elif self.scenario=="Partial":
			file = open("size_test_knn/generlised1/kmerge_cost_partial_m"+ str(indexes[0]) + ".txt", "w")
			
			for i in indexes:
				tp=[]
				trajs=self.merge_list[i]
				idd=0
				if len(trajs) <= 10:
					for j in range(len(trajs)):
	# 					print(len(tdata_full_shenzen[trajs[j]]))
						traj1=np.zeros((len(tdata_full[trajs[j]]), 9))

						for k in range(len(traj1)):
							traj1[k] = np.append(tdata_full[trajs[j]][k], idd)

						tp.extend( [*traj1] )

						idd = idd + 1
					tp = np.array(tp)
# 					print(i, tp.shape[0], tp.shape[1], len(trajs), trajs)
				
					this_results = get_node_trajs_m(tp, len(trajs))
				else:
					gtraz=[]
					count=0
					for l in range(int(len(trajs)/self.k)+1):
						chosen_trajs=trajs[10*l:10*(l+1)]
						if len(chosen_trajs)>0:
							tp=[]
							idd=0
							for j in range(len(chosen_trajs)):
			# 					print(len(tdata_full_shenzen[trajs[j]]))
								traj1=np.zeros((len(tdata_full[chosen_trajs[j]]), 9))

								for k in range(len(traj1)):
									traj1[k] = np.append(tdata_full[chosen_trajs[j]][k], idd)

								tp.extend( [*traj1] )

								idd = idd + 1
							tp = np.array(tp)
	# 						print(i, tp, tp.shape[0], tp.shape[1], len(chosen_trajs), chosen_trajs)

# 							print(i, tp, chosen_trajs)
							trs=get_node_trajs_m(tp, len(chosen_trajs))
							trs = trs.reshape(trs.shape[0], 8)

							t=np.zeros( (trs.shape[0], 9) )

							for k in range(trs.shape[0]):
								t[k] = np.append(trs[k], l)
	# 						print(trs)
	
							gtraz.extend( t )
		
							count = count + 1
					gtraz=np.array(gtraz)
# 					print(gtraz.shape)
# 					gtraz = gtraz.reshape(gtraz.shape[0], 8)
# 					print(i, gtraz)
					this_results = get_node_trajs_m(gtraz, count)
	
				trace=""
				for res in this_results:
					if res[0][0]!=0 or res[1][0]!=0:
						trace=trace + ' '.join(str(int(x)) for x in res[0][:3]) + ' ' + ' '.join(str(int(x)) for x in res[0][4:])
						trace=trace + ' '.join(str(int(x)) for x in res[1][:3]) + ' ' + ' '.join(str(int(x)) for x in res[1][4:]) + '\n'
				trace=trace+'\n'

				file.write(trace)#this_results
		
		elif self.scenario=="Civ":
			file = open("generalised_civnew/kmerge_cost_civ_m"+ str(indexes[0]) + ".txt", "w")
			
			for i in indexes:
				tp=[]
				trajs=self.merge_list[i]
# 				print(trajs)
				idd=0
				if len(trajs) <= 10:
					for j in range(len(trajs)):
	# 					print(len(tdata_full_shenzen[trajs[j]]))
						traj1=np.zeros((len(tdata_full_civ[trajs[j]]), 9))

						for k in range(len(traj1)):
							traj1[k] = np.append(tdata_full_civ[trajs[j]][k], idd)

						tp.extend( [*traj1] )

						idd = idd + 1
					tp = np.array(tp)
# 					print(i, tp.shape[0], tp.shape[1], len(trajs), trajs)
				
					this_results = get_node_trajs_m(tp, len(trajs))
				else:
					gtraz=[]
					count=0
					for l in range(int(len(trajs)/self.k)+1):
						chosen_trajs=trajs[10*l:10*(l+1)]
						if len(chosen_trajs)>0:
							tp=[]
							idd=0
							for j in range(len(chosen_trajs)):
			# 					print(len(tdata_full_shenzen[trajs[j]]))
								traj1=np.zeros((len(tdata_full_civ[chosen_trajs[j]]), 9))

								for k in range(len(traj1)):
									traj1[k] = np.append(tdata_full_civ[chosen_trajs[j]][k], idd)

								tp.extend( [*traj1] )

								idd = idd + 1
							tp = np.array(tp)
	# 						print(i, tp, tp.shape[0], tp.shape[1], len(chosen_trajs), chosen_trajs)

# 							print(i, tp, chosen_trajs)
							trs=get_node_trajs_m(tp, len(chosen_trajs))
							trs = trs.reshape(trs.shape[0], 8)

							t=np.zeros( (trs.shape[0], 9) )

							for k in range(trs.shape[0]):
								t[k] = np.append(trs[k], l)
	# 						print(trs)
	
							gtraz.extend( t )
		
							count = count + 1
					gtraz=np.array(gtraz)
# 					print(gtraz.shape)
# 					gtraz = gtraz.reshape(gtraz.shape[0], 8)
# 					print(i, gtraz)
					this_results = get_node_trajs_m(gtraz, count)
	
				trace=""
				for res in this_results:
					if res[0][0]!=0 or res[1][0]!=0:
						trace=trace + ' '.join(str(int(x)) for x in res[0][:3]) + ' ' + ' '.join(str(int(x)) for x in res[0][4:])
						trace=trace + ' '.join(str(int(x)) for x in res[1][:3]) + ' ' + ' '.join(str(int(x)) for x in res[1][4:]) + '\n'
				trace=trace+'\n'

				file.write(trace)#this_results
		
		elif self.scenario=="Million":
			file = open("generalised_million10newmetric/kmerge_cost_million_m"+ str(indexes[0]) + ".txt", "w") #generalised_million10new
			
			for i in indexes:
				tp=[]
				trajs=self.merge_list[i]
				idd=0
				if len(trajs) <= 10:
					for j in range(len(trajs)):
	# 					print(len(tdata_full_shenzen[trajs[j]]))
						traj1=np.zeros((len(tdata_full[trajs[j]]), 9))

						for k in range(len(traj1)):
							traj1[k] = np.append(tdata_full[trajs[j]][k], idd)

						tp.extend( [*traj1] )

						idd = idd + 1
					tp = np.array(tp)
# 					print(i, tp.shape[0], tp.shape[1], len(trajs), trajs)
				
					this_results = get_node_trajs_m(tp, len(trajs))
				else:
					gtraz=[]
					count=0
					for l in range(int(len(trajs)/self.k)+1):
						chosen_trajs=trajs[10*l:10*(l+1)]
						if len(chosen_trajs)>0:
							tp=[]
							idd=0
							for j in range(len(chosen_trajs)):
			# 					print(len(tdata_full_shenzen[trajs[j]]))
								traj1=np.zeros((len(tdata_full[chosen_trajs[j]]), 9))

								for k in range(len(traj1)):
									traj1[k] = np.append(tdata_full[chosen_trajs[j]][k], idd)

								tp.extend( [*traj1] )

								idd = idd + 1
							tp = np.array(tp)
	# 						print(i, tp, tp.shape[0], tp.shape[1], len(chosen_trajs), chosen_trajs)

# 							print(i, tp, chosen_trajs)
							trs=get_node_trajs_m(tp, len(chosen_trajs))
							trs = trs.reshape(trs.shape[0], 8)

							t=np.zeros( (trs.shape[0], 9) )

							for k in range(trs.shape[0]):
								t[k] = np.append(trs[k], l)
	# 						print(trs)
	
							gtraz.extend( t )
		
							count = count + 1
					gtraz=np.array(gtraz)
# 					print(gtraz.shape)
# 					gtraz = gtraz.reshape(gtraz.shape[0], 8)
# 					print(i, gtraz)
					this_results = get_node_trajs_m(gtraz, count)
	
				trace=""
				for res in this_results:
					if res[0][0]!=0 or res[1][0]!=0:
						trace=trace + ' '.join(str(int(x)) for x in res[0][:3]) + ' ' + ' '.join(str(int(x)) for x in res[0][4:])
						trace=trace + ' '.join(str(int(x)) for x in res[1][:3]) + ' ' + ' '.join(str(int(x)) for x in res[1][4:]) + '\n'
				trace=trace+'\n'

				file.write(trace)#this_results
		file.close()
	
	def generaliseTrajectories(self):
		if self.scenario=="Shenzen":
			with open('shenzen_merge_list.pickle', 'rb') as handle:
				self.merge_list = pickle.load(handle)
			
# 			print(self.merge_list)
			
			indexes=[]
			num_cores=32
			num_trajs=len(self.merge_list)
			
			for i in range(num_cores):
				il = int(num_trajs/num_cores)
				indexes.append([j for j in range(i*il, i*il+il)])

			if indexes[-1][-1]!=num_trajs-1:
				indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])
			
			with Pool() as pool_per_key:
				pool_per_key.starmap( self.get_kmerge_trajs_cost, zip(indexes) )
		
		elif self.scenario=="Taka":
			with open('taka_merge_list.pickle', 'rb') as handle:
				self.merge_list = pickle.load(handle)
			
# 			print(self.merge_list)
			
			indexes=[]
			num_cores=32
			num_trajs=len(self.merge_list)
			
			for i in range(num_cores):
				il = int(num_trajs/num_cores)
				indexes.append([j for j in range(i*il, i*il+il)])

			if indexes[-1][-1]!=num_trajs-1:
				indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])
			
			with Pool() as pool_per_key:
				pool_per_key.starmap( self.get_kmerge_trajs_cost, zip(indexes) )

		elif self.scenario=="Senegal":
			with open('senegal_merge_list.pickle', 'rb') as handle:
				self.merge_list = pickle.load(handle)
			
			indexes=[]
			num_cores=32
			num_trajs=len(self.merge_list)
			
			for i in range(num_cores):
				il = int(num_trajs/num_cores)
				indexes.append([j for j in range(i*il, i*il+il)])

			if indexes[-1][-1]!=num_trajs-1:
				indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])
			
			with Pool() as pool_per_key:
				pool_per_key.starmap( self.get_kmerge_trajs_cost, zip(indexes) )
		
		elif self.scenario=="Week":
			with open('week_merge_list.pickle', 'rb') as handle:
				self.merge_list = pickle.load(handle)
			
			indexes=[]
			num_cores=32
			num_trajs=len(self.merge_list)
			
			for i in range(num_cores):
				il = int(num_trajs/num_cores)
				indexes.append([j for j in range(i*il, i*il+il)])

			if indexes[-1][-1]!=num_trajs-1:
				indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])
			
			with Pool() as pool_per_key:
				pool_per_key.starmap( self.get_kmerge_trajs_cost, zip(indexes) )
		
		elif self.scenario=="Partial":
			with open('partial_merge_list'+str(self.num_partial_trajs)+'.pickle', 'rb') as handle:
				self.merge_list = pickle.load(handle)
			
# 			with open('partial_merge_list'+str(self.num_partial_trajs)+'.pickle', 'rb') as handle:
# 				self.merge_list = pickle.load(handle)
			
# 			print(self.merge_list)
			
# 			with open('senegal_data_rand_'+str(self.num_partial_trajs)+'.pickle', 'rb') as handle:
# 					rand_samples = pickle.load(handle)

# 			tr=list(rand_samples)

# 			traj_dict=defaultdict(int)
# 			for i, val in enumerate(rand_samples):
# 				traj_dict[val]=i
			
# 			for val in self.merge_list:
# 				for traj in val:
# 					print(traj_dict[traj], traj)
				
# 				print("Next: ")

# 			print(self.merge_list)
			
			indexes=[]
			num_cores=32
			num_trajs=len(self.merge_list)
			
			for i in range(num_cores):
				il = int(num_trajs/num_cores)
				indexes.append([j for j in range(i*il, i*il+il)])

			if indexes[-1][-1]!=num_trajs-1:
				indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])
			
			with Pool() as pool_per_key:
				pool_per_key.starmap( self.get_kmerge_trajs_cost, zip(indexes) )
		
		elif self.scenario=="Civ":
			with open('civ_merge_list.pickle', 'rb') as handle:
				self.merge_list = pickle.load(handle)
			
			indexes=[]
			num_cores=32
			num_trajs=len(self.merge_list)
			
			for i in range(num_cores):
				il = int(num_trajs/num_cores)
				indexes.append([j for j in range(i*il, i*il+il) if j not in self.faulty_civ])

			if indexes[-1][-1]!=num_trajs-1:
				indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs) if j not in self.faulty_civ])
			
# 			print(indexes)
			
			with Pool() as pool_per_key:
				pool_per_key.starmap( self.get_kmerge_trajs_cost, zip(indexes) )
		
		elif self.scenario=="Million":
			with open('million_merge_list.pickle', 'rb') as handle:
				self.merge_list = pickle.load(handle)
			
			indexes=[]
			num_cores=1000
			num_trajs=len(self.merge_list)
			
			#Just a workaround
# 			fl = glob.glob("generalised_million10new/*")
# 			aps = [int(f.split(".")[0].split("_")[-1][1:]) for f in fl]
# # 			print(aps, len(aps))
# # 			print(indexes)
	
# 			with open("miss_trajs10.pickle", 'rb') as handle:
# 				dt = pickle.load(handle)
			
# 			ids=[768, 643, 654, 527, 661, 534, 670, 542, 676, 549, 685, 558, 691, 565, 700, 574, 706, 581, 589, 719, 726, 599, 606, 735, 740, 615, 748, 621, 754, 629, 761, 638]
			
			for i in range(num_cores):
				il = int(num_trajs/num_cores)
# 				if i in ids:
				indexes.append([j for j in range(i*il, i*il+il)])

# 			if indexes[-1][-1]!=num_trajs-1:
# 				indexes.append([j for j in range(indexes[-1][-1]+1, num_trajs)])
			
# 			print(len(indexes), indexes)
			
			with Pool() as pool_per_key:
				pool_per_key.starmap( self.get_kmerge_trajs_cost, zip(indexes) )


# In[163]:


# kaf = KAnonFinal("Shenzen")

# # kaf.sparseCostMatrix(optshow_final.getShenzenCommunities(), ec_final.getShenzenKNN(), ec_final.getFinalMergeCostShenzen())
# # kaf.commonCostMatrix(ec_final.getFinalMergeCostShenzen(), ec_final.getShenzenKNN())
# kaf.kAnonFinal("Shenzen")

# kaf.generaliseTrajectories()


# In[164]:


# kaf = KAnonFinal("Taka")

# # kaf.sparseCostMatrix(optshow_final.getTakaCommunities(), ec_final.getTakaKNN(), ec_final.getFinalMergeCostTaka())
# kaf.commonCostMatrix(ec_final.getFinalMergeCostTaka(), ec_final.getTakaKNN())
# kaf.kAnonFinal("Taka")

# kaf.generaliseTrajectories()


# In[165]:


# kaf2 = KAnonFinal("Senegal")

# # kaf2.sparseCostMatrix(optshow_final.getSenegalCommunities(), ec_final2.getSenegalKNN(), ec_final2.getFinalMergeCostSenegal())
# # kaf2.commonCostMatrix(ec_final2.getFinalMergeCostSenegal(), ec_final2.getSenegalKNN())
# # kaf2.kAnonFinal("Senegal")

# kaf2.generaliseTrajectories()


# In[166]:


# kaf3 = KAnonFinal("Civ")

# # kaf3.sparseCostMatrix(optshow_final.getCivCommunities(), ec_final3.getCivKNN(), ec_final3.getFinalMergeCostCiv())
# # kaf3.commonCostMatrix(ec_final3.getFinalMergeCostCiv(), ec_final3.getCivKNN())
# kaf3.kAnonFinal("Civ")

# kaf3.generaliseTrajectories()


# In[167]:


# count=0

# # cts = ec_final3.getFinalMergeCostCiv()

# # for ct in tqdm(cts):
# # 	if not isinstance(ct, int):
# # 		count=count+1

# # print(count)

# i=0
# for key in cts.keys():
# 	print(key, cts[key])
# 	i=i+1
	
# 	if i>1000:
# 		break


# In[168]:


# kaf4 = KAnonFinal("Million")

# # kaf4.sparseCostMatrix(optshow_final.getMillionCommunities(), ec_final4.getMillionKNN(), ec_final4.getFinalMergeCostMillion())
# # kaf4.commonCostMatrix(ec_final4.getFinalMergeCostMillion(), ec_final4.getMillionKNN())
# kaf4.kAnonFinal("Million")

# # kaf4.generaliseTrajectories()


# In[169]:


# kaf4 = KAnonFinal("Million")

# # kaf4.sparseCostMatrix(optshow_final.getMillionCommunities(), ec_final4.getMillionKNN(), ec_final4.getFinalMergeCostMillion())
# # kaf4.commonCostMatrix(ec_final4.getFinalMergeCostMillion(), ec_final4.getMillionKNN())
# kaf4.kAnonFinal("Million")

# # kaf4.generaliseTrajectories()


# In[170]:


# kaf5 = KAnonFinal("Partial")

# # kaf5.sparseCostMatrix(optshow_final.getPartialCommunities(), ec_final8.getPartialKNN(), ec_final8.getFinalMergeCostPartial())
# # kaf5.commonCostMatrix(ec_final8.getFinalMergeCostPartial(), ec_final8.getPartialKNN())
# # kaf5.kAnonFinal("Partial")

# kaf5.generaliseTrajectories()


# In[171]:


# kaf6 = KAnonFinal("Week")

# # kaf6.sparseCostMatrix(optshow_final.getWeekCommunities(), ec_final7.getWeekKNN(), ec_final7.getFinalMergeCostWeek())
# # kaf6.commonCostMatrix(ec_final7.getFinalMergeCostWeek(), ec_final7.getWeekKNN())
# # kaf6.kAnonFinal("Week")

# kaf6.generaliseTrajectories()


# In[1]:


kaf5 = KAnonFinal("Million")

# kaf5.sparseCostMatrix(optshow_final.getMillionCommunities(), ec_final9.getMillionKNN(), ec_final9.getFinalMergeCostMillion())
# kaf5.commonCostMatrix(ec_final9.getFinalMergeCostMillion(), ec_final9.getMillionKNN())
# kaf5.kAnonFinal("Million")

kaf5.generaliseTrajectories()


# In[ ]:


# optshow.analyseSpatialCosts()
# optshow.sizeDensityRelation()


# In[ ]:


# optshow.shenzenSpatialCosts()


# In[ ]:


# optshow.trajsDakarRandom()
# optshow.trajMaps()
# optshow.gtAssignments()
# optshow.analyseOverallKNN()
# optshow.plotOverallKNN()

# optshow.kLawCost()
# optshow.analyseShenzenKNNSize()
# optshow.analyseShenzenKNNDensity()
# optshow.plotShenzenCosts()


# In[ ]:


# optshow.kLawCostSenegal()
# optshow.analyseSenegalKNNSize()
# optshow.analyseSenegalKNNDensity()
# optshow.plotSenegalCosts()


# In[ ]:


# import pickle
# from tqdm import tqdm

# with open('million_merge_list.pickle', 'rb') as handle:
# 	merge_list1 = pickle.load(handle)

# # with open('senegal_merge_list.pickle', 'rb') as handle:
# # 	merge_list2 = pickle.load(handle)


# In[ ]:


# print(merge_list1[:100], len(merge_list1))


# In[ ]:


# def getDist(p1, p2):
# 	return np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )

# # cst=ec_final4.getFinalMergeCostMillion()
# # knn_m_million=ec_final4.getMillionKNN()

# with open('com_values_senegal_million.pickle', 'rb') as handle:
# 	rcm_array1 = pickle.load(handle)

# with open('com_values.pickle', 'rb') as handle:
# 	rcm_array2 = pickle.load(handle)


# In[ ]:


# md=[]
# # sd=[]

# for i in tqdm(range(300000)):
# 	d1 = getDist(rcm_array1[merge_list1[i][0]], rcm_array1[merge_list1[i][1]])
	
# 	d2 = getDist(rcm_array2[merge_list2[i][0]], rcm_array2[merge_list2[i][1]])
	
# 	md.append(d1 - d2)
# # 	sd.append(d1)


# In[ ]:


# plt.figure(figsize=(12, 12))
# axi = plt.gca()
# #fig, axi = plt.subplots(1, 1)

# sns.histplot(md, binwidth=100)

# # plt.xlabel("Trajs.")
# plt.ylabel("Distance between CMs")

# # plt.ylim([0, 0.15e10])
# # # plt.ylim([0, 0.15e12])
# plt.xlim([-5000, 5000])

# # plt.label()

# plt.savefig("merge_cost_comp.pdf")


# In[ ]:


# count=0
# for val in md:
# 	if val<0:
# 		count=count+1

# print((count/300000)*100)


# In[ ]:


# tdata_full[merge_list1[1][0]]


# In[ ]:


# tdata_full[merge_list2[1][1]]


# In[ ]:


# num_sigletons=0
# for pair in tqdm(merge_list2):
# 	ct1=len(set([val[3] for val in tdata_full[pair[1]]]))
# # 	ct2=len(set([val[3] for val in tdata_full[pair[0]]]))
	
# 	if ct1==1:
# 		num_sigletons=num_sigletons+1
# # 		print(pair)

# print(num_sigletons)


# In[ ]:


# with open('size_test_sa_per_trajMillion2.pickle', 'rb') as handle:
# 	gc1 = pickle.load(handle)

# with open('size_test_sa_per_trajSenegal2.pickle', 'rb') as handle:
# 	gc2 = pickle.load(handle)


# In[ ]:


# print(gc1[1], gc2[1])

# print(merge_list1[2][1], merge_list2[1][1])


# In[ ]:


# with open('knn_matrix_senegal_million500.pickle', 'rb') as handle:
# 	knn_m = pickle.load(handle)

# print("Making full K-merge matrix")

# rows, cols = knn_m.nonzero()


# In[ ]:


# count=0

# for i in tqdm(range(300000)):
# 	if merge_list2[i][1] in cols[i*500: (i+1)*500]:
# 		count=count+1
# 		print(count, merge_list2[i][1])
# 		break

# print(count)


# In[ ]:


# t = ec_final4.getFinalMergeCostMillion()


# In[ ]:


# print(merge_list1[0][1])


# In[ ]:


# vals=[]
# for i in range(500, 1000):
# 	print(cols[i], t[i])
	
# 	vals.append(t[i])

# print(sorted(vals)[:5])


# In[ ]:


# with open('knn_matrix500.pickle', 'rb') as handle:
# 	knn_m = pickle.load(handle)

# print("Making full K-merge matrix")

# rows1, cols1 = knn_m.nonzero()


# In[ ]:


# print(cols1[1000:1500])


# In[ ]:


# with open('million_ccm.pickle', 'rb') as handle:
# 	ccm = pickle.load(handle)


# In[ ]:


# rows, cols = ccm.nonzero()


# In[ ]:


# with open('senegal_ccm.pickle', 'rb') as handle:
# 	ccm2 = pickle.load(handle)


# In[ ]:


# rows2, cols2 = ccm2.nonzero()


# In[ ]:


# with open('partial_ccm100000.pickle', 'rb') as handle:
# 	ccm3 = pickle.load(handle)


# In[ ]:


# rows3, cols3 = ccm3.nonzero()

# dt3= ccm3.data


# In[ ]:


# with open('million_row_min.pickle', 'rb') as handle:
# 	row_min = pickle.load(handle)


# In[ ]:


# with open('singletons_million.pickle', 'rb') as handle:
# 	singletons = pickle.load(handle)


# In[ ]:


# count=0

# for val in tqdm(row_min):
# 	if row_min[val][1]==0:
# 		print(row_min[val])
# 		count=count+1

# print(count)


# In[ ]:


# from tqdm import tqdm

# dt=ccm.data
# dt2=ccm2.data


# In[ ]:


# zros=0
# for data in tqdm(dt2):
# 	if data==0:
# 		zros=zros+1

# print(zros)


# In[ ]:


# import sys

# md=[]
# sd=[]
# sdp1=[]

# count1=0
# count2=0
# count3=0

# for i in tqdm(range(300000)):
# 	d1 = min(dt[i*500:(i+1)*500])
# 	if d1==0 or d1==sys.maxsize:
# 		count1=count1+1
# 	else:
# 		md.append(d1)
	
# 	d2 = min(dt2[i*500:(i+1)*500])
# 	if d2==0:
# 		count2=count2+1
# 	else:
# 		sd.append(d2)
	
# 	if i<100000:
# 		d3 = min(dt3[i*500:(i+1)*500])
# 		if d3==0:
# 			count3=count3+1
# 		else:
# 			sdp1.append(d3)

# print(count1, count2, count3)


# In[ ]:


# plt.figure(figsize=(12, 12))
# axi = plt.gca()
# # fig, axi = plt.subplots(1, 1)

# sns.ecdfplot(md, label="Million")
# sns.ecdfplot(sd, label="300k")
# sns.ecdfplot(sdp1, label="100k")

# # plt.xlabel("Trajs.")
# # plt.ylabel("Distance between CMs")

# # print(md[:500])

# # plt.ylim([0, 0.15e10])
# # # plt.ylim([0, 0.15e12])
# plt.xlim([0, 1e10])

# plt.legend()

# plt.savefig("kmerge_cost_comp.pdf")

