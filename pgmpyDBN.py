#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 07:14:29 2022

@author: bkamos
"""

from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.inference import DBNInference
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

import numpy as np
import pandas as pd

### Add all nodes at all time slices
### Add all edges from all notes across two time slices... (will need to look into that, i think it will just be a full connected DBN, next attempt would be Sparcc outputs above threshold as edges)
### Create Conditional Probability Matrix - still unsure on what exactly this is - I think that it's a matrix with probabilities from two time steps ago
### Could make a CPD table dynammically by pushing all the isolates into time vectors at a single scale and using that as the reference for the predecessor function
### 

data = pd.read_csv('/Users/bkamos/Documents/GitHub/InRoot/data/RelativeAbundance_Filtered_ordered.csv')
taxa = pd.read_csv('/Users/bkamos/Documents/GitHub/InRoot/data/Taxa_New_ISOTAXA.csv')
meta = pd.read_csv('/Users/bkamos/Documents/GitHub/InRoot/data/ES_Meta.csv')

print(data)
isolates = data.iloc[:,0]
print(isolates)

days1 = []
batches1 = []
for cols in data.columns:
    if cols.startswith('DPI'):
        components = cols.split("_")
        # print(components)
        batches = components[2]
        days = components[1]
        days1.append(days)
        batches1.append(batches)
        # print(batches)
        # print(days)
        # print(cols)
    else:
        pass
    
timePoints = set(days1)
batches = set(batches1)

colNames = []
for i in range(len(timePoints)):
    for j in isolates:
        # print((j,i))
        colNames.append((j, i))
        
### Split frame based on lenth of timepoints
### flatten each batch
### Bring batches back together
### Create dataframe with samples by isolate by time
### Create Arcs between t and t+1
### Might need to discretize my relative abundance values
### Fit data to model
### Use seaborn to Visualize data with relAbundance graphs
# for i in range(len(data.index)):
    # data.iloc[i].append(data.iloc[i+1])
# for i in colNames:
    # print(i)
# isolate_names = np.array(isolates)
# print(isolate_names)
isolate_table = data.values # abundance matrix
# print(isolate_table)
isolate_table[isolate_table==0] = 1e-6 # bodge to prevent division by 0
isolate_table = isolate_table.T
print(isolate_table)
# print(isolate_table)
# print(isolate_table.shape)
# print(colNames)

# print(len(timePoints))

# import numpy as np
# import pandas as pd
# from hmmlearn import hmm
# import re
# from skbio.stats.composition import clr
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.colors import LogNorm, Normalize
#
# df = pd.read_csv('../data/Timecourse/TC_data_FL.tsv',index_col=0,delimiter='\t')
# isolate_df = df.filter(like='ES',axis='columns') # select only endosphere samples
# int_ts_idx = [int(re.sub("[^0-9]", "", s)) for s in isolate_df.columns] # turn index into dpi and replicate concatenated
# isolate_df = isolate_df.rename(columns={o:n for o,n in zip(isolate_df.columns,int_ts_idx)}) # apply new index to dataframe
# isolate_df = isolate_df.reindex(sorted(isolate_df.columns), axis=1) # sort by days post incoulation
# isolate_df = isolate_df.loc[:, ~isolate_df.columns.isin([142,143,144])] # remove 14dpi timepoint (has 3 replicates instead of 4)
# replicate = np.array(isolate_df.columns)%10 # replicate number of each column
# dpi = np.round(isolate_df.columns,decimals=-1)//10 # days post inoculation of each column
# isolate_df = isolate_df.iloc[np.array(np.sum(isolate_df,axis=1)>0),:] # remove rows with all zero values
# isolate_names = np.array(isolate_df.index)
# isolate_table = isolate_df.values # abundance matrix
# isolate_table[isolate_table==0] = 1e-6 # bodge to prevent division by 0
# isolate_table = isolate_table.T # flipping so rows = samples, columns = features
# isolate_clr = clr(isolate_table) # centered log ratio

# timePoints = list(timePoints)
# batches = list(batches)

# tPS = timePoints.sort()
# batchesSorted = batches.sort()

# print(tPS)
# print(batchesSorted)


             
# print(timePoints)
# print(batches)
    # batches = cols[-3:]
    # Days = 
        
# model = DBN(
#     [
#         (("A", 0), ("B", 0)),
#         (("A", 0), ("C", 0)),
#         (("B", 0), ("D", 0)),
#         (("C", 0), ("D", 0)),
#         (("A", 0), ("A", 1)),
#         (("B", 0), ("B", 1)),
#         (("C", 0), ("C", 1)),
#         (("D", 0), ("D", 1)),
#     ]
# )
# data = np.random.randint(low=0, high=2, size=(1000, 20))
# colnames = []
# for t in range(5):
#     colnames.extend([("A", t), ("B", t), ("C", t), ("D", t)])
# df = pd.DataFrame(data, columns=colnames)
# print(df)
# model.fit(df)
# print(model.simulate(n_samples=10, n_time_slices=5))
# print(model.fit(df))

# model.fit(df)
# # print(model.fit(df))
# # print(model.initialize_initial_state())
# print(model.check_model())
# print(model.get_cpds())
# print(model.edges())
# # print(model.get_immoralities())
# # print(model.get_inter_edges())
# # print(model.get_independencies())
# # print(model.get_leaves())
# # print(model.get_slice_nodes(3))
# # print(model.get_parents(node=('A', 1)))
# # print(model.is_multigraph())
# print(model.size())
# # print(model.get_cpds())
# # infer = VariableElimination(model)
# # g_dist = infer.query(('A', 0))
# # print(g_dist)

# # model.add_cpds(model.fit(df))

# print(DBNInference(model))
# print(model.simulate(n_samples=10, n_time_slices=3))