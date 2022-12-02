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

# dbn = DBN()

# # dbn.add_edge(('209',0),('209',1))
# dbn.add_edges_from([(('209',0),('209',1)),(('209',0),('10',1)), (('10',0),('10',1)), (('10', 0),('209',1)) ])

# cpd1 = TabularCPD(('209', 1), 2, [[0.1], [0.9]])
# cpd2 = TabularCPD(('10', 1), 2, [[0.1], [0.9]])
# cpd3 = TabularCPD(('209', 0), 2, [[.4, .1,.4,.1],[.1,.4,.25,.25]], evidence=[('209',1), ('10',1)], evidence_card=[2,2])
# cpd4 = TabularCPD(('10', 0), 2, [[.4,.1,.3,.2],[.1,.5,.2,.2]], evidence=[('209',1), ('10',1)], evidence_card=[2,2])

# dbn.add_cpds(cpd1, cpd2, cpd3, cpd4)

# print(dbn.nodes())
# print(dbn.edges())
# print(dbn.get_cpds())


# model = DBN(
    # [
    #     (("A", 0), ("B", 0)),
    #     (("A", 0), ("C", 0)),
    #     (("B", 0), ("D", 0)),
    #     (("C", 0), ("D", 0)),
    #     (("A", 0), ("A", 1)),
    #     (("A", 0), ("B", 1)),
    #     (("A", 0), ("C", 1)),
    #     (("A", 0), ("D", 1)),
    #     (("B", 0), ("B", 1)),
    #     (("C", 0), ("C", 1)),
    #     (("D", 0), ("D", 1)),
    #     (("C", 0), ("A", 1)),
    #     (("C", 0), ("B", 1)),
    #     (("A", 1), ("B", 1)),
    #     (("A", 1), ("C", 1)),
    #     (("B", 1), ("D", 1)),
    #     (("C", 1), ("D", 1)),
    #     (("A", 1), ("A", 2)),
    #     (("B", 1), ("B", 2)),
    #     (("C", 1), ("C", 2)),
    #     (("D", 1), ("D", 2)),
    #     (("A", 2), ("B", 2)),
    #     (("A", 2), ("C", 2)),
    #     (("B", 2), ("D", 2)),
    #     (("C", 2), ("D", 2)),
    #     (("A", 2), ("A", 3)),
    #     (("B", 2), ("B", 3)),
    #     (("C", 2), ("C", 3)),
    #     (("D", 2), ("D", 3)),
    #     (("A", 3), ("B", 3)),
    #     (("A", 3), ("C", 3)),
    #     (("B", 3), ("D", 3)),
    #     (("C", 3), ("D", 3)),
    #     (("A", 3), ("A", 4)),
    #     (("B", 3), ("B", 4)),
    #     (("C", 3), ("C", 4)),
    #     (("D", 3), ("D", 4)),
    #     (("A", 4), ("B", 4)),
    #     (("A", 4), ("C", 4)),
    #     (("B", 4), ("D", 4)),
    #     (("C", 4), ("D", 4))
    # ]
# )


# model = DBN()

# model.add_nodes_from(['A', 'B', 'C', 'D'])
# model.add_edges_from([
#     (("A", 0), ("B", 0)),
#     (("A", 0), ("A", 1)),
#     (("A", 1), ("B", 2)),
#     (("B", 2), ("C", 3)),
#     (("C", 3), ("D", 4)),
#     (("A", 0), ("A", 1)),
#     (("A", 0), ("B", 1)),
#     (("A", 0), ("C", 1)),
#     (("A", 0), ("D", 1)),
#     (("B", 0), ("B", 1)),
#     (("C", 0), ("C", 1)),
#     (("D", 0), ("D", 1)),
#     (("C", 0), ("A", 1)),
#     (("C", 0), ("B", 1)),
#     (("A", 1), ("B", 1)),
#     (("A", 1), ("C", 1)),
#     (("B", 1), ("D", 1)),
#     (("C", 1), ("D", 1)),
#     (("A", 1), ("A", 2)),
#     (("B", 1), ("B", 2)),
#     (("C", 1), ("C", 2)),
#     (("D", 1), ("D", 2)),
#     (("A", 2), ("B", 2)),
#     (("A", 2), ("C", 2)),
#     (("B", 2), ("D", 2)),
#     (("C", 2), ("D", 2)),
#     (("A", 2), ("A", 3)),
#     (("B", 2), ("B", 3)),
#     (("C", 2), ("C", 3)),
#     (("D", 2), ("D", 3)),
#     (("A", 3), ("B", 3)),
#     (("A", 3), ("C", 3)),
#     (("B", 3), ("D", 3)),
#     (("C", 3), ("D", 3)),
#     (("A", 3), ("A", 4)),
#     (("B", 3), ("B", 4)),
#     (("C", 3), ("C", 4)),
#     (("D", 3), ("D", 4)),
#     (("A", 4), ("B", 4)),
#     (("A", 4), ("C", 4)),
#     (("B", 4), ("D", 4)),
#     (("C", 4), ("D", 4))
# ])
# data = np.random.randint(low=0, high=5, size=(1000, 20))
# # print(len(data))
# # print(data)
# colnames = []
# for t in range(5):
#     colnames.extend([("A", t), ("B", t), ("C", t), ("D", t)])
# df = pd.DataFrame(data, columns=colnames)
# print(colnames)
# print(df)
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

model = DBN(
    [
        (("A", 0), ("B", 0)),
        (("A", 0), ("C", 0)),
        (("B", 0), ("D", 0)),
        (("C", 0), ("D", 0)),
        (("A", 0), ("A", 1)),
        (("B", 0), ("B", 1)),
        (("C", 0), ("C", 1)),
        (("D", 0), ("D", 1)),
    ]
)
data = np.random.randint(low=0, high=2, size=(1000, 20))
colnames = []
for t in range(5):
    colnames.extend([("A", t), ("B", t), ("C", t), ("D", t)])
df = pd.DataFrame(data, columns=colnames)
model.fit(df)
print(model.simulate(n_samples=10, n_time_slices=5))
# print(model.fit(df))