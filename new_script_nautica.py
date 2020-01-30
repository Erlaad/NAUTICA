# 0 - Import statements
import csv
import matplotlib.pyplot as plt
# import graphviz
# import matplotlib.pyplot as plt
# import networkx as nx
import numpy as np
# import os
import pandas as pd
# # import random
# # random.seed(0)
# import scipy.stats as sts
# import seaborn as sb
# import time
# import urllib
# import xml.etree.ElementTree as ET
# # import seaborn as sb

from utilities import *
# # 
# from functools import partial
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
# from sklearn import tree
# from sklearn.decomposition import PCA
from tqdm import tqdm
from datetime import datetime
from sys import argv

_, recompute = argv

print("Begin execution.\n")
tstart = datetime.now()

# 1 - Global variables and ancillary methods definition

BIOGRID_FILE = 'data/BIOGRID-MV-Physical-3.4.162.tab2.txt'
# CORUM_FILE = 'data/allComplexes.txt'
TAXON_NMBR = 9606  # yeast: 559292
TF_FILE = 'data/human_tfs.tsv'
OUTPUT_NULL_FILE = 'data/nautica_null.csv'
PRED_CSV_FILE = 'data/tica_03/TICA_predictions.csv'
# MEDLINE_FLDR = 'data/medline/'
TR = 'data/validation/tr.csv'
# #TR = 'data/validation/megaTR.csv'
# TS1 = 'data/validation/ts1.csv'
# TS2 = 'data/validation/ts2.csv'
TS = 'data/validation/ts.csv'
# # CENTDIST_FILE = 'data/centdist_pos.csv'
# # PROCESSED_TRRUST_FILE = 'data/trrust_interactions.csv'

# # HGEOM_THRESHOLD = 0.01
MIN_DEGREE = 3
T_H = 8
T_L = 5

# def cm2inch(*tupl):
#     inch = 2.54
#     if isinstance(tupl[0], tuple):
#         return tuple(i/inch for i in tupl[0])
#     else:
#         return tuple(i/inch for i in tupl)



# class TF_Interaction(Interaction):
#     def __init__(self,prot1,prot2,i_stats,i_tests):
#         Interaction.__init__(self,prot1,prot2)
#         self.stats = i_stats
#         self.tests = i_tests
#         self.passed = int(sum([1 if i == 'Passed' else 0 for i in self.tests])>=3)
    
#     def __str__(self):
#         return '{},{},{},{},{}'.format(self.p1,self.p2,
#                                       ','.join([str(i) for i in self.stats]),
#                                       ','.join(self.tests),
#                                       self.passed)

#     def get_interactors(self):
#         return (self.p1,self.p2)
    
#     def get_status(self):
        # return self.passed

print("Loading and parsing BioGRID...\n")



biogrid_interactions = []
with open(BIOGRID_FILE, 'r') as biogrid_infile:
    bg_reader = csv.reader(biogrid_infile, delimiter='\t')
    for row in bg_reader:
        if row[15] == str(TAXON_NMBR) and row[16] == str(TAXON_NMBR):
            biogrid_interactions.append((row[7], row[8]) if row[7] <= row[8] else (row[8],row[7]))  # Int1, Int2, PID

bg_prots = sorted(list(set([i[0] for i in biogrid_interactions]).union(set([i[1] for i in biogrid_interactions]))))

biogrid = Network("BioGRID",bg_prots,biogrid_interactions)

print("Loading TF LIST...\n")
tfs = set()
with open(TF_FILE, 'r') as tf_infile:
    tf_reader = csv.reader(tf_infile, delimiter='\t')
    for row in tf_reader:
        tfs.add(row[0])

biogrid.get_tfs(tfs)
biogrid.build_graph()

print(biogrid)
midtime = print_time(tstart)

print("Building TF interaction predictions...\n")

# Here, "algo" represents either TICA or 
# any other interaction prediction algorithm

tf_algo_predictions = dict()
with open(PRED_CSV_FILE, 'r') as pred_file:
    pred_f_reader = csv.reader(pred_file)
    for row in pred_f_reader:
        i = (row[0], row[1]) if row[0] <= row[1] else (row[1],row[0])
        if i not in tf_algo_predictions.keys():
            tf_algo_predictions[i] = 1 if row[-1] == 'True' else 0
        else:
            tf_algo_predictions[i] = 1 if row[-1] == 'True' or tf_algo_predictions[i] == 1 else 0

tf_algo_prots = sorted(list(set([i[0] for i in tf_algo_predictions]).union(set([i[1] for i in tf_algo_predictions]))))

algo = AlgoNetwork("TICA_30",tf_algo_prots,tf_algo_predictions)  # change accordingly
algo.get_tfs(tfs)
algo.build_graph()
print(algo)

midtime = print_time(midtime)

# useful functions:
# all_neighbors(graph, node)    Returns all of the neighbors of a node in the graph.
# non_neighbors(graph, node)    Returns the non-neighbors of the node in the graph.
# common_neighbors(G, u, v) Return the common neighbors of two nodes in a graph.
# info(G[, n])  Print short summary of information for the graph G or the node n.

## Build a Network object using nautica
null_header = ['P1','P2','N_12','Is_Biogrid','INT_pred','PREDICTION']
if recompute.upper() == 'Y':
    print("Computing NAUTICA predictions...\n")
    with open(OUTPUT_NULL_FILE, 'w') as ofile:
        nautica_writer = csv.writer(ofile, delimiter=',',
                                quotechar='#')
        nautica_writer.writerow(null_header)
        L = len(biogrid.proteins)  # for easier reading
        for p1,p2 in tqdm(build_couples(biogrid.proteins),total=int(L*(L-1)/2)):
            p1_neighbors = set(biogrid.graph[p1])
            p2_neighbors = set(biogrid.graph[p2])
            if len(p1_neighbors) >= MIN_DEGREE and len(p2_neighbors) >= MIN_DEGREE:
                n_p1p2 = len(p1_neighbors.intersection(p2_neighbors))
                # s_prot_net = biogrid.graph.subgraph(nx.common_neighbors(biogrid.graph,p1,p2))
                # density = 2*len(biogrid.graph.subgraph(p1p2_neighbors).edges) / (n_p1p2 * (n_p1p2 -1)) if n_p1p2 >1 else -1 # Density of a network of  n nodes is defined by # of observed edges divided (n *(n-1) / 2).
                #print(density)
                is_in_biogrid = 1 if (p1,p2) in biogrid.interactions else 0
                is_predicted_by_algo = -1 if (p1,p2) not in algo.interactions else algo.interactions[(p1,p2)]
                nautica_result = nautica_predictor(n_p1p2,is_in_biogrid,is_predicted_by_algo,T_H,T_L)
                nautica_writer.writerow([p1,p2,
                    n_p1p2,
                    is_in_biogrid, # this is the biogrid edge
                    is_predicted_by_algo,
                    nautica_result]
                    )
    midtime = print_time(midtime)

print("Loading NAUTICA prediction, TR and TS datasets...")
nautica_null = pd.read_csv(OUTPUT_NULL_FILE,sep=',',header=0,
    names=null_header,index_col=False,low_memory=False)
nautica_null = nautica_null[~nautica_null['PREDICTION'].isna()]
tr_dataset = pd.read_csv(TR, sep=',',index_col=False,header=0)
tr_features = pd.merge(tr_dataset,nautica_null,left_on=['TF1','TF2'],right_on=['P1','P2']).drop(['P1','P2'],axis=1)
ts_dataset = pd.read_csv(TS, sep=',',index_col=False,header=0)
ts_features = pd.merge(ts_dataset,nautica_null,left_on=['TF1','TF2'],right_on=['P1','P2']).drop(['P1','P2'],axis=1)

print(tr_features.head())

midtime = print_time(midtime)

print("Computing NULL and COOP weights...")

# FULL NULL
x_labels = ['0','1','2','3','4','5','6','7','8','9','10+']
weights = np.zeros((11,2))
null_lengths = np.array([len(nautica_null[(nautica_null['N_12'] == 0)]),
           len(nautica_null[(nautica_null['N_12'] == 1)]),
           len(nautica_null[(nautica_null['N_12'] == 2)]),
           len(nautica_null[(nautica_null['N_12'] == 3)]),
           len(nautica_null[(nautica_null['N_12'] == 4)]),
           len(nautica_null[(nautica_null['N_12'] == 5)]),
           len(nautica_null[(nautica_null['N_12'] == 6)]),
           len(nautica_null[(nautica_null['N_12'] == 7)]),
           len(nautica_null[(nautica_null['N_12'] == 8)]),
           len(nautica_null[(nautica_null['N_12'] == 9)]),
           len(nautica_null[(nautica_null['N_12'] >= 10)]),
          ]
                  )
weights[:,0] = null_lengths/min(null_lengths)

fig,axes = plt.subplots(1,1,figsize=(12,8))
axes.set_title("Figure 5 - NINT bin sizes")
axes.set_xlabel("$N_{12}$")
axes.set_ylabel("Bin_size (bin_weight)")
rects = axes.bar(range(0,11),null_lengths,color='grey',
	tick_label=x_labels)
for idx,rect in enumerate(rects):
    height = rect.get_height()
    axes.annotate(f'{weights[idx,0]:5.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')
fig.tight_layout()
plt.savefig("figure5.eps")
plt.close()

# COOP ONLY
coop_lengths = np.array([len(tr_features[(tr_features['N_12'] == 0) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] == 1) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] == 2) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] == 3) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] == 4) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] == 5) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] == 6) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] == 7) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] == 8) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] == 9) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] >= 10) & (tr_features['LABEL'] == 'COOP')]),
          ]
                  )
weights[:,1] = coop_lengths/min(coop_lengths)

fig,axes = plt.subplots(1,1,figsize=(12,8))
axes.set_title("Figure 6 - COOP bin sizes")
axes.set_xlabel("$N_{12}$")
axes.set_ylabel("Bin_size (bin_weight)")
rects = axes.bar(range(0,11),coop_lengths,color='green',
	tick_label=x_labels)
for idx,rect in enumerate(rects):
    height = rect.get_height()
    axes.annotate(f'{weights[idx,1]:5.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')
fig.tight_layout()
plt.savefig("figure6.eps")
plt.close()

# Assign weights
ts_features['WEIGHTS'] = ts_features.apply(lambda row:give_weight(row['N_12'],row['LABEL'],weights),axis=1)

print(ts_features)

midtime = print_time(midtime)

print("Computing recall for the different methods...")

for l in ['COOP','COMP','NINT']:
    print(l)
    this_label = ts_features[ts_features['LABEL']==l]
    print(f"Number of {l} cases in TS: {len(this_label)}")
    if l=='NINT':
        nautica_tp = len(this_label[this_label['PREDICTION']==l])
        tica_tp = len(this_label[this_label['INT_pred']==0])
    else:
        nautica_tp = len(this_label[this_label['PREDICTION']!='NINT'])
        tica_tp = len(this_label[this_label['INT_pred']==1])
    print('NAUTICA TP:',nautica_tp)
    print('NAUTICA RECALL:',nautica_tp*100/len(this_label))
    print('TICA RECALL:',tica_tp)
    print('TICA RECALL:',tica_tp*100/len(this_label))
    print()
    tot_cases = sum(this_label['WEIGHTS'])
    print(f"Weighted number of {l} cases in TS: {tot_cases}")
    if l == 'NINT':
        nautica_w_tp = sum(this_label[this_label['PREDICTION']==l]['WEIGHTS'])
        tica_w_tp = sum(this_label[this_label['INT_pred']==0]['WEIGHTS'])
    else:
        nautica_w_tp = sum(this_label[this_label['PREDICTION']!='NINT']['WEIGHTS'])
        tica_w_tp = sum(this_label[this_label['INT_pred']==1]['WEIGHTS'])
    #print(this_label[this_label['NAUTICA']==l]['WEIGHTS'])
    print('NAUTICA WEIGHTED TP:',nautica_w_tp)
    print('NAUTICA WEIGHTED RECALL:',nautica_w_tp*100/tot_cases)
    print('TICA WEIGHTED TP:',tica_w_tp)
    print('TICA WEIGHTED RECALL:',tica_w_tp*100/tot_cases)

midtime = print_time(midtime)

print("Script finished! :)")
tend = print_time(tstart)

