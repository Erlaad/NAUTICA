import networkx as nx
import numpy as np
from datetime import datetime

def print_time(current_time):
	new_time = datetime.now()
	print(f"Done in {(new_time-current_time).total_seconds()} seconds.\n")
	return new_time

def give_weight(n12,label,w_table):
    column = 1 if label == 'COOP' else 0
    row = 10 if n12 >= 10 else n12
    #print(n12,label,np.floor(weights[row,column]/min(weights[:,column])))
    return np.floor(w_table[row,column]/min(w_table[:,column]))

def build_couples(L):
	# minimethod to generate all couples from a list L
	for i1,item1 in enumerate(L[:-1]):
		for item2 in L[i1+1:]:
			yield (item1,item2)

def nautica_predictor(N12,biogrid,inter,t_h,t_l):
    if inter == -1:
        pred = 'NA'
    elif (inter == 1 and biogrid) or (inter == 1 and not biogrid and N12 >= t_l) or (inter == 0 and biogrid and N12 >= t_l):
        pred = 'COOP'
    elif (inter  == 1 and not biogrid and 0 < N12 < t_l) or (inter  == 0 and not biogrid and N12 >= t_h):
        pred = 'COMP'
    else:
        pred = 'NINT'
    return pred

# class Interaction:
#     def __init__(self,prot1,prot2):
#         self.p1 = prot1 if prot1 <= prot2 else prot2
#         self.p2 = prot2 if prot1 <= prot2 else prot1
    
#     def __str__(self):
#         return '{},{}'.format(self.p1,self.p2)

#     def get_interactors(self):
#         return (self.p1,self.p2)

class Network:
	def __init__(self,i_name,i_prots,i_inter):
		self.name = i_name
		self.proteins = i_prots
		self.interactions = i_inter  # list of interactions

	def get_tfs(self,tf_list):
		self.tfs = [p for p in self.proteins if p in tf_list]
		self.tfinteractions = [i for i in self.interactions if i[0] in self.tfs and i[1] in self.tfs]
		return (self.tfs, self.tfinteractions)

	def build_graph(self):
		self.graph = nx.Graph()
		self.graph.add_edges_from(self.interactions)
		self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
		return self.graph

	def __str__(self):
		return(f"Name: {self.name}\nNumber of proteins: {len(self.proteins)} "
			f"(of which {len(self.tfs)} are TFs)\n"
			f"Number of edges: {len(self.interactions)} "
			f"(of which {len(self.tfinteractions)} are betweeen TFs).")

class AlgoNetwork(Network):
	def __init__(self,i_name,i_prots,i_inter):
		# re-written for clarity
		self.name = i_name
		self.proteins = i_prots
		self.interactions = i_inter  # dictionary of Interactions

	def build_graph(self):
		self.graph = nx.Graph()
		self.graph.add_edges_from([i for i in self.interactions if self.interactions[i] == 1])
		self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
		return self.graph
