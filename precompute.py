import os
if os.getcwd() == "/gpfs01/berens/user/mchrist":
    os.chdir("./coarse_grain/")

import numpy as np
import pickle
import os
from os.path import join
import sklearn
import matplotlib.pyplot as plt
import helper
import data_loader
import OOP_Multilevel_tsne
import OOP_Connecting
import OOP_Sampling
import metrics
import pandas as pd
import scipy.sparse as sp
import copy
from openTSNE import TSNE
from sklearn.preprocessing import normalize


def load_coarsen_graph(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def load_graphs(start_fname:str,end_fname:str,variation:list):
    graphs = []
    for var in variation:
        graphs.append(load_coarsen_graph(start_fname+str(var)+end_fname))
    return graphs

def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def save_pickle(X,filename):
    with open(filename,"wb") as file:
                pickle.dump(X,file)

"""
data_name = "DIGITS"
dataset_directory = f"./datasets/{data_name}"
directed=False
n=10
reduction = 0.1
seed=42
X,y,class_table = data_loader.select_dataset(kind=data_name,directory=dataset_directory,seed=seed)

dir_plots = join("./plots",data_name) 
dir = join("./results",data_name)

X_knn = sp.csr_array(sklearn.neighbors.kneighbors_graph(X,n_neighbors=n))
#X_knn_undirected = X_knn.T + X_knn
#X_knn_undirected.data = np.array(len(X_knn_undirected.data)*[1])

#laplacian_knn_undirected = -X_knn_undirected 
#laplacian_knn_undirected.setdiag(laplacian_knn_undirected.diagonal() - laplacian_knn_undirected.sum(axis=1))
#X_knn_undirected = normalize(X_knn_undirected,axis=1,norm="l1")

X_knn = sp.csr_array(sklearn.neighbors.kneighbors_graph(X,n_neighbors=n))
X_knn_directed = X_knn

closeness_knn_directed = metrics.harmonic_centrality(W=X_knn_directed)
closeness_knn_directed = metrics.betweenness_centrality(W=X_knn_directed,directed=True)
#closeness_knn_undirected = metrics.closeness_centrality(X_knn_undirected,k=1,cutoff=2)

with open(join(dir,f"{data_name}_closeness_directed.pkl"),"wb") as file:
            pickle.dump(closeness_knn_directed,file)
print("closeness done")
with open(join(dir,f"{data_name}_betweenness_directed.pkl"),"wb") as file:
            pickle.dump(closeness_knn_directed,file)
print("closeness done")
"""
print("Start FMNIST")
data_name = "SWISSROLL"
dataset_directory = f"./datasets/{data_name}"
directed=False
n=10
reduction = 0.1
seed=42
X,y = data_loader.select_dataset(kind=data_name,directory=dataset_directory,seed=seed)

dir_plots = join("./plots",data_name) 
#dir = join("./results",data_name + "2")
dir = join("./results",data_name)

X_knn = sp.csr_array(sklearn.neighbors.kneighbors_graph(X,n_neighbors=n))
X_knn_undirected = X_knn.T + X_knn
X_knn_undirected.data = np.array(len(X_knn_undirected.data)*[1])

laplacian_knn_undirected = -X_knn_undirected 
laplacian_knn_undirected.setdiag(laplacian_knn_undirected.diagonal() - laplacian_knn_undirected.sum(axis=1))
X_knn_undirected = normalize(X_knn_undirected,axis=1,norm="l1")

X_knn = sp.csr_array(sklearn.neighbors.kneighbors_graph(X,n_neighbors=n))
X_knn_directed = X_knn

closeness_knn_directed = metrics.harmonic_centrality(W=X_knn_directed)
betweenness_knn_directed = metrics.betweenness_centrality(W=X_knn_directed,directed=True)
closeness_knn_undirected = metrics.harmonic_centrality(X_knn_undirected)
betweenness_knn_undirected = metrics.betweenness_centrality(X_knn_undirected,directed=False)
eigenvalues_big_undirected = metrics.compute_eigenvalues(X_knn_undirected,n_eigenvals=1000)
save_pickle(X=closeness_knn_directed,filename=join(dir,f"{data_name}_closeness_directed.pkl"))
save_pickle(X=closeness_knn_directed,filename=join(dir,f"{data_name}_betweenness_directed.pkl"))
save_pickle(X=closeness_knn_directed,filename=join(dir,f"{data_name}_closeness.pkl"))
save_pickle(X=closeness_knn_directed,filename=join(dir,f"{data_name}_betweenness.pkl"))
save_pickle(X=closeness_knn_directed,filename=join(dir,f"{data_name}_eigenvalues_big_undirected.pkl"))
print("SWISSROLL done")

print("Start DNA")
data_name = "DNA"
dataset_directory = f"./datasets/{data_name}"
directed=False
n=10
reduction = 0.1
seed=42
X,y = data_loader.select_dataset(kind=data_name,directory=dataset_directory,seed=seed)
#X,y,class_table = data_loader.select_dataset(kind=data_name,directory=dataset_directory,seed=seed)

dir_plots = join("./plots",data_name) 
#dir = join("./results",data_name + "2")
dir = join("./results",data_name)

X_knn = sp.csr_array(sklearn.neighbors.kneighbors_graph(X,n_neighbors=n))
X_knn_undirected = X_knn.T + X_knn
X_knn_undirected.data = np.array(len(X_knn_undirected.data)*[1])

laplacian_knn_undirected = -X_knn_undirected 
laplacian_knn_undirected.setdiag(laplacian_knn_undirected.diagonal() - laplacian_knn_undirected.sum(axis=1))
X_knn_undirected = normalize(X_knn_undirected,axis=1,norm="l1")

X_knn = sp.csr_array(sklearn.neighbors.kneighbors_graph(X,n_neighbors=n))
X_knn_directed = X_knn

closeness_knn_directed = metrics.harmonic_centrality(W=X_knn_directed)
betweenness_knn_directed = metrics.betweenness_centrality(W=X_knn_directed,directed=True)
closeness_knn_undirected = metrics.harmonic_centrality(X_knn_undirected)
betweenness_knn_undirected = metrics.betweenness_centrality(X_knn_undirected,directed=False)
eigenvalues_big_undirected = metrics.compute_eigenvalues(X_knn_undirected,n_eigenvals=1000)
save_pickle(X=closeness_knn_directed,filename=join(dir,f"{data_name}_closeness_directed.pkl"))
save_pickle(X=closeness_knn_directed,filename=join(dir,f"{data_name}_betweenness_directed.pkl"))
save_pickle(X=closeness_knn_directed,filename=join(dir,f"{data_name}_closeness.pkl"))
save_pickle(X=closeness_knn_directed,filename=join(dir,f"{data_name}_betweenness.pkl"))
save_pickle(X=closeness_knn_directed,filename=join(dir,f"{data_name}_eigenvalues_big_undirected.pkl"))
print("DNA done")