
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
import plotting_graphs as plg
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

data_name = "TASIC"
dataset_directory = f"./datasets/{data_name}"
max_eigenvalues = 1000
n=10
reduction = 0.1
seed=42
X,y,class_table = data_loader.select_dataset(kind=data_name,directory=dataset_directory,seed=seed)

def get_data(data_name,seed=seed,give_directed=False):
    dataset_directory = f"./datasets/{data_name}"
    data = data_loader.select_dataset(kind=data_name,directory=dataset_directory,seed=seed)
    dir = join("./results",data_name)
    X_knn = sp.csr_array(sklearn.neighbors.kneighbors_graph(X,n_neighbors=n))
    X_knn_undirected = X_knn.T + X_knn
    X_knn_undirected.data = np.array(len(X_knn_undirected.data)*[1])

    laplacian_knn_undirected = -X_knn_undirected 
    laplacian_knn_undirected.setdiag(laplacian_knn_undirected.diagonal() - laplacian_knn_undirected.sum(axis=1))
    X_knn_undirected = normalize(X_knn_undirected,axis=1,norm="l1")
    undirected = X_knn_undirected,laplacian_knn_undirected

    if give_directed:
        X_knn_directed = X_knn    
        laplacian_knn_directed = -X_knn_directed 
        laplacian_knn_directed.setdiag(laplacian_knn_directed.diagonal() - laplacian_knn_directed.sum(axis=1))
        X_knn_directed = normalize(X_knn_directed,axis=1,norm="l1")
        directed = X_knn_directed,laplacian_knn_directed
        return data,dir,undirected,directed

    return data,dir,undirected,directed


def get_metric_if_exist(fname,function,dir=dir,**kwargs):
    if os.path.isfile(join(dir,fname)):
        print("found_file")
        return load_from_pickle(join(dir,fname))
    else:
        print(f"File not found: {fname}, create metric")
        output = function(**kwargs)
        save_pickle(output,join(dir,fname))
        return output

def get_metrics_if_exist(fname,functions,dir=dir,**kwargs):
    if os.path.isfile(join(dir,fname)):
        print("found_file")
        return load_from_pickle(join(dir,fname))
    else:
        print(f"File not found: {fname}, create metric")
        out = []
        for function in functions:
            output = function(**kwargs)
            out.append(output)
        save_pickle(out,join(dir,fname))
        return output


def get_spectral_graph_metrics(graph,eigenvalues_big,**kwargs):
    W = graph.get_T()
    graph_laplacian = -W
    graph_laplacian.setdiag(graph_laplacian.diagonal() + np.ones(graph_laplacian.shape[0]))
    eigenvalues_small = metrics.compute_eigenvalues(graph_laplacian,n_eigenvals=1000)
    d_spectral = metrics.spectral_graph_distance(eigenvalues_big,eigenvalues_small)[0]
    rel_eigval_e = metrics.relative_eigenvalue_error(eigenvalues_big,eigenvalues_small)[0]
    return d_spectral,rel_eigval_e


def get_graph_metrics(graph,closeness_knn,betweenness_knn,**kwargs):
    W = graph.get_T()
    graph_close_centrality = metrics.harmonic_centrality(W,k=10000)
    graph_between_centrality = metrics.betweenness_centrality(W,directed=True,k=10000)
    close = metrics.KL_Div(closeness_knn,graph_close_centrality)
    between = metrics.KL_Div(betweenness_knn,graph_between_centrality)
    return close,between


def get_embedding_metrics(graph,X,y,**kwargs):
    TSNE_emb = graph.TSNE
    current_landmarks = graph.find_corresponding_landmarks_at_level()
    f_cluster = y[current_landmarks]
    knn_acc=metrics.knn_acc(X[current_landmarks],TSNE_emb,k=10)
    trust=metrics.trustworthiness(X[current_landmarks],TSNE_emb,k=10)
    silh=metrics.silhouette_score(TSNE_emb,f_cluster)[0]
    dbi=metrics.Davies_bouldin_index(TSNE_emb,f_cluster)
    return knn_acc,trust,silh,dbi



# %%

(X,y,class_table), dir, (W_udir,L_udir),(W_dir,L_dir) = get_data(data_name,give_directed=True)


# %%
def load_file_if_exists(fname,dir=dir):
    if os.path.isfile(join(dir,fname)):
        print("found_file")
        return load_from_pickle(join(dir,fname))
    else:
        print(f"File not found: {fname}")


def create_graph(sampling,connection,X,y,filename,dir=dir,level=2,seed=seed,directed=False,discrete=True):
    if os.path.isfile(join(dir,filename)):
        print("found_file")
        graph = load_from_pickle(join(dir,filename))
    else:
        print(f"no file found {join(dir,filename)}")
        graph_1 = OOP_Multilevel_tsne.KNNGraph(data=X,labels=y,n=n,data_name=data_name,directed=directed,weighted=True,landmark_sampling=copy.deepcopy(sampling),connection=copy.deepcopy(connection),discrete_labels=discrete,seed=seed)
        graph_1.TSNE_to_attribute(random_init=False)
        if level!=1:
            graph_2 = graph_1.create_new_level()
            graph_2.TSNE_to_attribute(random_init=False)
            if level!=2:
                graph=graph_2.create_new_level()
                graph.TSNE_to_attribute(random_init=False)
            else:
                graph=graph_2
        else:
            graph = graph_1
        save_pickle(graph,join(dir,filename))
    return graph



# %%
sampling_rand = OOP_Sampling.RandomSampling(seed=seed,shrinkage=reduction)
sampling_rw = OOP_Sampling.RandomWalksSampling(n_walks=100,shrinkage=reduction,seed=seed)
sampling_hubs = OOP_Sampling.HighestDegreeSampling(shrinkage=reduction)
sampling_hbn = OOP_Sampling.HighDegreeExclusionNN(shrinkage=reduction,k=10)
connection_slrw = OOP_Connecting.StateToLandmarksRandomWalks(n_walks=100,W=np.ones(X.shape[0]))
connection_exact = OOP_Connecting.StateToLandmarksExact(W=np.ones(X.shape[0]) ,use_gambler=True,threshold_I=1e-4,threshold_T=1e-4)
# %%
#TODO: run
directed_slrw_rw = create_graph(X=X,y=y,sampling=sampling_rw,connection=connection_slrw,filename="directed-slrw-rw.pkl",dir=dir,directed=True)
print("directed_slrw_rw")
directed_slrw_hubs = create_graph(X=X,y=y,sampling=sampling_hubs,connection=connection_slrw,filename="directed-slrw-hubs.pkl",dir=dir,directed=True)
print("directed_slrw_hubs")
directed_slrw_random = create_graph(X=X,y=y,sampling=sampling_rand,connection=connection_slrw,filename="directed-slrw-random.pkl",dir=dir,directed=True)
print("directed_slrw_random")
directed_slrw_hbn = create_graph(X=X,y=y,sampling=sampling_hbn,connection=connection_slrw,filename="directed-slrw-hbn.pkl",dir=dir,directed=True)
print("directed_slrw_hbn")
directed_slrw_hbn = create_graph(X=X,y=y,sampling=sampling_hubs,connection=connection_exact,filename="directed-exact-hubs.pkl",dir=dir,directed=True)
print("directed_slrw_exact")
# %%
