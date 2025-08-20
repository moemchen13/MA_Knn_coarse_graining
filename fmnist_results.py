
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

data_name = "FMNIST"

#colors = plt.cm.tab20.colors
colors = plt.cm.tab10.colors
dataset_directory = f"./datasets/{data_name}"
#colors = np.load(join(dataset_directory,"tasic-colors.npy"))

#cmap = plt.get_cmap('coolwarm')
#colors = [cmap(i) for i in [0.1,0.15,0.95,0.9,0.85,0.8,0.75,0.7,0.2,0.25]]

max_eigenvalues = 1000
n=10
reduction = 0.1
seed=42
X,y,class_table = data_loader.select_dataset(kind=data_name,directory=dataset_directory,seed=seed)
print(type(X))
print(X.shape)
print(len(y))

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


def get_spectral_graph_metrics(eigenvalues_big,graph=None,T= None,**kwargs):
    if T is None:
        W = graph.get_T()
    else:
        W = T
    graph_laplacian = -W
    graph_laplacian.setdiag(graph_laplacian.diagonal() + np.ones(graph_laplacian.shape[0]))
    eigenvalues_small = metrics.compute_eigenvalues(graph_laplacian,n_eigenvals=1000)
    d_spectral = metrics.spectral_graph_distance(eigenvalues_big,eigenvalues_small)[0]
    rel_eigval_e = metrics.relative_eigenvalue_error(eigenvalues_big,eigenvalues_small)[0]
    return d_spectral,rel_eigval_e


def get_graph_metrics(closeness_knn,betweenness_knn,graph=None,T=None,**kwargs):
    if T is None:
        W = graph.get_T()
    else:
        W=T
    graph_close_centrality = metrics.harmonic_centrality(W,k=10000)
    graph_between_centrality = metrics.betweenness_centrality(W,directed=True,k=10000)
    close = metrics.KL_Div(closeness_knn,graph_close_centrality)
    between = metrics.KL_Div(betweenness_knn,graph_between_centrality)
    return close,between


def get_embedding_metrics(X,y,graph=None,emb=None,landmarks=None,**kwargs):
    if emb is None or landmarks is None:
        TSNE_emb = graph.TSNE
        current_landmarks = graph.find_corresponding_landmarks_at_level()
    else:
        TSNE_emb = emb
        current_landmarks = landmarks 

    f_cluster = y[current_landmarks]
    knn_acc=metrics.knn_acc(X[current_landmarks],TSNE_emb,k=10)
    trust=metrics.trustworthiness(X[current_landmarks],TSNE_emb,k=10)
    silh=metrics.silhouette_score(TSNE_emb,f_cluster)[0]
    dbi=metrics.Davies_bouldin_index(TSNE_emb,f_cluster)
    return knn_acc,trust,silh,dbi



# %%
(X,y,class_table), dir, (W_udir,L_udir),(W_dir,L_dir) = get_data(data_name,give_directed=True)

# %%

def get_metric_if_exist(fname,function,dir=dir,**kwargs):
    if os.path.isfile(join(dir,fname)):
        print("found_file")
        return load_from_pickle(join(dir,fname))
    
    print(f"File not found: {fname}, create metric")
    output = function(**kwargs)
    save_pickle(output,join(dir,fname))
    return output


def get_metrics_if_exist(fname,functions,dir=dir,**kwargs):
    if os.path.isfile(join(dir,fname)):
        print("found_file")
        return load_from_pickle(join(dir,fname))
    
    print(f"File not found: {fname}, create metric")
    out = []
    for function in functions:
        output = function(**kwargs)
        out.append(output)

    save_pickle(out,join(dir,fname))
    return out

def load_file_if_exists(fname,dir=dir):
    if os.path.isfile(join(dir,fname)):
        print("found_file")
        return load_from_pickle(join(dir,fname))
    else:
        print(f"File not found: {fname}")

def create_or_load(fname,function,dir=dir,**kwargs):
    if os.path.isfile(join(dir,fname)):
        print("found_file")
        return load_from_pickle(join(dir,fname))
    else:
        print(f"File not found: {fname}")
        out = function(**kwargs)
        save_pickle(out,join(dir,fname))
        return out


def create_graph(sampling,connection,X,y,filename,dir=dir,level=2,seed=seed,directed=False,discrete=True,k=10):
    if os.path.isfile(join(dir,filename)):
        print("found_file")
        graph = load_from_pickle(join(dir,filename))
    else:
        print(f"no file found {join(dir,filename)}")
        graph_1 = OOP_Multilevel_tsne.KNNGraph(data=X,labels=y,n=k,data_name=data_name,directed=directed,weighted=True,landmark_sampling=copy.deepcopy(sampling),connection=copy.deepcopy(connection),discrete_labels=discrete,seed=seed)
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

def save_pickle_dir(fname,X,dir=dir):
        save_pickle(X,join(dir,fname))


# %%
closeness_knn_undirected = load_file_if_exists(f"{data_name}_closeness.pkl",dir)
betweenness_knn_undirected = load_file_if_exists(f"{data_name}_betweenness.pkl",dir)
eigenvalues_big_undirected = load_file_if_exists(f"{data_name}_eigenvalues_big_undirected.pkl",dir)


# %%
closeness_knn_directed = load_file_if_exists(f"{data_name}_closeness_directed.pkl",dir)
betweenness_knn_directed = load_file_if_exists(f"{data_name}_betweenness_directed.pkl",dir)

# %% [markdown]
# comparison directed vs undirected kNN graph

# %%
sampling_rand = OOP_Sampling.RandomSampling(seed=seed,shrinkage=reduction)
sampling_rw = OOP_Sampling.RandomWalksSampling(n_walks=100,shrinkage=reduction,seed=seed)
sampling_hubs = OOP_Sampling.HighestDegreeSampling(shrinkage=reduction)
sampling_hbn = OOP_Sampling.HighDegreeExclusionNN(shrinkage=reduction,k=10)
connection_slrw = OOP_Connecting.StateToLandmarksRandomWalks(n_walks=100,W=np.ones(X.shape[0]),seed=seed)
connection_exact = OOP_Connecting.StateToLandmarksExact(W=np.ones(X.shape[0]) ,use_gambler=True,threshold_I=1e-4,threshold_T=1e-4,seed=seed)

# %%
#TODO: run
directed_slrw_rw = create_graph(X=X,y=y,sampling=sampling_rw,connection=connection_slrw,filename="directed-slrw-rw.pkl",dir=dir,directed=True)
directed_slrw_hubs = create_graph(X=X,y=y,sampling=sampling_hubs,connection=connection_slrw,filename="directed-slrw-hubs.pkl",dir=dir,directed=True)
directed_slrw_random = create_graph(X=X,y=y,sampling=sampling_rand,connection=connection_slrw,filename="directed-slrw-random.pkl",dir=dir,directed=True)
directed_exact_hubs = create_graph(X=X,y=y,sampling=sampling_hubs,connection=connection_exact,filename="directed-exact-hubs.pkl",dir=dir,directed=True)

# %%
directed_slrw_hubs = create_graph(X=X,y=y,sampling=sampling_hubs,connection=connection_slrw,filename="directed-slrw-hubs.pkl",dir=dir,directed=True)
directed_slrw_hubs.bigger_graph.get_landmarks

# %%

slrw_random = create_graph(X=X,y=y,sampling=sampling_rand,connection=connection_slrw,filename="slrw-random.pkl",dir=dir)
slrw_rw   = create_graph(X=X,y=y,sampling=sampling_rw,connection=connection_slrw,filename="slrw-rw.pkl",dir=dir)
slrw_hubs = create_graph(X=X,y=y,sampling=sampling_rw,connection=connection_slrw,filename="slrw-hubs.pkl",dir=dir)


# %%
#undirected
metrics_slrw_random_1 =   get_metrics_if_exist("slrw_random_metrics_1",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=slrw_random.bigger_graph,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
metrics_slrw_random_2 =   get_metrics_if_exist("slrw_random_metrics_2",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=slrw_random,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)

metrics_slrw_rw_1 =   get_metrics_if_exist("slrw_rw_metrics_1",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=slrw_rw.bigger_graph,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
metrics_slrw_rw_2 =   get_metrics_if_exist("slrw_rw_metrics_2",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=slrw_rw,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)

metrics_slrw_hubs_1 =   get_metrics_if_exist("slrw_hubs_metrics_1",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=slrw_hubs.bigger_graph,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
metrics_slrw_hubs_2 =   get_metrics_if_exist("slrw_hubs_metrics_2",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=slrw_hubs,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)


# %%
metrics_directed_slrw_random_1 =   get_metrics_if_exist("directed_slrw_random_metrics_1",   [get_graph_metrics,get_embedding_metrics],dir=dir,graph=directed_slrw_random.bigger_graph,closeness_knn=closeness_knn_directed,betweenness_knn=betweenness_knn_directed,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y,directed=True)
metrics_directed_slrw_random_2 =   get_metrics_if_exist("directed_slrw_random_metrics_2",   [get_graph_metrics,get_embedding_metrics],dir=dir,graph=directed_slrw_random,closeness_knn=closeness_knn_directed,betweenness_knn=betweenness_knn_directed,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y,directed=True)

metrics_directed_slrw_rw_1 =   get_metrics_if_exist("directed_slrw_rw_metrics_1",   [get_graph_metrics,get_embedding_metrics],dir=dir,graph=directed_slrw_rw.bigger_graph,closeness_knn=closeness_knn_directed,betweenness_knn=betweenness_knn_directed,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y,directed=True)
metrics_directed_slrw_rw_2 =   get_metrics_if_exist("directed_slrw_rw_metrics_2",   [get_graph_metrics,get_embedding_metrics],dir=dir,graph=directed_slrw_rw,closeness_knn=closeness_knn_directed,betweenness_knn=betweenness_knn_directed,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y,directed=True)

metrics_directed_slrw_hubs_1 =   get_metrics_if_exist("directed_slrw_hubs_metrics_1",   [get_graph_metrics,get_embedding_metrics],dir=dir,graph=directed_slrw_hubs.bigger_graph,closeness_knn=closeness_knn_directed,betweenness_knn=betweenness_knn_directed,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y,directed=True)
metrics_directed_slrw_hubs_2 =   get_metrics_if_exist("directed_slrw_hubs_metrics_2",   [get_graph_metrics,get_embedding_metrics],dir=dir,graph=directed_slrw_hubs,closeness_knn=closeness_knn_directed,betweenness_knn=betweenness_knn_directed,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y,directed=True)

# %%
#lower leve
landmarks = [slrw_rw.bigger_graph.find_corresponding_landmarks_at_level(),directed_slrw_rw.bigger_graph.find_corresponding_landmarks_at_level(),slrw_hubs.bigger_graph.find_corresponding_landmarks_at_level(),directed_slrw_hubs.bigger_graph.find_corresponding_landmarks_at_level()]
kind_labels = ["RW","Hubs"]
plg.plot_directed_undirected(y,class_table,landmarks,kind_labels,normalize=True, data_name=data_name,figsize=(10,5),fontsize=10,use_legend=False,colors=colors,fsave=f"{data_name}_directed_undirected_landmarks_comp_2.png")


# %%

landmarks = [slrw_rw.bigger_graph.find_corresponding_landmarks_at_level(),directed_slrw_rw.bigger_graph.find_corresponding_landmarks_at_level(),slrw_hubs.bigger_graph.find_corresponding_landmarks_at_level(),directed_slrw_hubs.bigger_graph.find_corresponding_landmarks_at_level(),slrw_random.bigger_graph.find_corresponding_landmarks_at_level(),directed_slrw_random.bigger_graph.find_corresponding_landmarks_at_level()]
kind_labels = ["RW","Hubs","Random"]
plg.plot_directed_undirected(y,class_table,landmarks,kind_labels,normalize=True, data_name=f"{data_name}",figsize=(14,4),fontsize=8,colors=colors,fsave=f"{data_name}_directed_undirected_landmarks_comp.png")


# %%
print(np.std(np.histogram(y[directed_slrw_hubs.bigger_graph.find_corresponding_landmarks_at_level()])[0]))

# %%
print(np.std(np.histogram(y[slrw_rw.bigger_graph.find_corresponding_landmarks_at_level()])[0]))

# %%
#higher level

landmarks = [directed_slrw_rw.find_corresponding_landmarks_at_level(),slrw_rw.find_corresponding_landmarks_at_level(),directed_slrw_hubs.find_corresponding_landmarks_at_level(),slrw_hubs.find_corresponding_landmarks_at_level()]
kind_labels = ["RW","Hubs"]
plg.plot_directed_undirected(y,class_table,landmarks,kind_labels,normalize=True, data_name=f"{data_name} level 2",figsize=(10,5),fontsize=10,colors=colors,fsave=f"{data_name}_directed_undirected_landmarks_comp_2.png")


# %%

#embeddings = [
#    [((slrw_random.bigger_graph.TSNE,slrw_random.bigger_graph.find_corresponding_landmarks_at_level())),(slrw_random.TSNE,slrw_random.find_corresponding_landmarks_at_level())],
#    [((slrw_rw.bigger_graph.TSNE,slrw_rw.bigger_graph.find_corresponding_landmarks_at_level())),(slrw_rw.TSNE,slrw_rw.find_corresponding_landmarks_at_level())]
#]


embeddings = [
    [((directed_slrw_rw.bigger_graph.TSNE,directed_slrw_rw.bigger_graph.find_corresponding_landmarks_at_level())),(directed_slrw_random.TSNE,directed_slrw_random.find_corresponding_landmarks_at_level())],
    [((slrw_rw.bigger_graph.TSNE,slrw_rw.bigger_graph.find_corresponding_landmarks_at_level())),(slrw_rw.TSNE,slrw_rw.find_corresponding_landmarks_at_level())]
]
names=["Level 1","Level 2"]
titles=["directed","undirected"]
plg.plot_embeddings_mosaic_discrete(embeddings,titles=titles,names=names,y=y,colors=colors,fsave=f"{data_name}_directed_undirected_over_levels")


# %%
embeddings = [
    [((directed_slrw_hubs.bigger_graph.TSNE,directed_slrw_hubs.bigger_graph.find_corresponding_landmarks_at_level())),((directed_slrw_rw.bigger_graph.TSNE,directed_slrw_rw.bigger_graph.find_corresponding_landmarks_at_level())),((directed_slrw_random.bigger_graph.TSNE,directed_slrw_random.bigger_graph.find_corresponding_landmarks_at_level()))],
    [((slrw_hubs.bigger_graph.TSNE,slrw_hubs.bigger_graph.find_corresponding_landmarks_at_level())),((slrw_rw.bigger_graph.TSNE,slrw_rw.bigger_graph.find_corresponding_landmarks_at_level())),((slrw_random.bigger_graph.TSNE,slrw_random.bigger_graph.find_corresponding_landmarks_at_level()))]
]
names=["Hubs","RW","Random"]
titles=["directed","undirected"]
plg.plot_embeddings_mosaic_discrete(embeddings,titles=titles,names=names,y=y,colors=colors,fsave=f"{data_name}_directed_undirected_over_levels")


# %%
sampling_hubs = OOP_Sampling.HighestDegreeSampling(shrinkage=reduction)
connection_llrw = OOP_Connecting.LandmarksToLandmarksRandomWalks(n_walks=100)
threshold = 1e-4
connection_exact = OOP_Connecting.StateToLandmarksExact(W=np.ones(X.shape[0]),threshold_I=threshold,threshold_T=threshold)


# %% [markdown]
# Best ways to connect landmarks

# %%

noises = [0,0.1,0.2,0.3]
llrw_hubs_n0 = create_graph(X=X,y=y,sampling=sampling_hubs,connection=connection_llrw,filename="llrw-hubs.pkl",dir=dir)
slrw_hubs_n0 = create_graph(X=X,y=y,sampling=sampling_hubs,connection=connection_slrw,filename="slrw-hubs.pkl",dir=dir)
exact_hubs_n0 = create_graph(X=X,y=y,sampling=sampling_hubs,connection=connection_exact,filename="exact-hubs.pkl",dir=dir)


noise = noises[1]
noisy_X = helper.generate_noise(X,noise=noise,seed=seed,data_name=data_name)

llrw_hubs_n1 = create_graph(X=noisy_X,y=y,sampling=sampling_hubs,connection=connection_llrw,filename="llrw-hubs-n01.pkl",dir=dir)
slrw_hubs_n1 = create_graph(X=noisy_X,y=y,sampling=sampling_hubs,connection=connection_slrw,filename="slrw-hubs-n01.pkl",dir=dir)
exact_hubs_n1 = create_graph(X=noisy_X,y=y,sampling=sampling_hubs,connection=connection_exact,filename="exact-hubs-n01.pkl",dir=dir)


noise = noises[2]
noisy_X = helper.generate_noise(X,noise=noise,seed=seed,data_name=data_name)

llrw_hubs_n2 = create_graph(X=noisy_X,y=y,sampling=sampling_hubs,connection=connection_llrw,filename="llrw-hubs-n02.pkl",dir=dir)
slrw_hubs_n2 = create_graph(X=noisy_X,y=y,sampling=sampling_hubs,connection=connection_slrw,filename="slrw-hubs-n02.pkl",dir=dir)
exact_hubs_n2 = create_graph(X=noisy_X,y=y,sampling=sampling_hubs,connection=connection_exact,filename="exact-hubs-n02.pkl",dir=dir)


noise = noises[3]
noisy_X = helper.generate_noise(X,noise=noise,seed=seed,data_name=data_name)

llrw_hubs_n3 = create_graph(X=noisy_X,y=y,sampling=sampling_hubs,connection=connection_llrw,filename="llrw-hubs-n03.pkl",dir=dir)
slrw_hubs_n3 = create_graph(X=noisy_X,y=y,sampling=sampling_hubs,connection=connection_slrw,filename="slrw-hubs-n03.pkl",dir=dir)
exact_hubs_n3 = create_graph(X=noisy_X,y=y,sampling=sampling_hubs,connection=connection_exact,filename="exact-hubs-n03.pkl",dir=dir)

graphs = [
    [llrw_hubs_n0,llrw_hubs_n1,llrw_hubs_n2,llrw_hubs_n3],
    [slrw_hubs_n0,slrw_hubs_n1,slrw_hubs_n2,slrw_hubs_n3],
    [exact_hubs_n0,exact_hubs_n1,exact_hubs_n2,exact_hubs_n3],
]

# %%

#metrics
ll_metrics_n0 = get_metrics_if_exist(fname="ll_metrics_hubs",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=llrw_hubs_n0.bigger_graph ,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)
sl_metrics_n0 = get_metrics_if_exist(fname="slrw_hubs_metrics_1",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=slrw_hubs_n0.bigger_graph ,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)
ex_metrics_n0 = get_metrics_if_exist(fname="ex_metrics_hubs",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=exact_hubs_n0.bigger_graph,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)

ll_metrics_n1 = get_metrics_if_exist(fname="ll_metrics_hubs_n1",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=llrw_hubs_n1.bigger_graph ,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)
sl_metrics_n1 = get_metrics_if_exist(fname="sl_metrics_hubs_n1",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=slrw_hubs_n1.bigger_graph ,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)
ex_metrics_n1 = get_metrics_if_exist(fname="ex_metrics_hubs_n1",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=exact_hubs_n1.bigger_graph,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)

ll_metrics_n2 = get_metrics_if_exist(fname="ll_metrics_hubs_n2",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=llrw_hubs_n2.bigger_graph ,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)
sl_metrics_n2 = get_metrics_if_exist(fname="sl_metrics_hubs_n2",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=slrw_hubs_n2.bigger_graph ,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)
ex_metrics_n2 = get_metrics_if_exist(fname="ex_metrics_hubs_n2",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=exact_hubs_n2.bigger_graph,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)

ll_metrics_n3 = get_metrics_if_exist(fname="ll_metrics_hubs_n3",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=llrw_hubs_n3.bigger_graph ,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)
sl_metrics_n3 = get_metrics_if_exist(fname="sl_metrics_hubs_n3",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=slrw_hubs_n3.bigger_graph ,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)
ex_metrics_n3 = get_metrics_if_exist(fname="ex_metrics_hubs_n3",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=exact_hubs_n3.bigger_graph,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)


# %%

ll_metrics_n0_2 = get_metrics_if_exist(fname="ll_metrics_hubs_2",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=llrw_hubs_n0 ,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)
sl_metrics_n0_2 = get_metrics_if_exist(fname="slrw_hubs_metrics_2",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=slrw_hubs_n0 ,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)
ex_metrics_n0_2 = get_metrics_if_exist(fname="ex_metrics_hubs_2",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=exact_hubs_n0,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)

ll_metrics_n1_2 = get_metrics_if_exist(fname="ll_metrics_hubs_n1_2",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=llrw_hubs_n1 ,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)
sl_metrics_n1_2 = get_metrics_if_exist(fname="sl_metrics_hubs_n1_2",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=slrw_hubs_n1 ,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)
ex_metrics_n1_2 = get_metrics_if_exist(fname="ex_metrics_hubs_n1_2",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=exact_hubs_n1,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)

ll_metrics_n2_2 = get_metrics_if_exist(fname="ll_metrics_hubs_n2_2",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=llrw_hubs_n2 ,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)
sl_metrics_n2_2 = get_metrics_if_exist(fname="sl_metrics_hubs_n2_2",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=slrw_hubs_n2 ,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)
ex_metrics_n2_2 = get_metrics_if_exist(fname="ex_metrics_hubs_n2_2",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=exact_hubs_n2,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)

ll_metrics_n3_2 = get_metrics_if_exist(fname="ll_metrics_hubs_n3_2",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=llrw_hubs_n3 ,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)
sl_metrics_n3_2 = get_metrics_if_exist(fname="sl_metrics_hubs_n3_2",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=slrw_hubs_n3 ,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)
ex_metrics_n3_2 = get_metrics_if_exist(fname="ex_metrics_hubs_n3_2",functions= [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=exact_hubs_n3,eigenvalues_big=eigenvalues_big_undirected,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)


# %%
embeddings_lv2 = [[(graph.TSNE,graph.find_corresponding_landmarks_at_level()) for graph in noise_graphs] for noise_graphs in graphs]
#shouldnt this be y not landmarks idx?
titles=["LL-Hubs","SL-Hubs","Kron-Hubs"]
names=[f"Noise p:{noises[i]}" for i in range(4)]
plg.plot_embeddings_mosaic_discrete(embeddings_lv2,titles,names,y,colors=colors,fsave=f"{data_name}_embeddings_increasing_noise_plot_lv2.png")


# %%
embeddings_lv1 = [[(graph.bigger_graph.TSNE,graph.bigger_graph.find_corresponding_landmarks_at_level()) for graph in noise_graphs] for noise_graphs in graphs]
plg.plot_embeddings_mosaic_discrete(embeddings_lv1,titles,names,y,colors=colors,fsave=f"{data_name}_embeddings_increasing_noise_plot_lv1.png")

# %% [markdown]
# Node aggregation methods

# %%
import leidenalg as la
from igraph import Graph
import openTSNE
import helper

def connectivity(f_clustering,graph):
        n_clusters = len(np.unique(f_clustering))
        adj = sp.dok_array((n_clusters,n_clusters))
        for point in range(graph.vcount()):
            clusterA = int(f_clustering[point])
            neighbors = graph.neighbors(point,mode="out")
            for neighbor in neighbors:
                clusterB = int(f_clustering[neighbor])
                if clusterA != clusterB:
                    adj[clusterA,clusterB] +=1

        adj = adj.tocsr()
        return adj

def get_igraph_from_W(W:sp.sparray,directed:bool,weighted:bool):
    W.setdiag(0)
    W.eliminate_zeros()
    N = W.shape[0]
    if not directed:
        W = W.maximum(W.T)
    if weighted:
        W = normalize(W)
    W = W.tocoo()

    edges = [(s, t) for s, t in zip(*W.tocoo().coords)]
    graph = Graph(N,edges=edges, directed=True)
    if weighted:
        graph.es['weight'] = W.data

    return graph


def create_TSNE_embedding_from_Affinity(affinity,X):
    P = affinity
    P.setdiag(0)
    P = normalize(P,"l1")
    P = P + P.T
    P /= 2*P.shape[0]
    custom_affinity = openTSNE.affinity.PrecomputedAffinities(P)
    init = openTSNE.initialization.pca(X)
    embedding = openTSNE.TSNEEmbedding(embedding=init,affinities=custom_affinity)
    
    embedding.optimize(n_iter=250,exaggeration=12)
    embedded_points = embedding.optimize(n_iter=500,exaggeration=1)
    return embedded_points



def select_point_from_cluster(fcluster,seed=None):
    if seed is not None:
        np.random.seed(seed)
    unique_clusters = np.unique(fcluster)
    selected_indices = []

    for cluster in unique_clusters:
        cluster_indices = np.where(fcluster == cluster)[0]
        selected_indices.append(np.random.choice(cluster_indices))

    return np.array(selected_indices)

# %%
u_knn_igraph = get_igraph_from_W(W_udir,directed=False,weighted=True)
dir_knn_igraph = get_igraph_from_W(W_dir,directed=True,weighted=True)

# %%
def connectivity(f_clustering,graph):
        n_clusters = len(np.unique(f_clustering))
        adj = sp.dok_array((n_clusters,n_clusters))
        for point in range(graph.vcount()):
            clusterA = int(f_clustering[point])
            neighbors = graph.neighbors(point,mode="out")
            for neighbor in neighbors:
                clusterB = int(f_clustering[neighbor])
                if clusterA != clusterB:
                    adj[clusterA,clusterB] +=1

        adj = adj.tocsr()
        return adj

# %%
#walktrap
def create_walktrap_connectivity(graph,red=reduction):
    walktrap = graph.community_walktrap(steps=3)
    walktrap_clusters = walktrap.as_clustering(n=int(red*graph.vcount()))
    fcluster_walktrap = helper.cluster_to_fcluster(walktrap_clusters)
    print(f"Found {len(np.unique(fcluster_walktrap))} cluster for level 1")
    W_walktrap_1 = connectivity(fcluster_walktrap,graph)
    W_walktrap_1 = normalize(W_walktrap_1,norm="l1",axis=1)
    u_knn_igraph_2 = get_igraph_from_W(W_walktrap_1,directed=True,weighted=True)
    walktrap_2 = u_knn_igraph_2.community_walktrap(steps=3)
    walktrap_clusters_2 = walktrap_2.as_clustering(n=int(reduction*u_knn_igraph_2.vcount()))
    fcluster_walktrap_2 = helper.cluster_to_fcluster(walktrap_clusters_2)
    W_walktrap_2 = connectivity(fcluster_walktrap_2,u_knn_igraph_2)
    W_walktrap_2 = normalize(W_walktrap_2,norm="l1",axis=1)
    print(f"Found {len(np.unique(fcluster_walktrap_2))} cluster for level 2")

    return fcluster_walktrap,fcluster_walktrap_2, W_walktrap_1,W_walktrap_2

# %%
fcluster_walktrap_1,fcluster_walktrap_2, W_walktrap_1,W_walktrap_2 = create_or_load(fname="walktrap-connectivity",function=create_walktrap_connectivity,graph=u_knn_igraph)

# %%
dir_fcluster_walktrap_1,dir_fcluster_walktrap_2, dir_W_walktrap_1,dir_W_walktrap_2 = create_or_load(fname="walktrap-connectivity-dir",function=create_walktrap_connectivity,graph=dir_knn_igraph)

# %%
#leiden
def create_leiden_connectivity(graph,res1=752,res2=50,seed=seed):
    leiden_clusters = la.find_partition(u_knn_igraph, la.RBConfigurationVertexPartition, resolution_parameter=res1,seed=seed)
    fcluster_leiden = helper.cluster_to_fcluster(leiden_clusters)
    print(f"Found {len(np.unique(fcluster_leiden))} cluster for level 1")
    W_leiden_1 = connectivity(fcluster_leiden,graph)
    W_leiden_1 = normalize(W_leiden_1,norm="l1",axis=1)
    
    u_knn_igraph_2 = get_igraph_from_W(W_leiden_1,directed=True,weighted=True)
    leiden_clusters_2 = la.find_partition(u_knn_igraph_2, la.RBConfigurationVertexPartition, resolution_parameter=res2,seed=seed)
    fcluster_leiden_2 = helper.cluster_to_fcluster(leiden_clusters_2)
    W_leiden_2 = connectivity(fcluster_leiden_2,u_knn_igraph_2)
    W_leiden_2 = normalize(W_leiden_2,norm="l1",axis=1)
    print(f"Found {len(np.unique(fcluster_leiden_2))} cluster for level 2")

    return fcluster_leiden,fcluster_leiden_2, W_leiden_1,W_leiden_2

# %%
fcluster_leiden_1,fcluster_leiden_2, W_leiden_1,W_leiden_2 = create_or_load(fname="leiden-connectivity",function=create_leiden_connectivity,graph=u_knn_igraph,res1=752,res2=67)

# %%
dir_fcluster_leiden_1,dir_fcluster_leiden_2, dir_W_leiden_1,dir_W_leiden_2 = create_or_load(fname="leiden-connectivity-dir",function=create_leiden_connectivity,graph=dir_knn_igraph,res1=752,res2=67)

# %%
random_sampling = OOP_Sampling.RandomSampling(seed=seed)
connectivi = OOP_Connecting.Connectivity(seed=seed)

# %%

label_propagation_udir = create_graph(sampling=random_sampling,connection=connectivi,X=X,y=y,filename="lp-random",dir=dir,level=2,seed=seed,directed=False,discrete=True)
label_propagation_dir = create_graph(sampling=random_sampling,connection=connectivi,X=X,y=y,filename="lp-random_dir",dir=dir,level=2,seed=seed,directed=True,discrete=True)

# %%
label_propagation_hubs = create_graph(sampling=sampling_hubs,connection=connectivi,X=X,y=y,filename="lp-hubs",dir=dir,level=2,seed=seed,directed=False,discrete=True)
embeddings = [(label_propagation_hubs.bigger_graph.TSNE,label_propagation_hubs.bigger_graph.find_corresponding_landmarks_at_level()),(label_propagation_hubs.TSNE,label_propagation_hubs.find_corresponding_landmarks_at_level())]
plg.plot_embeddings_row_discrete(embeddings,["level 1", "level 2"],y,colors=colors,fsave=f"{data_name}_cluster_lp_hubs.png")

# %%
directed_label_propagation_hubs = create_graph(sampling=sampling_hubs,connection=connectivi,X=X,y=y,filename="lp-hubs-directed",dir=dir,level=2,seed=seed,directed=True,discrete=True)
embeddings = [(directed_label_propagation_hubs.bigger_graph.TSNE,directed_label_propagation_hubs.bigger_graph.find_corresponding_landmarks_at_level()),(directed_label_propagation_hubs.TSNE,directed_label_propagation_hubs.find_corresponding_landmarks_at_level())]
plg.plot_embeddings_row_discrete(embeddings,["level 1", "level 2"],y,colors=colors,fsave=f"{data_name}_cluster_lp_hubs.png")

# %%
def create_embedding(fcluster,W,X,first_level=None):
    landmarks = select_point_from_cluster(fcluster)
    if first_level is not None:
        for level in first_level[::-1]:
            landmarks=level[landmarks]
    embedding = create_TSNE_embedding_from_Affinity(affinity=W,X=X[landmarks])
    return embedding,landmarks

# %%


embedding_leiden_dir_1,landmarks_leiden_dir_1 = create_or_load(fname="embedding_dir_leiden_1",function=create_embedding,fcluster=dir_fcluster_leiden_1,W=dir_W_leiden_1,X=X)
landmarks_labels_leiden_dir_1 = y[landmarks_leiden_dir_1]
embedding_leiden_1, landmarks_leiden_1 = create_or_load(fname="embedding_leiden_1",function=create_embedding,fcluster=fcluster_leiden_1,W=W_leiden_1,X=X)
landmarks_labels_leiden_1 = y[landmarks_leiden_1]

embedding_leiden_dir_2 ,landmarks_leiden_dir_2= create_or_load(fname="embedding_dir_leiden_2",function=create_embedding,fcluster=dir_fcluster_leiden_2,W=dir_W_leiden_2,X=X,first_level=[landmarks_leiden_dir_1])
landmark_labels_leiden_dir_2 = y[landmarks_leiden_dir_2]
embedding_leiden_2, landmarks_leiden_2 = create_or_load(fname="embedding_leiden_2",function=create_embedding,fcluster=fcluster_leiden_2,W=W_leiden_2,X=X,first_level=[landmarks_leiden_1])
landmark_labels_leiden_2 = y[landmarks_leiden_2]


embedding_walktrap_dir_1,landmarks_walktrap_dir_1 = create_or_load(fname="embedding_dir_walktrap_1",function=create_embedding,fcluster=dir_fcluster_walktrap_1,W=dir_W_walktrap_1,X=X)
landmarks_labels_walktrap_dir_1 = y[landmarks_walktrap_dir_1]
embedding_walktrap_1,landmarks_walktrap_1 = create_or_load(fname="embedding_walktrap_1",function=create_embedding,fcluster=fcluster_walktrap_1,W=W_walktrap_1,X=X)
landmarks_labels_walktrap_1 = y[landmarks_walktrap_1]


embedding_walktrap_dir_2,landmarks_walktrap_dir_2 = create_or_load(fname="embedding_dir_walktrap_2",function=create_embedding,fcluster=dir_fcluster_walktrap_2,W=dir_W_walktrap_2,X=X,first_level=[landmarks_walktrap_dir_1])
landmark_labels_walktrap_dir_2 = y[landmarks_walktrap_dir_2]
embedding_walktrap_2,landmarks_walktrap_2= create_or_load(fname="embedding_walktrap_2",function=create_embedding,fcluster=fcluster_walktrap_2,W=W_walktrap_2,X=X,first_level=[landmarks_walktrap_1])
landmark_labels_walktrap_2 = y[landmarks_walktrap_2]

# %%
#undirected per level
embeddings = [
    [(embedding_leiden_1,landmarks_leiden_1),(embedding_walktrap_1,landmarks_walktrap_1),(label_propagation_udir.bigger_graph.TSNE,label_propagation_udir.bigger_graph.get_landmarks())],
    [(embedding_leiden_2,landmarks_leiden_2),(embedding_walktrap_2,landmarks_walktrap_2),(label_propagation_udir.TSNE,label_propagation_udir.find_corresponding_landmarks_at_level())]
              ]
titles = ["Level:1","Level:2"]
names = ["Leiden","Walktrap","Label Propagation"]
plg.plot_embeddings_mosaic_discrete(embeddings,titles,names,y,colors=colors,fsave=f"{data_name}_cluster_methods_undirected_per_level.png")



# %%
#directed per level
embeddings = [
    [(embedding_leiden_dir_1,landmarks_leiden_dir_1),(embedding_walktrap_dir_1,landmarks_walktrap_dir_1),(label_propagation_dir.bigger_graph.TSNE,label_propagation_dir.bigger_graph.find_corresponding_landmarks_at_level())],
    [(embedding_leiden_dir_2,landmarks_leiden_dir_2),(embedding_walktrap_dir_2,landmarks_walktrap_dir_2),(label_propagation_dir.TSNE,label_propagation_dir.find_corresponding_landmarks_at_level())]
              ]
titles = ["Level:1","Level:2"]
names = ["Leiden","Walktrap","Label Propagation"]
plg.plot_embeddings_mosaic_discrete(embeddings,titles,names,y,colors=colors,fsave=f"{data_name}_cluster_methods_directed_per_level.png")



# %%
#directed vs directed lv1
embeddings = [
    [(embedding_leiden_dir_1,landmarks_leiden_dir_1),(embedding_walktrap_dir_1,landmarks_walktrap_dir_1),(label_propagation_dir.bigger_graph.TSNE,label_propagation_dir.bigger_graph.find_corresponding_landmarks_at_level())],
    [(embedding_leiden_1,landmarks_leiden_1),(embedding_walktrap_1,landmarks_walktrap_1),(label_propagation_udir.bigger_graph.TSNE,label_propagation_udir.bigger_graph.find_corresponding_landmarks_at_level())]
              ]
titles = ["directed","undirected"]
names = ["Leiden","Walktrap","Label Propagation"]
plg.plot_embeddings_mosaic_discrete(embeddings,titles,names,y,colors=colors,fsave=f"{data_name}_cluster_methods_comparison_directed_undirected_lv1.png")



# %%
#directed vs directed lv2
embeddings = [
    [(embedding_leiden_dir_2,landmarks_leiden_dir_2),(embedding_walktrap_dir_2,landmarks_walktrap_dir_2),(label_propagation_dir.TSNE,label_propagation_dir.find_corresponding_landmarks_at_level())],
    [(embedding_leiden_2,landmarks_leiden_2),(embedding_walktrap_2,landmarks_walktrap_2),(label_propagation_udir.TSNE,label_propagation_udir.find_corresponding_landmarks_at_level())]
              ]
titles = ["directed","undirected"]
names = ["Leiden","Walktrap","Label Propagation"]
plg.plot_embeddings_mosaic_discrete(embeddings,titles,names,y,colors=colors,fsave=f"{data_name}_cluster_methods_comparison_directed_undirected_lv2.png")

# %%
#Evaluation of the Clustering

print("undirected")
print("lv1 (walktrap/leiden/lp):")
print(metrics.silhouette_score(X,fcluster_walktrap_1)[0])
print(metrics.silhouette_score(X,fcluster_leiden_1)[0])
print(metrics.silhouette_score(X,label_propagation_udir.bigger_graph.connection.f_cluster)[0])
print("lv2 (walktrap/leiden/lp):")
print(metrics.silhouette_score(X[landmarks_walktrap_1],fcluster_walktrap_2)[0])
print(metrics.silhouette_score(X[landmarks_leiden_1],fcluster_leiden_2)[0])
print(metrics.silhouette_score(X[label_propagation_udir.bigger_graph.find_corresponding_landmarks_at_level()],label_propagation_udir.connection.f_cluster)[0])

print("undirected")
print("lv1 (walktrap/leiden/lp):")
print(metrics.silhouette_score(X,dir_fcluster_walktrap_1)[0])
print(metrics.silhouette_score(X,dir_fcluster_leiden_1)[0])
print(metrics.silhouette_score(X,label_propagation_udir.bigger_graph.connection.f_cluster)[0])
print("lv2 (walktrap/leiden/lp):")
print(metrics.silhouette_score(X[landmarks_walktrap_1],dir_fcluster_walktrap_2)[0])
print(metrics.silhouette_score(X[landmarks_leiden_1],dir_fcluster_leiden_2)[0])
print(metrics.silhouette_score(X[label_propagation_udir.bigger_graph.find_corresponding_landmarks_at_level()],label_propagation_udir.connection.f_cluster)[0])

# %%
metrics_leiden_1 =   get_metrics_if_exist("leiden_metric_1",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_leiden_1,landmarks=landmarks_leiden_1,T=W_leiden_1,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
metrics_leiden_2 =   get_metrics_if_exist("leiden_metric_2",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_leiden_2,landmarks=landmarks_leiden_2,T=W_leiden_2,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)

metrics_walktrap_1 =   get_metrics_if_exist("walktrap_metric_1",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_walktrap_1,landmarks=landmarks_walktrap_1,T=W_walktrap_1,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
metrics_walktrap_2 =   get_metrics_if_exist("walktrap_metric_2",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_walktrap_2,landmarks=landmarks_walktrap_2,T=W_walktrap_2,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)

metrics_labelpropagation_1 =   get_metrics_if_exist("lp_metric_1",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=label_propagation_udir.bigger_graph,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
metrics_labelpropagation_2 =   get_metrics_if_exist("lp_metric_2",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=label_propagation_udir,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)


# %%
metrics_leiden_dir_1 =   get_metrics_if_exist("leiden_metric_dir_1",   [get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_leiden_dir_1,landmarks=landmarks_leiden_dir_1,T=dir_W_leiden_1,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)
metrics_leiden_dir_2 =   get_metrics_if_exist("leiden_metric_dir_2",   [get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_leiden_dir_2,landmarks=landmarks_leiden_dir_2,T=dir_W_leiden_2,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)

metrics_walktrap_dir_1 =   get_metrics_if_exist("walktrap_metric_dir_1",   [get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_walktrap_dir_1,landmarks=landmarks_walktrap_dir_1,T=dir_W_walktrap_1,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)
metrics_walktrap_dir_2 =   get_metrics_if_exist("walktrap_metric_dir_2",   [get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_walktrap_dir_2,landmarks=landmarks_walktrap_dir_2,T=dir_W_walktrap_2,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)

metrics_labelpropagation_dir_1 =   get_metrics_if_exist("lp_metric_dir_1",   [get_graph_metrics,get_embedding_metrics],dir=dir,graph=label_propagation_dir.bigger_graph,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)
metrics_labelpropagation_dir_2 =   get_metrics_if_exist("lp_metric_dir_2",   [get_graph_metrics,get_embedding_metrics],dir=dir,graph=label_propagation_dir,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,X=X,y=y)


# %%
noises = [0,0.1,0.2,0.3]

# %%

noise = noises[1]
X_n1 = helper.generate_noise(X,noise=noise,seed=seed,data_name=data_name)
X_knn_n1 = sp.csr_array(sklearn.neighbors.kneighbors_graph(X_n1,n_neighbors=10))
X_knn_undirected_n1 = X_knn_n1.T + X_knn_n1
X_knn_undirected_n1.data = np.array(len(X_knn_undirected_n1.data)*[1])
u_knn_igraph_n1 = get_igraph_from_W(X_knn_undirected_n1,directed=False,weighted=True)


fcluster_walktrap_1_n1,fcluster_walktrap_2_n1, W_walktrap_1_n1,W_walktrap_2_n1 = create_or_load(fname="walktrap-connectivity_n1",function=create_walktrap_connectivity,graph=u_knn_igraph_n1)

embedding_walktrap_1_n1,landmarks_walktrap_1_n1 = create_or_load(fname="embedding_walktrap_1_n1",function=create_embedding,fcluster=fcluster_walktrap_1_n1,W=W_walktrap_1_n1,X=X_n1)
landmarks_labels_walktrap_1_n1 = y[landmarks_walktrap_1_n1]

embedding_walktrap_2_n1,landmarks_walktrap_2_n1= create_or_load(fname="embedding_walktrap_2_n1",function=create_embedding,fcluster=fcluster_walktrap_2_n1,W=W_walktrap_2_n1,X=X_n1,first_level=[landmarks_walktrap_1_n1])
landmark_labels_walktrap_2_n1 = y[landmarks_walktrap_2_n1]


label_propagation_udir_n1 = create_graph(sampling=random_sampling,connection=connectivi,X=X_n1,y=y,filename="lp-random-n1",dir=dir,level=2,seed=seed,directed=False,discrete=True)


fcluster_leiden_1_n1,fcluster_leiden_2_n1, W_leiden_1_n1,W_leiden_2_n1 = create_or_load(fname="leiden-connectivity_n1",function=create_leiden_connectivity,graph=u_knn_igraph_n1,res1=752,res2=67)

embedding_leiden_1_n1, landmarks_leiden_1_n1 = create_or_load(fname="embedding_leiden_1_n1",function=create_embedding,fcluster=fcluster_leiden_1_n1,W=W_leiden_1_n1,X=X_n1)
landmarks_labels_leiden_1_n1 = y[landmarks_leiden_1_n1]

embedding_leiden_2_n1, landmarks_leiden_2_n1 = create_or_load(fname="embedding_leiden_2_n1",function=create_embedding,fcluster=fcluster_leiden_2_n1,W=W_leiden_2_n1,X=X_n1,first_level=[landmarks_leiden_1_n1])
landmark_labels_leiden_2_n1 = y[landmarks_leiden_2_n1]


# %%
metrics_leiden_1_n1 =   get_metrics_if_exist("leiden_metric_1_n1",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_leiden_1_n1,landmarks=landmarks_leiden_1_n1,T=W_leiden_1_n1,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
metrics_leiden_2_n1 =   get_metrics_if_exist("leiden_metric_2_n1",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_leiden_2_n1,landmarks=landmarks_leiden_2_n1,T=W_leiden_2_n1,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)

metrics_walktrap_1_n1 =   get_metrics_if_exist("walktrap_metric_1_n1",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_walktrap_1_n1,landmarks=landmarks_walktrap_1_n1,T=W_walktrap_1_n1,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
metrics_walktrap_2_n1 =   get_metrics_if_exist("walktrap_metric_2_n1",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_walktrap_2_n1,landmarks=landmarks_walktrap_2_n1,T=W_walktrap_2_n1,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)

metrics_labelpropagation_1_n1 =   get_metrics_if_exist("lp_metric_1_n1",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=label_propagation_udir_n1.bigger_graph,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
metrics_labelpropagation_2_n1 =   get_metrics_if_exist("lp_metric_2_n1",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=label_propagation_udir_n1,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)


# %%

noise = noises[2]
X_n2 = helper.generate_noise(X,noise=noise,seed=seed,data_name=data_name)
X_knn_n2 = sp.csr_array(sklearn.neighbors.kneighbors_graph(X_n2,n_neighbors=10))
X_knn_undirected_n2 = X_knn_n2.T + X_knn_n2
X_knn_undirected_n2.data = np.array(len(X_knn_undirected_n2.data)*[1])
u_knn_igraph_n2 = get_igraph_from_W(X_knn_undirected_n2,directed=False,weighted=True)


fcluster_walktrap_1_n2,fcluster_walktrap_2_n2, W_walktrap_1_n2,W_walktrap_2_n2 = create_or_load(fname="walktrap-connectivity_n2",function=create_walktrap_connectivity,graph=u_knn_igraph_n2)

embedding_walktrap_1_n2,landmarks_walktrap_1_n2 = create_or_load(fname="embedding_walktrap_1_n2",function=create_embedding,fcluster=fcluster_walktrap_1_n2,W=W_walktrap_1_n2,X=X_n2)
landmarks_labels_walktrap_1_n2 = y[landmarks_walktrap_1_n2]

embedding_walktrap_2_n2,landmarks_walktrap_2_n2= create_or_load(fname="embedding_walktrap_2_n2",function=create_embedding,fcluster=fcluster_walktrap_2_n2,W=W_walktrap_2_n2,X=X_n2,first_level=[landmarks_walktrap_1_n2])
landmark_labels_walktrap_2_n2 = y[landmarks_walktrap_2_n2]


label_propagation_udir_n2 = create_graph(sampling=random_sampling,connection=connectivi,X=X_n2,y=y,filename="lp-random-n2",dir=dir,level=2,seed=seed,directed=False,discrete=True)


fcluster_leiden_1_n2,fcluster_leiden_2_n2, W_leiden_1_n2,W_leiden_2_n2 = create_or_load(fname="leiden-connectivity_n2",function=create_leiden_connectivity,graph=u_knn_igraph_n2,res1=752,res2=67)

embedding_leiden_1_n2, landmarks_leiden_1_n2 = create_or_load(fname="embedding_leiden_n2",function=create_embedding,fcluster=fcluster_leiden_1_n2,W=W_leiden_1_n2,X=X_n2)
landmarks_labels_leiden_1_n2 = y[landmarks_leiden_1_n2]

embedding_leiden_2_n2, landmarks_leiden_2_n2 = create_or_load(fname="embedding_leiden_2_n2",function=create_embedding,fcluster=fcluster_leiden_2_n2,W=W_leiden_2_n2,X=X_n2,first_level=[landmarks_leiden_1_n2])
landmark_labels_leiden_2_n2 = y[landmarks_leiden_2_n2]


# %%
metrics_leiden_1_n2 =   get_metrics_if_exist("leiden_metric_1_n2",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_leiden_1_n2,landmarks=landmarks_leiden_1_n2,T=W_leiden_1_n2,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
metrics_leiden_2_n2 =   get_metrics_if_exist("leiden_metric_2_n2",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_leiden_2_n2,landmarks=landmarks_leiden_2_n2,T=W_leiden_2_n2,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)

metrics_walktrap_1_n2 =   get_metrics_if_exist("walktrap_metric_1_n2",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_walktrap_1_n2,landmarks=landmarks_walktrap_1_n2,T=W_walktrap_1_n2,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
metrics_walktrap_2_n2 =   get_metrics_if_exist("walktrap_metric_2_n2",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_walktrap_2_n2,landmarks=landmarks_walktrap_2_n2,T=W_walktrap_2_n2,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)

metrics_labelpropagation_1_n2 =   get_metrics_if_exist("lp_metric_1_n2",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=label_propagation_udir_n2.bigger_graph,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
metrics_labelpropagation_2_n2 =   get_metrics_if_exist("lp_metric_2_n2",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=label_propagation_udir_n2,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)


# %%

noise = noises[3]
X_n3 = helper.generate_noise(X,noise=noise,seed=seed,data_name=data_name)
X_knn_n3 = sp.csr_array(sklearn.neighbors.kneighbors_graph(X_n3,n_neighbors=10))
X_knn_undirected_n3 = X_knn_n3.T + X_knn_n3
X_knn_undirected_n3.data = np.array(len(X_knn_undirected_n3.data)*[1])
u_knn_igraph_n3 = get_igraph_from_W(X_knn_undirected_n3,directed=False,weighted=True)


fcluster_walktrap_1_n3,fcluster_walktrap_2_n3, W_walktrap_1_n3,W_walktrap_2_n3 = create_or_load(fname="walktrap-connectivity_n3",function=create_walktrap_connectivity,graph=u_knn_igraph_n3)

embedding_walktrap_1_n3,landmarks_walktrap_1_n3 = create_or_load(fname="embedding_walktrap_1_n3",function=create_embedding,fcluster=fcluster_walktrap_1_n3,W=W_walktrap_1_n3,X=X_n3)
landmarks_labels_walktrap_1_n3 = y[landmarks_walktrap_1_n3]

embedding_walktrap_2_n3,landmarks_walktrap_2_n3= create_or_load(fname="embedding_walktrap_2_n3",function=create_embedding,fcluster=fcluster_walktrap_2_n3,W=W_walktrap_2_n3,X=X_n3,first_level=[landmarks_walktrap_1_n3])
landmark_labels_walktrap_2_n3 = y[landmarks_walktrap_2_n3]


label_propagation_udir_n3 = create_graph(sampling=random_sampling,connection=connectivi,X=X_n3,y=y,filename="lp-random-n2",dir=dir,level=2,seed=seed,directed=False,discrete=True)


fcluster_leiden_1_n3,fcluster_leiden_2_n3, W_leiden_1_n3,W_leiden_2_n3 = create_or_load(fname="leiden-connectivity_n3",function=create_leiden_connectivity,graph=u_knn_igraph_n3,res1=752,res2=67)

embedding_leiden_1_n3, landmarks_leiden_1_n3 = create_or_load(fname="embedding_leiden_n3",function=create_embedding,fcluster=fcluster_leiden_1_n3,W=W_leiden_1_n3,X=X_n3)
landmarks_labels_leiden_1_n3 = y[landmarks_leiden_1_n3]

embedding_leiden_2_n3, landmarks_leiden_2_n3 = create_or_load(fname="embedding_leiden_2_n3",function=create_embedding,fcluster=fcluster_leiden_2_n3,W=W_leiden_2_n3,X=X_n3,first_level=[landmarks_leiden_1_n3])
landmark_labels_leiden_2_n3 = y[landmarks_leiden_2_n3]


# %%
metrics_leiden_1_n3 =   get_metrics_if_exist("leiden_metric_1_n3",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_leiden_1_n3,landmarks=landmarks_leiden_1_n3,T=W_leiden_1_n3,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
metrics_leiden_2_n3 =   get_metrics_if_exist("leiden_metric_2_n3",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_leiden_2_n3,landmarks=landmarks_leiden_2_n3,T=W_leiden_2_n3,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)

metrics_walktrap_1_n3 =   get_metrics_if_exist("walktrap_metric_1_n3",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_walktrap_1_n3,landmarks=landmarks_walktrap_1_n3,T=W_walktrap_1_n2,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
metrics_walktrap_2_n3 =   get_metrics_if_exist("walktrap_metric_2_n3",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=embedding_walktrap_2_n3,landmarks=landmarks_walktrap_2_n3,T=W_walktrap_2_n2,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)

metrics_labelpropagation_1_n3 =   get_metrics_if_exist("lp_metric_1_n3",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=label_propagation_udir_n3.bigger_graph,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
metrics_labelpropagation_2_n3 =   get_metrics_if_exist("lp_metric_2_n3",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=label_propagation_udir_n3,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)


# %% [markdown]
# #TODO compute metrics

# %% [markdown]
# Comparison Loukas Local variation algorithm

# %%
import pygsp as gsp
from pygsp import graphs
from graph_coarsening import *
k=3 #like paper

# %%
def sample_landmarks(C,seed=seed):
    np.random.seed(seed)
    C.tocsr()
    n,N = C.shape
    landmarks = np.empty(n)
    
    for row in range(n):
        col_data = C.getrow(row)
        valid_rows = col_data.nonzero()[1]

        if len(valid_rows) == 0:
            raise ValueError(f"{row}")

        landmarks[row] = np.random.choice(valid_rows, size=1)[0]

    return landmarks.astype(int)

# %%
out = load_file_if_exists(f"local_variation_neighborhood")
if out is not None:
    (W_variation_neighborhood,landmarks_variation_neighborhood,emb_variation_neighborhood),(W_variation_neighborhood_2,landmarks_variation_neighborhood_2,emb_variation_neighborhood_2) = out
else:
    knn_graph = W_udir
    sources, targets = knn_graph.nonzero()
    knn_igraph = Graph(edges=list(zip(sources, targets)), directed=False)
    knn_pygsp_graph = graphs.Graph(knn_igraph.get_adjacency())
    method='variation_neighborhood'
    C_n, Gc_n, Call_n, Gall_n = coarsen(knn_pygsp_graph, K=k, r=0.5, method=method)
    C_n1, Gc_n1, Call_n1, Gall_n1 = coarsen(Gc_n, K=k, r=0.8, method=method)

    lvn = sample_landmarks(C_n)
    lvn1 = sample_landmarks(C_n1)
    landmarks_variation_neighborhood = lvn[lvn1]
    W_variation_neighborhood = sp.csr_array(Gc_n1.W)
    emb_variation_neighborhood = create_TSNE_embedding_from_Affinity(affinity=W_variation_neighborhood,X=X[landmarks_variation_neighborhood])

    C_n2, Gc_n2, Call_n2, Gall_n2 = coarsen(Gc_n1, K=k, r=1-reduction, method=method)
    landmarks_variation_neighborhood_2 = landmarks_variation_neighborhood[sample_landmarks(C_n2)]
    W_variation_neighborhood_2 = sp.csr_array(Gc_n2.W)
    emb_variation_neighborhood_2 = create_TSNE_embedding_from_Affinity(affinity=W_variation_neighborhood_2,X=X[landmarks_variation_neighborhood_2])
    save_pickle_dir(X = ((W_variation_neighborhood,landmarks_variation_neighborhood,emb_variation_neighborhood),(W_variation_neighborhood_2,landmarks_variation_neighborhood_2,emb_variation_neighborhood_2)),fname=f"local_variation_neighborhood")

# %%
out = load_file_if_exists(f"local_variation_edge")
if out is not None:
    (W_variation_edge,landmarks_variation_edge,emb_variation_edge),(W_variation_edge_2,landmarks_variation_edge_2,emb_variation_edge_2) = out
else:
    knn_graph = W_udir
    sources, targets = knn_graph.nonzero()
    knn_igraph = Graph(edges=list(zip(sources, targets)), directed=False)
    knn_pygsp_graph = graphs.Graph(knn_igraph.get_adjacency())
    method='variation_edges'
    C_e, Gc_e, Call_e, Gall_e = coarsen(knn_pygsp_graph, K=k, r=1-reduction, method=method)
    landmarks_variation_edge = sample_landmarks(C_e,seed=seed+1)
    W_variation_edge = sp.csr_array(Gc_e.W)
    emb_variation_edge = create_TSNE_embedding_from_Affinity(affinity=W_variation_edge,X=X[landmarks_variation_edge])

    C_e2, Gc_e2, Call_e2, Gall_e2 = coarsen(Gc_e, K=k, r=1-reduction, method=method)
    landmarks_variation_edge_2 = landmarks_variation_edge[sample_landmarks(C_e2)]
    W_variation_edge_2 = sp.csr_array(Gc_e2.W)
    emb_variation_edge_2 = create_TSNE_embedding_from_Affinity(affinity=W_variation_edge_2,X=X[landmarks_variation_edge_2])
    save_pickle_dir(X = ((W_variation_edge,landmarks_variation_edge,emb_variation_edge),(W_variation_edge_2,landmarks_variation_edge_2,emb_variation_edge_2)),fname=f"local_variation_edge")

# %%
embeddings = [
    (emb_variation_edge,landmarks_variation_edge),(emb_variation_edge_2,landmarks_variation_edge_2)
              ]
names = ["Level:1","Level:2"]
plg.plot_embeddings_row_discrete(embeddings,names,y=y,colors=colors,fsave=f"{data_name}_local_variation_edge.png")

# %%
embeddings = [
    (emb_variation_neighborhood,landmarks_variation_neighborhood),(emb_variation_neighborhood_2,landmarks_variation_neighborhood_2)
              ]
names = ["Level:1","Level:2"]
plg.plot_embeddings_row_discrete(embeddings,names,y,colors=colors,fsave=f"{data_name}_local_variation_neighborhood.png")

# %%
embeddings = [
    [(emb_variation_edge,landmarks_variation_edge),(emb_variation_edge_2,landmarks_variation_edge_2)],
    [(emb_variation_neighborhood,landmarks_variation_neighborhood),(emb_variation_neighborhood_2,landmarks_variation_neighborhood_2)]
              ]
titles = ["Local variation edge","Local variation neighborhood"]
names = ["Level:1","Level:2"]
plg.plot_embeddings_mosaic_discrete(embeddings,titles,names,y,colors=colors,fsave=f"{data_name}_local_variation_comparison.png")

# %%
metrics_lv_neighborhood_1 =   get_metrics_if_exist("lv_neighborhood_metric_1",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=emb_variation_neighborhood  ,landmarks=landmarks_variation_neighborhood  ,T=W_variation_neighborhood ,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected ,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
metrics_lv_neighborhood_2 =   get_metrics_if_exist("lv_neighborhood_metric_2",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=emb_variation_neighborhood_2,landmarks=landmarks_variation_neighborhood_2,T=W_variation_neighborhood_2,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)

# %%
metrics_lv_edge_1 =   get_metrics_if_exist("lv_edge_metric_1",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=emb_variation_edge  ,landmarks=landmarks_variation_edge  ,T=W_variation_edge  ,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
metrics_lv_edge_2 =   get_metrics_if_exist("lv_edge_metric_2",   [get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],dir=dir,graph=None,emb=emb_variation_edge_2,landmarks=landmarks_variation_edge_2,T=W_variation_edge_2,closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)

# %%
#print(metrics_lv_edge_1)
#print(metrics_lv_edge_2)
#
#print(metrics_lv_neighborhood_1)
#print(metrics_lv_neighborhood_2)


