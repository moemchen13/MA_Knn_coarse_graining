
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
import openTSNE

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

np.random.seed(42)
data_name = "MNIST"


dataset_directory = f"./datasets/{data_name}"
dir = f"./results/{data_name}"

#colors = np.load(join(dataset_directory,"tasic-colors.npy"))

#cmap = plt.get_cmap('coolwarm')
#colors = [cmap(i) for i in [0.1,0.15,0.95,0.9,0.85,0.8,0.75,0.7,0.2,0.25]]

#colors = plt.cm.tab20.colors
colors = plt.cm.tab10.colors
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

def dataframe_min_max_latex_rows(df,data_name=data_name):
    average = [f"{val:.4f}" for val in df.mean()]
    stds = [f"{val:.4f}" for val in df.std()]
    columns = ["&"+ str(col) for col in df.columns]
    data_name + ": &".join(str(col) for col in columns) +r"\\ \hline \n"
    header = data_name + ": " + " & ".join(str(col) for col in columns) + r"\\ \hline"
    row = data_name + " & " + ' & '.join(val_avg + r"(\pm"+ val_std + ")" for val_avg, val_std in zip(average, stds)) + r"\\ \hline"
    row = header + row
    return row

# %%

# %%

def get_metric_if_exist(fname,function,dir=dir,**kwargs):
    if os.path.isfile(join(dir,fname)):
        print("found_file")
        return load_from_pickle(join(dir,fname))
    
    print(f"File not found: {fname}, create metric")
    output = function(**kwargs)
    save_pickle(output,join(dir,fname))
    return output

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


def save_pickle_dir(fname,X,dir=dir):
        save_pickle(X,join(dir,fname))


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


def get_subset(X,y,seed,r=0.1):
    np.random.seed(seed)
    N = X.shape[0]
    n = int(N*r)
    subset = np.random.choice(N,size=n,replace=False)
    X_subset = X[subset]
    y_subset = y[subset]
    data = (X_subset,y_subset,subset)
    X_knn = sp.csr_array(sklearn.neighbors.kneighbors_graph(X_subset,n_neighbors=10))
    X_knn_undirected = X_knn.T + X_knn
    X_knn_undirected.data = np.array(len(X_knn_undirected.data)*[1])
    X_knn_undirected = normalize(X_knn_undirected,axis=1,norm="l1")
    emb_udir = create_TSNE_embedding_from_Affinity(X_knn_undirected,X_subset)
    undirected = X_knn_undirected,emb_udir

    X_knn_directed = X_knn    
    X_knn_directed = normalize(X_knn_directed,axis=1,norm="l1")
    emb_dir = create_TSNE_embedding_from_Affinity(X_knn_directed,X_subset)
    directed = X_knn_directed,emb_dir


    return data,undirected,directed


def compute_metrics(functions,**kwargs):
    out = []
    for function in functions:
        output = function(**kwargs)
        out.append(output)
    return out

# %%
np.random.seed(42)
data_name = "MNIST"
dataset_directory = f"./datasets/{data_name}"
dir = f"./results/{data_name}"

# %%

def get_metric_if_exist(fname,function,dir=dir,**kwargs):
    if os.path.isfile(join(dir,fname)):
        print("found_file")
        return load_from_pickle(join(dir,fname))
    
    print(f"File not found: {fname}, create metric")
    output = function(**kwargs)
    save_pickle(output,join(dir,fname))
    return output

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


def save_pickle_dir(fname,X,dir=dir):
        save_pickle(X,join(dir,fname))


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


def get_subset(X,y,seed,r=0.1):
    np.random.seed(seed)
    N = X.shape[0]
    n = int(N*r)
    subset = np.random.choice(N,size=n,replace=False)
    X_subset = X[subset]
    y_subset = y[subset]
    data = (X_subset,y_subset,subset)
    X_knn = sp.csr_array(sklearn.neighbors.kneighbors_graph(X_subset,n_neighbors=10))
    X_knn_undirected = X_knn.T + X_knn
    X_knn_undirected.data = np.array(len(X_knn_undirected.data)*[1])
    X_knn_undirected = normalize(X_knn_undirected,axis=1,norm="l1")
    emb_udir = create_TSNE_embedding_from_Affinity(X_knn_undirected,X_subset)
    undirected = X_knn_undirected,emb_udir

    X_knn_directed = X_knn    
    X_knn_directed = normalize(X_knn_directed,axis=1,norm="l1")
    emb_dir = create_TSNE_embedding_from_Affinity(X_knn_directed,X_subset)
    directed = X_knn_directed,emb_dir


    return data,undirected,directed


def compute_metrics(functions,**kwargs):
    out = []
    for function in functions:
        output = function(**kwargs)
        out.append(output)
    return out

# %%

X,y,class_table = data_loader.select_dataset(kind=data_name,directory=dataset_directory)

# %%
closeness_knn_undirected = load_file_if_exists(f"{data_name}_closeness.pkl",dir)
betweenness_knn_undirected = load_file_if_exists(f"{data_name}_betweenness.pkl",dir)
eigenvalues_big_undirected = load_file_if_exists(f"{data_name}_eigenvalues_big_undirected.pkl",dir)
closeness_knn_directed = load_file_if_exists(f"{data_name}_closeness_directed.pkl",dir)
betweenness_knn_directed = load_file_if_exists(f"{data_name}_betweenness_directed.pkl",dir)

# %%
metrics_names = ["dspectral","rel. eigenerr","centrality","betweenness","kNN-Accuracy","Trustworthiness","Silhouette","DBI"]

directed_frame = pd.DataFrame(np.nan,index= range(5),columns=metrics_names[1:])
undirected_frame = pd.DataFrame(np.nan,index=range(5),columns=metrics_names)
directed_frame_2 = pd.DataFrame(np.nan,index= range(5),columns=metrics_names[1:])
undirected_frame_2 = pd.DataFrame(np.nan,index=range(5),columns=metrics_names)

for i in range(5):
    (X_subset,y_subset,landmarks),udir,dir = get_subset(X,y,seed=i)

    metrics_udir =   compute_metrics([get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=None,emb=udir[1],landmarks=landmarks,T=udir[0],closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
    undirected_frame.loc[i,["dspectral","rel. eigenerr"]] = metrics_udir[0]
    undirected_frame.loc[i,["centrality","betweenness"]] = metrics_udir[1]
    undirected_frame.loc[i,["kNN-Accuracy","Trustworthiness","Silhouette","DBI"]] = metrics_udir[2]

    metrics_dir =   compute_metrics([get_graph_metrics,get_embedding_metrics],graph=None,emb=dir[-1],landmarks=landmarks,T=dir[0],closeness_knn=closeness_knn_directed,betweenness_knn=betweenness_knn_directed,X=X,y=y)
    directed_frame.loc[i,["centrality","betweenness"]] = metrics_dir[0]
    directed_frame.loc[i,["kNN-Accuracy","Trustworthiness","Silhouette","DBI"]] = metrics_dir[1]


    (X_subsubset,y_subsubset,landmarks_subset),udir_subset,dir_subset = get_subset(X_subset,y_subset,seed=i)
    metrics_udir =   compute_metrics([get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=None,emb=udir_subset[-1],landmarks=landmarks_subset,T=udir_subset[0],closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X_subset,y=y_subset)

    undirected_frame_2.loc[i,["dspectral","rel. eigenerr"]] = metrics_udir[0]
    undirected_frame_2.loc[i,["centrality","betweenness"]] = metrics_udir[1]
    undirected_frame_2.loc[i,["kNN-Accuracy","Trustworthiness","Silhouette","DBI"]] = metrics_udir[2]
    
    metrics_dir =   compute_metrics([get_graph_metrics,get_embedding_metrics],graph=None,emb=dir_subset[-1],landmarks=landmarks_subset,T=dir_subset[0],closeness_knn=closeness_knn_directed,betweenness_knn=betweenness_knn_directed,X=X_subset,y=y_subset)
    directed_frame_2.loc[i,["centrality","betweenness"]] = metrics_dir[0]
    directed_frame_2.loc[i,["kNN-Accuracy","Trustworthiness","Silhouette","DBI"]] = metrics_dir[1]

print('############################################')
print(data_name)
print("Level 1")
print("directed")
print(directed_frame.max())
print(directed_frame.mean())
print(directed_frame.min())

print("undirected")
print(undirected_frame.max())
print(undirected_frame.mean())
print(undirected_frame.min())

print("Level 2")
print("directed")
print(directed_frame_2.max())
print(directed_frame_2.mean())
print(directed_frame_2.min())

print("undirected")
print(undirected_frame_2.max())
print(undirected_frame_2.mean())
print(undirected_frame_2.min())

print('############################################')


# %%

print("Level 1")
print("directed")
print(dataframe_min_max_latex_rows(directed_frame))
print("undirected")
print(dataframe_min_max_latex_rows(undirected_frame))


print("Level 2")
print("directed")
print(dataframe_min_max_latex_rows(directed_frame_2))
print("undirected")
print(dataframe_min_max_latex_rows(undirected_frame_2))



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
import openTSNE

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

np.random.seed(42)
data_name = "FMNIST"


dataset_directory = f"./datasets/{data_name}"
dir = f"./results/{data_name}"

#colors = np.load(join(dataset_directory,"tasic-colors.npy"))

#cmap = plt.get_cmap('coolwarm')
#colors = [cmap(i) for i in [0.1,0.15,0.95,0.9,0.85,0.8,0.75,0.7,0.2,0.25]]

#colors = plt.cm.tab20.colors
colors = plt.cm.tab10.colors
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

def dataframe_min_max_latex_rows(df,data_name=data_name):
    average = [f"{val:.4f}" for val in df.mean()]
    stds = [f"{val:.4f}" for val in df.std()]
    columns = ["&"+ str(col) for col in df.columns]
    data_name + ": &".join(str(col) for col in columns) +r"\\ \hline \n"
    header = data_name + ": " + " & ".join(str(col) for col in columns) + r"\\ \hline"
    row = data_name + " & " + ' & '.join(val_avg + r"(\pm"+ val_std + ")" for val_avg, val_std in zip(average, stds)) + r"\\ \hline"
    row = header + row
    return row

# %%

# %%

def get_metric_if_exist(fname,function,dir=dir,**kwargs):
    if os.path.isfile(join(dir,fname)):
        print("found_file")
        return load_from_pickle(join(dir,fname))
    
    print(f"File not found: {fname}, create metric")
    output = function(**kwargs)
    save_pickle(output,join(dir,fname))
    return output

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


def save_pickle_dir(fname,X,dir=dir):
        save_pickle(X,join(dir,fname))


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


def get_subset(X,y,seed,r=0.1):
    np.random.seed(seed)
    N = X.shape[0]
    n = int(N*r)
    subset = np.random.choice(N,size=n,replace=False)
    X_subset = X[subset]
    y_subset = y[subset]
    data = (X_subset,y_subset,subset)
    X_knn = sp.csr_array(sklearn.neighbors.kneighbors_graph(X_subset,n_neighbors=10))
    X_knn_undirected = X_knn.T + X_knn
    X_knn_undirected.data = np.array(len(X_knn_undirected.data)*[1])
    X_knn_undirected = normalize(X_knn_undirected,axis=1,norm="l1")
    emb_udir = create_TSNE_embedding_from_Affinity(X_knn_undirected,X_subset)
    undirected = X_knn_undirected,emb_udir

    X_knn_directed = X_knn    
    X_knn_directed = normalize(X_knn_directed,axis=1,norm="l1")
    emb_dir = create_TSNE_embedding_from_Affinity(X_knn_directed,X_subset)
    directed = X_knn_directed,emb_dir


    return data,undirected,directed


def compute_metrics(functions,**kwargs):
    out = []
    for function in functions:
        output = function(**kwargs)
        out.append(output)
    return out

# %%
np.random.seed(42)
data_name = "FMNIST"
dataset_directory = f"./datasets/{data_name}"
dir = f"./results/{data_name}"

# %%

def get_metric_if_exist(fname,function,dir=dir,**kwargs):
    if os.path.isfile(join(dir,fname)):
        print("found_file")
        return load_from_pickle(join(dir,fname))
    
    print(f"File not found: {fname}, create metric")
    output = function(**kwargs)
    save_pickle(output,join(dir,fname))
    return output

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


def save_pickle_dir(fname,X,dir=dir):
        save_pickle(X,join(dir,fname))


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


def get_subset(X,y,seed,r=0.1):
    np.random.seed(seed)
    N = X.shape[0]
    n = int(N*r)
    subset = np.random.choice(N,size=n,replace=False)
    X_subset = X[subset]
    y_subset = y[subset]
    data = (X_subset,y_subset,subset)
    X_knn = sp.csr_array(sklearn.neighbors.kneighbors_graph(X_subset,n_neighbors=10))
    X_knn_undirected = X_knn.T + X_knn
    X_knn_undirected.data = np.array(len(X_knn_undirected.data)*[1])
    X_knn_undirected = normalize(X_knn_undirected,axis=1,norm="l1")
    emb_udir = create_TSNE_embedding_from_Affinity(X_knn_undirected,X_subset)
    undirected = X_knn_undirected,emb_udir

    X_knn_directed = X_knn    
    X_knn_directed = normalize(X_knn_directed,axis=1,norm="l1")
    emb_dir = create_TSNE_embedding_from_Affinity(X_knn_directed,X_subset)
    directed = X_knn_directed,emb_dir


    return data,undirected,directed


def compute_metrics(functions,**kwargs):
    out = []
    for function in functions:
        output = function(**kwargs)
        out.append(output)
    return out

# %%

X,y,class_table = data_loader.select_dataset(kind=data_name,directory=dataset_directory)

# %%
closeness_knn_undirected = load_file_if_exists(f"{data_name}_closeness.pkl",dir)
betweenness_knn_undirected = load_file_if_exists(f"{data_name}_betweenness.pkl",dir)
eigenvalues_big_undirected = load_file_if_exists(f"{data_name}_eigenvalues_big_undirected.pkl",dir)
closeness_knn_directed = load_file_if_exists(f"{data_name}_closeness_directed.pkl",dir)
betweenness_knn_directed = load_file_if_exists(f"{data_name}_betweenness_directed.pkl",dir)

# %%
metrics_names = ["dspectral","rel. eigenerr","centrality","betweenness","kNN-Accuracy","Trustworthiness","Silhouette","DBI"]

directed_frame = pd.DataFrame(np.nan,index= range(5),columns=metrics_names[1:])
undirected_frame = pd.DataFrame(np.nan,index=range(5),columns=metrics_names)
directed_frame_2 = pd.DataFrame(np.nan,index= range(5),columns=metrics_names[1:])
undirected_frame_2 = pd.DataFrame(np.nan,index=range(5),columns=metrics_names)

for i in range(5):
    (X_subset,y_subset,landmarks),udir,dir = get_subset(X,y,seed=i)

    metrics_udir =   compute_metrics([get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=None,emb=udir[1],landmarks=landmarks,T=udir[0],closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
    undirected_frame.loc[i,["dspectral","rel. eigenerr"]] = metrics_udir[0]
    undirected_frame.loc[i,["centrality","betweenness"]] = metrics_udir[1]
    undirected_frame.loc[i,["kNN-Accuracy","Trustworthiness","Silhouette","DBI"]] = metrics_udir[2]

    metrics_dir =   compute_metrics([get_graph_metrics,get_embedding_metrics],graph=None,emb=dir[-1],landmarks=landmarks,T=dir[0],closeness_knn=closeness_knn_directed,betweenness_knn=betweenness_knn_directed,X=X,y=y)
    directed_frame.loc[i,["centrality","betweenness"]] = metrics_dir[0]
    directed_frame.loc[i,["kNN-Accuracy","Trustworthiness","Silhouette","DBI"]] = metrics_dir[1]


    (X_subsubset,y_subsubset,landmarks_subset),udir_subset,dir_subset = get_subset(X_subset,y_subset,seed=i)
    metrics_udir =   compute_metrics([get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=None,emb=udir_subset[-1],landmarks=landmarks_subset,T=udir_subset[0],closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X_subset,y=y_subset)

    undirected_frame_2.loc[i,["dspectral","rel. eigenerr"]] = metrics_udir[0]
    undirected_frame_2.loc[i,["centrality","betweenness"]] = metrics_udir[1]
    undirected_frame_2.loc[i,["kNN-Accuracy","Trustworthiness","Silhouette","DBI"]] = metrics_udir[2]
    
    metrics_dir =   compute_metrics([get_graph_metrics,get_embedding_metrics],graph=None,emb=dir_subset[-1],landmarks=landmarks_subset,T=dir_subset[0],closeness_knn=closeness_knn_directed,betweenness_knn=betweenness_knn_directed,X=X_subset,y=y_subset)
    directed_frame_2.loc[i,["centrality","betweenness"]] = metrics_dir[0]
    directed_frame_2.loc[i,["kNN-Accuracy","Trustworthiness","Silhouette","DBI"]] = metrics_dir[1]

print('############################################')
print(data_name)
print("Level 1")
print("directed")
print(directed_frame.max())
print(directed_frame.mean())
print(directed_frame.min())

print("undirected")
print(undirected_frame.max())
print(undirected_frame.mean())
print(undirected_frame.min())

print("Level 2")
print("directed")
print(directed_frame_2.max())
print(directed_frame_2.mean())
print(directed_frame_2.min())

print("undirected")
print(undirected_frame_2.max())
print(undirected_frame_2.mean())
print(undirected_frame_2.min())

print('############################################')


# %%

print("Level 1")
print("directed")
print(dataframe_min_max_latex_rows(directed_frame))
print("undirected")
print(dataframe_min_max_latex_rows(undirected_frame))


print("Level 2")
print("directed")
print(dataframe_min_max_latex_rows(directed_frame_2))
print("undirected")
print(dataframe_min_max_latex_rows(undirected_frame_2))


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
import openTSNE

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

np.random.seed(42)
data_name = "20NEWSGROUP"


dataset_directory = f"./datasets/{data_name}"
dir = f"./results/{data_name}"

#colors = np.load(join(dataset_directory,"tasic-colors.npy"))

#cmap = plt.get_cmap('coolwarm')
#colors = [cmap(i) for i in [0.1,0.15,0.95,0.9,0.85,0.8,0.75,0.7,0.2,0.25]]

#colors = plt.cm.tab20.colors
colors = plt.cm.tab10.colors
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

def dataframe_min_max_latex_rows(df,data_name=data_name):
    average = [f"{val:.4f}" for val in df.mean()]
    stds = [f"{val:.4f}" for val in df.std()]
    columns = ["&"+ str(col) for col in df.columns]
    data_name + ": &".join(str(col) for col in columns) +r"\\ \hline \n"
    header = data_name + ": " + " & ".join(str(col) for col in columns) + r"\\ \hline"
    row = data_name + " & " + ' & '.join(val_avg + r"(\pm"+ val_std + ")" for val_avg, val_std in zip(average, stds)) + r"\\ \hline"
    row = header + row
    return row

# %%

# %%

def get_metric_if_exist(fname,function,dir=dir,**kwargs):
    if os.path.isfile(join(dir,fname)):
        print("found_file")
        return load_from_pickle(join(dir,fname))
    
    print(f"File not found: {fname}, create metric")
    output = function(**kwargs)
    save_pickle(output,join(dir,fname))
    return output

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


def save_pickle_dir(fname,X,dir=dir):
        save_pickle(X,join(dir,fname))


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


def get_subset(X,y,seed,r=0.1):
    np.random.seed(seed)
    N = X.shape[0]
    n = int(N*r)
    subset = np.random.choice(N,size=n,replace=False)
    X_subset = X[subset]
    y_subset = y[subset]
    data = (X_subset,y_subset,subset)
    X_knn = sp.csr_array(sklearn.neighbors.kneighbors_graph(X_subset,n_neighbors=10))
    X_knn_undirected = X_knn.T + X_knn
    X_knn_undirected.data = np.array(len(X_knn_undirected.data)*[1])
    X_knn_undirected = normalize(X_knn_undirected,axis=1,norm="l1")
    emb_udir = create_TSNE_embedding_from_Affinity(X_knn_undirected,X_subset)
    undirected = X_knn_undirected,emb_udir

    X_knn_directed = X_knn    
    X_knn_directed = normalize(X_knn_directed,axis=1,norm="l1")
    emb_dir = create_TSNE_embedding_from_Affinity(X_knn_directed,X_subset)
    directed = X_knn_directed,emb_dir


    return data,undirected,directed


def compute_metrics(functions,**kwargs):
    out = []
    for function in functions:
        output = function(**kwargs)
        out.append(output)
    return out

# %%
np.random.seed(42)
data_name = "20NEWSGROUP"
dataset_directory = f"./datasets/{data_name}"
dir = f"./results/{data_name}"

# %%

def get_metric_if_exist(fname,function,dir=dir,**kwargs):
    if os.path.isfile(join(dir,fname)):
        print("found_file")
        return load_from_pickle(join(dir,fname))
    
    print(f"File not found: {fname}, create metric")
    output = function(**kwargs)
    save_pickle(output,join(dir,fname))
    return output

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


def save_pickle_dir(fname,X,dir=dir):
        save_pickle(X,join(dir,fname))


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


def get_subset(X,y,seed,r=0.1):
    np.random.seed(seed)
    N = X.shape[0]
    n = int(N*r)
    subset = np.random.choice(N,size=n,replace=False)
    X_subset = X[subset]
    y_subset = y[subset]
    data = (X_subset,y_subset,subset)
    X_knn = sp.csr_array(sklearn.neighbors.kneighbors_graph(X_subset,n_neighbors=10))
    X_knn_undirected = X_knn.T + X_knn
    X_knn_undirected.data = np.array(len(X_knn_undirected.data)*[1])
    X_knn_undirected = normalize(X_knn_undirected,axis=1,norm="l1")
    emb_udir = create_TSNE_embedding_from_Affinity(X_knn_undirected,X_subset)
    undirected = X_knn_undirected,emb_udir

    X_knn_directed = X_knn    
    X_knn_directed = normalize(X_knn_directed,axis=1,norm="l1")
    emb_dir = create_TSNE_embedding_from_Affinity(X_knn_directed,X_subset)
    directed = X_knn_directed,emb_dir


    return data,undirected,directed


def compute_metrics(functions,**kwargs):
    out = []
    for function in functions:
        output = function(**kwargs)
        out.append(output)
    return out

# %%

X,y,class_table = data_loader.select_dataset(kind=data_name,directory=dataset_directory)

# %%
closeness_knn_undirected = load_file_if_exists(f"{data_name}_closeness.pkl",dir)
betweenness_knn_undirected = load_file_if_exists(f"{data_name}_betweenness.pkl",dir)
eigenvalues_big_undirected = load_file_if_exists(f"{data_name}_eigenvalues_big_undirected.pkl",dir)
closeness_knn_directed = load_file_if_exists(f"{data_name}_closeness_directed.pkl",dir)
betweenness_knn_directed = load_file_if_exists(f"{data_name}_betweenness_directed.pkl",dir)

# %%
metrics_names = ["dspectral","rel. eigenerr","centrality","betweenness","kNN-Accuracy","Trustworthiness","Silhouette","DBI"]

directed_frame = pd.DataFrame(np.nan,index= range(5),columns=metrics_names[1:])
undirected_frame = pd.DataFrame(np.nan,index=range(5),columns=metrics_names)
directed_frame_2 = pd.DataFrame(np.nan,index= range(5),columns=metrics_names[1:])
undirected_frame_2 = pd.DataFrame(np.nan,index=range(5),columns=metrics_names)

for i in range(5):
    (X_subset,y_subset,landmarks),udir,dir = get_subset(X,y,seed=i)

    metrics_udir =   compute_metrics([get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=None,emb=udir[1],landmarks=landmarks,T=udir[0],closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
    undirected_frame.loc[i,["dspectral","rel. eigenerr"]] = metrics_udir[0]
    undirected_frame.loc[i,["centrality","betweenness"]] = metrics_udir[1]
    undirected_frame.loc[i,["kNN-Accuracy","Trustworthiness","Silhouette","DBI"]] = metrics_udir[2]

    metrics_dir =   compute_metrics([get_graph_metrics,get_embedding_metrics],graph=None,emb=dir[-1],landmarks=landmarks,T=dir[0],closeness_knn=closeness_knn_directed,betweenness_knn=betweenness_knn_directed,X=X,y=y)
    directed_frame.loc[i,["centrality","betweenness"]] = metrics_dir[0]
    directed_frame.loc[i,["kNN-Accuracy","Trustworthiness","Silhouette","DBI"]] = metrics_dir[1]


    (X_subsubset,y_subsubset,landmarks_subset),udir_subset,dir_subset = get_subset(X_subset,y_subset,seed=i)
    metrics_udir =   compute_metrics([get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=None,emb=udir_subset[-1],landmarks=landmarks_subset,T=udir_subset[0],closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X_subset,y=y_subset)

    undirected_frame_2.loc[i,["dspectral","rel. eigenerr"]] = metrics_udir[0]
    undirected_frame_2.loc[i,["centrality","betweenness"]] = metrics_udir[1]
    undirected_frame_2.loc[i,["kNN-Accuracy","Trustworthiness","Silhouette","DBI"]] = metrics_udir[2]
    
    metrics_dir =   compute_metrics([get_graph_metrics,get_embedding_metrics],graph=None,emb=dir_subset[-1],landmarks=landmarks_subset,T=dir_subset[0],closeness_knn=closeness_knn_directed,betweenness_knn=betweenness_knn_directed,X=X_subset,y=y_subset)
    directed_frame_2.loc[i,["centrality","betweenness"]] = metrics_dir[0]
    directed_frame_2.loc[i,["kNN-Accuracy","Trustworthiness","Silhouette","DBI"]] = metrics_dir[1]

print('############################################')
print(data_name)
print("Level 1")
print("directed")
print(directed_frame.max())
print(directed_frame.mean())
print(directed_frame.min())

print("undirected")
print(undirected_frame.max())
print(undirected_frame.mean())
print(undirected_frame.min())

print("Level 2")
print("directed")
print(directed_frame_2.max())
print(directed_frame_2.mean())
print(directed_frame_2.min())

print("undirected")
print(undirected_frame_2.max())
print(undirected_frame_2.mean())
print(undirected_frame_2.min())

print('############################################')


# %%

print("Level 1")
print("directed")
print(dataframe_min_max_latex_rows(directed_frame))
print("undirected")
print(dataframe_min_max_latex_rows(undirected_frame))


print("Level 2")
print("directed")
print(dataframe_min_max_latex_rows(directed_frame_2))
print("undirected")
print(dataframe_min_max_latex_rows(undirected_frame_2))



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
import openTSNE

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

np.random.seed(42)
data_name = "TASIC"


dataset_directory = f"./datasets/{data_name}"
dir = f"./results/{data_name}"

#colors = np.load(join(dataset_directory,"tasic-colors.npy"))

#cmap = plt.get_cmap('coolwarm')
#colors = [cmap(i) for i in [0.1,0.15,0.95,0.9,0.85,0.8,0.75,0.7,0.2,0.25]]

#colors = plt.cm.tab20.colors
colors = plt.cm.tab10.colors
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

def dataframe_min_max_latex_rows(df,data_name=data_name):
    average = [f"{val:.4f}" for val in df.mean()]
    stds = [f"{val:.4f}" for val in df.std()]
    columns = ["&"+ str(col) for col in df.columns]
    data_name + ": &".join(str(col) for col in columns) +r"\\ \hline \n"
    header = data_name + ": " + " & ".join(str(col) for col in columns) + r"\\ \hline"
    row = data_name + " & " + ' & '.join(val_avg + r"(\pm"+ val_std + ")" for val_avg, val_std in zip(average, stds)) + r"\\ \hline"
    row = header + row
    return row

# %%

# %%

def get_metric_if_exist(fname,function,dir=dir,**kwargs):
    if os.path.isfile(join(dir,fname)):
        print("found_file")
        return load_from_pickle(join(dir,fname))
    
    print(f"File not found: {fname}, create metric")
    output = function(**kwargs)
    save_pickle(output,join(dir,fname))
    return output

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


def save_pickle_dir(fname,X,dir=dir):
        save_pickle(X,join(dir,fname))


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


def get_subset(X,y,seed,r=0.1):
    np.random.seed(seed)
    N = X.shape[0]
    n = int(N*r)
    subset = np.random.choice(N,size=n,replace=False)
    X_subset = X[subset]
    y_subset = y[subset]
    data = (X_subset,y_subset,subset)
    X_knn = sp.csr_array(sklearn.neighbors.kneighbors_graph(X_subset,n_neighbors=10))
    X_knn_undirected = X_knn.T + X_knn
    X_knn_undirected.data = np.array(len(X_knn_undirected.data)*[1])
    X_knn_undirected = normalize(X_knn_undirected,axis=1,norm="l1")
    emb_udir = create_TSNE_embedding_from_Affinity(X_knn_undirected,X_subset)
    undirected = X_knn_undirected,emb_udir

    X_knn_directed = X_knn    
    X_knn_directed = normalize(X_knn_directed,axis=1,norm="l1")
    emb_dir = create_TSNE_embedding_from_Affinity(X_knn_directed,X_subset)
    directed = X_knn_directed,emb_dir


    return data,undirected,directed


def compute_metrics(functions,**kwargs):
    out = []
    for function in functions:
        output = function(**kwargs)
        out.append(output)
    return out

# %%
np.random.seed(42)
data_name = "TASIC"
dataset_directory = f"./datasets/{data_name}"
dir = f"./results/{data_name}"

# %%

def get_metric_if_exist(fname,function,dir=dir,**kwargs):
    if os.path.isfile(join(dir,fname)):
        print("found_file")
        return load_from_pickle(join(dir,fname))
    
    print(f"File not found: {fname}, create metric")
    output = function(**kwargs)
    save_pickle(output,join(dir,fname))
    return output

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


def save_pickle_dir(fname,X,dir=dir):
        save_pickle(X,join(dir,fname))


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


def get_subset(X,y,seed,r=0.1):
    np.random.seed(seed)
    N = X.shape[0]
    n = int(N*r)
    subset = np.random.choice(N,size=n,replace=False)
    X_subset = X[subset]
    y_subset = y[subset]
    data = (X_subset,y_subset,subset)
    X_knn = sp.csr_array(sklearn.neighbors.kneighbors_graph(X_subset,n_neighbors=10))
    X_knn_undirected = X_knn.T + X_knn
    X_knn_undirected.data = np.array(len(X_knn_undirected.data)*[1])
    X_knn_undirected = normalize(X_knn_undirected,axis=1,norm="l1")
    emb_udir = create_TSNE_embedding_from_Affinity(X_knn_undirected,X_subset)
    undirected = X_knn_undirected,emb_udir

    X_knn_directed = X_knn    
    X_knn_directed = normalize(X_knn_directed,axis=1,norm="l1")
    emb_dir = create_TSNE_embedding_from_Affinity(X_knn_directed,X_subset)
    directed = X_knn_directed,emb_dir


    return data,undirected,directed


def compute_metrics(functions,**kwargs):
    out = []
    for function in functions:
        output = function(**kwargs)
        out.append(output)
    return out

# %%

X,y,class_table = data_loader.select_dataset(kind=data_name,directory=dataset_directory)

# %%
closeness_knn_undirected = load_file_if_exists(f"{data_name}_closeness.pkl",dir)
betweenness_knn_undirected = load_file_if_exists(f"{data_name}_betweenness.pkl",dir)
eigenvalues_big_undirected = load_file_if_exists(f"{data_name}_eigenvalues_big_undirected.pkl",dir)
closeness_knn_directed = load_file_if_exists(f"{data_name}_closeness_directed.pkl",dir)
betweenness_knn_directed = load_file_if_exists(f"{data_name}_betweenness_directed.pkl",dir)

# %%
metrics_names = ["dspectral","rel. eigenerr","centrality","betweenness","kNN-Accuracy","Trustworthiness","Silhouette","DBI"]

directed_frame = pd.DataFrame(np.nan,index= range(5),columns=metrics_names[1:])
undirected_frame = pd.DataFrame(np.nan,index=range(5),columns=metrics_names)
directed_frame_2 = pd.DataFrame(np.nan,index= range(5),columns=metrics_names[1:])
undirected_frame_2 = pd.DataFrame(np.nan,index=range(5),columns=metrics_names)

for i in range(5):
    (X_subset,y_subset,landmarks),udir,dir = get_subset(X,y,seed=i)

    metrics_udir =   compute_metrics([get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=None,emb=udir[1],landmarks=landmarks,T=udir[0],closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X,y=y)
    undirected_frame.loc[i,["dspectral","rel. eigenerr"]] = metrics_udir[0]
    undirected_frame.loc[i,["centrality","betweenness"]] = metrics_udir[1]
    undirected_frame.loc[i,["kNN-Accuracy","Trustworthiness","Silhouette","DBI"]] = metrics_udir[2]

    metrics_dir =   compute_metrics([get_graph_metrics,get_embedding_metrics],graph=None,emb=dir[-1],landmarks=landmarks,T=dir[0],closeness_knn=closeness_knn_directed,betweenness_knn=betweenness_knn_directed,X=X,y=y)
    directed_frame.loc[i,["centrality","betweenness"]] = metrics_dir[0]
    directed_frame.loc[i,["kNN-Accuracy","Trustworthiness","Silhouette","DBI"]] = metrics_dir[1]


    (X_subsubset,y_subsubset,landmarks_subset),udir_subset,dir_subset = get_subset(X_subset,y_subset,seed=i)
    metrics_udir =   compute_metrics([get_spectral_graph_metrics,get_graph_metrics,get_embedding_metrics],graph=None,emb=udir_subset[-1],landmarks=landmarks_subset,T=udir_subset[0],closeness_knn=closeness_knn_undirected,betweenness_knn=betweenness_knn_undirected,eigenvalues_big=eigenvalues_big_undirected,X=X_subset,y=y_subset)

    undirected_frame_2.loc[i,["dspectral","rel. eigenerr"]] = metrics_udir[0]
    undirected_frame_2.loc[i,["centrality","betweenness"]] = metrics_udir[1]
    undirected_frame_2.loc[i,["kNN-Accuracy","Trustworthiness","Silhouette","DBI"]] = metrics_udir[2]
    
    metrics_dir =   compute_metrics([get_graph_metrics,get_embedding_metrics],graph=None,emb=dir_subset[-1],landmarks=landmarks_subset,T=dir_subset[0],closeness_knn=closeness_knn_directed,betweenness_knn=betweenness_knn_directed,X=X_subset,y=y_subset)
    directed_frame_2.loc[i,["centrality","betweenness"]] = metrics_dir[0]
    directed_frame_2.loc[i,["kNN-Accuracy","Trustworthiness","Silhouette","DBI"]] = metrics_dir[1]

print('############################################')
print(data_name)
print("Level 1")
print("directed")
print(directed_frame.max())
print(directed_frame.mean())
print(directed_frame.min())

print("undirected")
print(undirected_frame.max())
print(undirected_frame.mean())
print(undirected_frame.min())

print("Level 2")
print("directed")
print(directed_frame_2.max())
print(directed_frame_2.mean())
print(directed_frame_2.min())

print("undirected")
print(undirected_frame_2.max())
print(undirected_frame_2.mean())
print(undirected_frame_2.min())

print('############################################')


# %%

print("Level 1")
print("directed")
print(dataframe_min_max_latex_rows(directed_frame))
print("undirected")
print(dataframe_min_max_latex_rows(undirected_frame))


print("Level 2")
print("directed")
print(dataframe_min_max_latex_rows(directed_frame_2))
print("undirected")
print(dataframe_min_max_latex_rows(undirected_frame_2))

