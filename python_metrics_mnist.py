# %%
#%load_ext autoreload
#%autoreload 2

# %%
import os
if os.getcwd() == "/gpfs01/berens/user/mchrist":
    os.chdir("./coarse_grain/")

# %%
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

# %%
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

# %%
data_name = "MNIST"
dataset_directory = f"./datasets/{data_name}"
directed=False
n=10
reduction = 0.1
seed=42
X,y,class_table = data_loader.select_dataset(kind=data_name,directory=dataset_directory,seed=seed)
n_eigenvalues = 1000
N,n_dim = X.shape
threshold=1e-4
bfs_k=10
bfs_use_distance=False
noise=0
dir_plots = join("./plots",data_name) 
dir = join("./results",data_name)
#gen_noise = helper.generate_noise(X,noise=noise,seed=seed)
#X = gen_noise
first_time_metric=True

# %%
#Colors

#colors = np.load(join(dataset_directory,"tasic-colors.npy"))

colors = plt.cm.tab10.colors
if len(colors) != len(class_table):
    raise Exception(f"please load correct number of colors\n colors:{len(colors)} categories:{len(class_table)}")
else:
    print("Loaded colors")

# %%
X_knn = sp.csr_array(sklearn.neighbors.kneighbors_graph(X,n_neighbors=n))
X_knn_undirected = X_knn.T + X_knn
X_knn_undirected.data = np.array(len(X_knn_undirected.data)*[1])

laplacian_knn_undirected = -X_knn_undirected 
laplacian_knn_undirected.setdiag(laplacian_knn_undirected.diagonal() - laplacian_knn_undirected.sum(axis=1))

#stochastic X_knn
X_knn_directed = X_knn
laplacian_knn_directed = -X_knn_directed 
laplacian_knn_directed.setdiag(laplacian_knn_directed.diagonal() - laplacian_knn_directed.sum(axis=1))

X_knn_undirected = X_knn_undirected/X_knn_undirected.sum(axis=1)
X_knn_undirected = X_knn_directed/X_knn_directed.sum(axis=1)

# %%
official_names = {"kNN-Accuracy":"kNN-acc","Trustworthiness":"trust","Silhouette score":"silh","Davies-Bouldin index":"dbi","spectral graph distance":"dspectral","Average relative eigenvalue error":"rel_eigval_e","Closeness centrality":"closeness","Betweenness centrality":"betweeness"}
metrics_order = {"kNN-acc":0,"trust":1,"silh":2,"dbi":3,"dspectral":4,"rel_eigval_e":5,"closeness":6,"betweeness":7}
sampling_order = {"Random":0,"RandomNN":1,"RW":2,"HUBS":3,"HUBSNN":4}
connection_order = {"CW":0,"CON":1,"KRON":2,"LL-BFS":3,"LL-RW":4,"SL-BFS":5,"LL-RW":6}
#[LL-BFS,LL-RW,SL-BFS,SL-RW,CON,CW,EXACT]
colors_con = {"LL-BFS":"#15A33B","LL-RW":"#25F500","SL-BFS":"#137AF0","SL-RW":"#05D5F5","CON":"#F247F5","CW":"#7A3E7A","EXACT":"#FA3300"}
level1_color,level2_color = "#25F500","#15A33B"

# %%
if first_time_metric:
    print("Started closeness")
    closeness_knn_undirected = metrics.closeness_centrality(W=X_knn_undirected)
    #closeness_knn_undirected = metrics.closeness_centrality(X_knn_undirected,k=1,cutoff=2)
    with open(join(dir,f"{data_name}_closeness.pkl"),"wb") as file:
                pickle.dump(closeness_knn_undirected,file)
    print("hooray")

    betweeness_knn_undirected = metrics.betweenness_centrality(W=X_knn_undirected)
    with open(join(dir,f"{data_name}_betweenness.pkl"),"wb") as file:
                pickle.dump(betweeness_knn_undirected,file)

    #only solve eigenvalues once as they are computational expensive
    eigenvalues_big_undirected = sp.linalg.eigsh(laplacian_knn_undirected,k=1000,which="SM",return_eigenvectors=False)
    with open(join(dir,f"{data_name}_eigenvalues_big_undirected.pkl"),"wb") as file:
                pickle.dump(eigenvalues_big_undirected,file)

# %%
# %%
closeness_knn_undirected = load_from_pickle(join(dir,f"{data_name}_closeness.pkl"))
betweeness_knn_undirected = load_from_pickle(join(dir,f"{data_name}_betweenness.pkl"))
eigenvalues_big_undirected = load_from_pickle(join(dir,f"{data_name}_eigenvalues_big_undirected.pkl"))

# %%
#eigenvalues_big_directed = load_from_pickle(join(dir,f"{data_name}_eigenvalues_big_directed.pkl"))

# %%
def get_metrics_from_graph(graph,this_level=False,eigenvalues_big=eigenvalues_big_undirected,n_eigenvalues=n_eigenvalues):
    if this_level:
        TSNE_emb = graph.TSNE
        W = graph.get_T()
        current_landmarks = graph.find_corresponding_landmarks_at_level()
    else:
        TSNE_emb = graph.bigger_graph.TSNE
        current_landmarks = graph.bigger_graph.find_corresponding_landmarks_at_level()
        W = graph.bigger_graph.get_T()
    f_cluster = y[current_landmarks]
    graph_laplacian = -W
    graph_laplacian.setdiag(graph_laplacian.diagonal() + np.ones(graph_laplacian.shape[0]))
    eigenvalues_small = metrics.compute_eigenvalues(graph_laplacian,n_eigenvalues)
    knn_acc=metrics.knn_acc(X[current_landmarks],TSNE_emb,k=10)
    trust=metrics.trustworthiness(X[current_landmarks],TSNE_emb,k=10)
    silh=metrics.silhouette_score(TSNE_emb,f_cluster)[0]
    dbi=metrics.Davies_bouldin_index(TSNE_emb,f_cluster)
    d_spectral = metrics.spectral_graph_distance(eigenvalues_big,eigenvalues_small)[0]
    rel_eigval_e = metrics.relative_eigenvalue_error(eigenvalues_big,eigenvalues_small)[0]
    graph_close_centrality = metrics.closeness_centrality(W)
    graph_between_centrality = metrics.betweenness_centrality(W)
    close = metrics.KL_Div(closeness_knn_undirected,graph_close_centrality)
    between = metrics.KL_Div(betweeness_knn_undirected,graph_between_centrality)
    return knn_acc,trust,silh,dbi,d_spectral,rel_eigval_e,close,between

# %%
def save_graph(data_name,sampling,connection,reduction,n,seed,X,y,noise=0,directed=False,level:int=2,discrete=False,directory=""):
    graph1 = OOP_Multilevel_tsne.KNNGraph(data=X,labels=y,n=n,data_name=data_name,directed=directed,weighted=True,landmark_sampling=copy.deepcopy(sampling),connection=copy.deepcopy(connection),discrete_labels=discrete,seed=seed)
    graph1.TSNE_to_attribute(random_init=False)
    if directed:
        filename = join(directory,f"d:{data_name}r:{reduction}n:{n}s:{seed}:noise{noise}_directed")
    else:
        filename = join(directory,f"d:{data_name}r:{reduction}n:{n}s:{seed}:noise{noise}")
    print("graph 1 ready")
    if level==1:
        graph1.save_all(filename)
    else:
        graph2 = graph1.create_new_level()
        graph2.TSNE_to_attribute(random_init=False)
        print("graph 2 ready")

        if level==2:
            graph2.save_all(filename)
        else:
            graph3 = graph3.create_new_level()
            graph3.TSNE_to_attribute(random_init=False)
            print("graph 2 ready")
            graph3.save_all(filename)
    return filename
# %%
cons = ["SL-RW","LL-RW","Connectivity"]
reductions = [0.5,0.4,0.3,0.2,0.1]
higher_reductions = [r**2 for r in reductions]
n_metrics = len(metrics_order)
metrics_lv1_redu = np.zeros((len(cons),len(reductions),n_metrics))
metrics_lv2_redu = np.zeros((len(cons),len(reductions),n_metrics))


con = "StateToLandmarksRandomWalks"
n_con=0
start_fname=join(dir,f"d:{data_name}r:")
end_fname=f"n:10s:42:noise0_la:HighestDegreeSampling_con:{con}_lv:2.pkl"
#graphs_redu_lv2 = load_graphs(start_fname=start_fname,end_fname=end_fname,variation=reductions)

graphs_redu_lv2 = load_graphs(start_fname=start_fname,end_fname=end_fname,variation=reductions)
end_fname=f"n:10s:42:noise0_la:HighestDegreeSampling_con:{con}_lv:1.pkl"
graphs_redu_lv1 = load_graphs(start_fname=start_fname,end_fname=end_fname,variation=higher_reductions)


for i,(graphlv2,graphlv1) in enumerate(zip(graphs_redu_lv2[1:],graphs_redu_lv1[1:])):
    print(i)
    metrics_lv2_redu[n_con,i+1,:] = get_metrics_from_graph(graphlv2,this_level=True)
    metrics_lv1_redu[n_con,i+1,:] = get_metrics_from_graph(graphlv1,this_level=True)
#save_pickle(metrics_lv2_redu,join(dir,"metrics_lv2_redu_1.pkl"))
#save_pickle(metrics_lv1_redu,join(dir,"metrics_lv1_redu_1.pkl"))


con = "LandmarksToLandmarksRandomWalks"
n_con=1
start_fname=join(dir,f"d:{data_name}r:")
end_fname=f"n:10s:42:noise0_la:HighestDegreeSampling_con:{con}_lv:2.pkl"
graphs_redu_lv2 = load_graphs(start_fname=start_fname,end_fname=end_fname,variation=reductions)
end_fname=f"n:10s:42:noise0_la:HighestDegreeSampling_con:{con}_lv:1.pkl"
graphs_redu_lv1 = load_graphs(start_fname=start_fname,end_fname=end_fname,variation=higher_reductions)

for i,(graphlv2,graphlv1) in enumerate(zip(graphs_redu_lv2,graphs_redu_lv1)):
    metrics_lv2_redu[n_con,i,:] = get_metrics_from_graph(graphlv2,this_level=True)
    metrics_lv1_redu[n_con,i,:] = get_metrics_from_graph(graphlv1,this_level=True)
#save_pickle(metrics_lv2_redu,join(dir,"metrics_lv2_redu_2.pkl"))
#save_pickle(metrics_lv1_redu,join(dir,"metrics_lv1_redu_2.pkl"))

con = "Connectivity"
n_con=2
start_fname=join(dir,f"d:{data_name}r:")
end_fname=f"n:10s:42:noise0_la:HighestDegreeSampling_con:{con}_lv:2.pkl"
#graphs_redu_lv2 = load_graphs(start_fname=start_fname,end_fname=end_fname,variation=reductions)
graphs_redu_lv2 = load_graphs(start_fname=start_fname,end_fname=end_fname,variation=reductions)
end_fname=f"n:10s:42:noise0_la:HighestDegreeSampling_con:{con}_lv:1.pkl"
graphs_redu_lv1 = load_graphs(start_fname=start_fname,end_fname=end_fname,variation=higher_reductions)

for i,(graphlv2,graphlv1) in enumerate(zip(graphs_redu_lv2,graphs_redu_lv1)):
    metrics_lv2_redu[n_con,i,:] = get_metrics_from_graph(graphlv2,this_level=True)
    metrics_lv1_redu[n_con,i,:] = get_metrics_from_graph(graphlv1,this_level=True)

save_pickle(metrics_lv2_redu,join(dir,"metrics_lv2_redu.pkl"))
save_pickle(metrics_lv1_redu,join(dir,"metrics_lv1_redu.pkl"))

noises = [0,0.1,0.2,0.4,0.5,0.6,0.8]

connections = ["LandmarksToLandmarksBFS","LandmarksToLandmarksRandomWalks","StateToLandmarksBFS","StateToLandmarksRandomWalks","Connectivity"]
scores = np.zeros(shape=(len(connections),len(metrics_order),len(noises)))
con = ["LL-BFS","LL-RW","SL-BFS","SL-RW","CON"]
c_colors = [colors_con[connect] for connect in con]

for i,connect in enumerate(connections):
    for j,noise in enumerate(noises):
        fname = join(dir,f"d:{data_name}r:0.1n:10s:42:noise{noise}_la:HighestDegreeSampling_con:{connect}_lv:2.pkl")
        graph = load_coarsen_graph(fname)
        score = get_metrics_from_graph(graph)
        scores[i,:,j] = score
save_pickle(scores,join(dir,"scores_over_noise.pkl"))

exact_graph = load_coarsen_graph(join(dir,f"d:{data_name}r:0.1n:10s:42:noise0_la:HighestDegreeSampling_con:StateToLandmarksExact_lv:2.pkl"))
if first_time_model:
    metrics_exact = get_metrics_from_graph(exact_graph)
    save_pickle(metrics_exact,join(dir,f"{data_name}_metric_exact.pkl"))

print("Ended metrics")
