import numpy as np
import scipy as sp
import sklearn
import sklearn.manifold
from tqdm import tqdm
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import igraph as ig


def harmonic_centrality(W:sp.sparse.csr_array,k=None,seed=42,cutoff=None):
    vertices = None
    W.eliminate_zeros()
    W = W.tocoo()
    #convert probabilities to 1-log(p) to make shortest path most likely transition x to y
    distance_W = W.data
    #distance_W[distance_W==0]=1e-6
    print("converted_to distance")
    
    edges = list(zip(W.row,W.col))
    G = ig.Graph(edges,directed=True)
    G.es["weight"] = distance_W
    if k is not None:
        np.random.seed(seed)
        vertices = np.random.choice(G.vs.indices, min(k,G.vcount()),replace=False)
    print("start closeness")
    return G.harmonic_centrality(vertices=vertices,mode="in",weights=G.es["weight"],cutoff=cutoff)


def betweenness_centrality(W:sp.sparse.csr_array,k=None,seed=42,cutoff=None,directed=False):
    vertices = None
    W.eliminate_zeros()
    W = W.tocoo()
    #convert probabilities to 1-log(p) to make shortest path most likely transition x to y
    distance_W = W.data
    #distance_W[distance_W==0] = 1e-6
    print("converted_to distance")
    
    edges = list(zip(W.row,W.col))
    G = ig.Graph(edges,directed=True)
    G.es["weight"] = distance_W

    if k is not None:
        np.random.seed(seed)
        vertices = np.random.choice(G.vs.indices, min(k,G.vcount()),replace=False)
    print("start betweeness")
    return G.betweenness(vertices= vertices,weights=G.es["weight"],directed=directed,cutoff=cutoff)




def KL_Div(P:np.array,Q:np.array):
    #change distances to a distribution of distances via histogram density
    bin_size = min(len(Q),len(P))
    hist_small, bin_edges = np.histogram(P,bins=bin_size,density=True)
    hist_big, _ = np.histogram(Q,bins=bin_size,density=True)
    
    #prevent underflow
    hist_small = np.clip(hist_small,1e-10,None)
    hist_big = np.clip(hist_big,1e-10,None)
    return sp.stats.entropy(pk=hist_small,qk=hist_big)


def compute_eigenvalues(matrix,n_eigenvals):
    
    #random_guess = np.random.rand(matrix.shape[0],n_eigenvals)
    #precondition = sp.sparse.diags(1/matrix.diagonal())
    #eigenval,_ = sp.sparse.linalg.lobpcg(matrix,random_guess,M=precondition,largest=False)
    #use other if no graph laplacian
    if matrix.shape[0]<=n_eigenvals:
        eigenval = np.linalg.eigvals(matrix.toarray())
    else:
        eigenval,_ = sp.sparse.linalg.eigsh(matrix,k=n_eigenvals,which="SA")
    return eigenval


def spectral_graph_distance(eigenvaluesA,eigenvaluesB,k=None):
    min_k = min(len(eigenvaluesA),len(eigenvaluesB))
    if k is None or k>min_k:
        k = min_k
    
    diff = np.linalg.norm(np.sort(eigenvaluesA[:k])-np.sort(eigenvaluesB[:k]))
    distance = np.sum(diff)
    return distance,diff


def relative_eigenvalue_error(lambda_big,lambda_small,k=None,eps=1e-6):
    min_k = min(len(lambda_big),len(lambda_small))
    if k is None or k>min_k:
        k=min_k
    lambda_fine_k = lambda_small[:k]
    lambda_coarse_k = lambda_big[:k]
    errors = np.abs(lambda_coarse_k - lambda_fine_k) / (np.abs(lambda_fine_k) + eps)
    return np.mean(errors),errors

###embedding metrics
def knn_acc(X:np.ndarray,embedding:np.ndarray,k:int=10,X_knn=None):
    N = embedding.shape[0]
    if X_knn is None:
        X_knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k+1).fit(X)
    X_nn = X_knn.kneighbors(X,return_distance=False)[:,1:]
    embedded_knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k+1).fit(embedding)
    embedded_nn = embedded_knn.kneighbors(embedding,return_distance=False)[:,1:]
    
    overlap:int = 0
    for NN_ori,NN_embedded in zip(X_nn,embedded_nn):
        overlap += len(np.intersect1d(NN_ori,NN_embedded))
    return overlap/(k*N)


def trustworthiness(X:np.ndarray,embedding:np.ndarray,k:int=10):
    #trust = 0
    #n = embedding.shape[0]
    #X_knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k+1).fit(X)
    #X_nn = X_knn.kneighbors(X,return_distance=False)[:,1:]
    #embedded_knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k+1).fit(embedding)
    #embedded_nn = embedded_knn.kneighbors(embedding,return_distance=False)[:,1:]
    #
    #for NN_ori,NN_embedded in tqdm(zip(X_nn,embedded_nn)):
    #    notin = ~np.isin(NN_ori,NN_embedded)
    #    rank = np.flatnonzero(notin)
    #    trust += rank.sum()-len(rank)*k

    #return 1-(2*trust/(n*k*(2*n-3*k-1)))
    if embedding.shape[0]/2<k:
        k=int(embedding.shape[0]/2)
    return sklearn.manifold.trustworthiness(X,embedding,n_neighbors=k)

def Davies_bouldin_index(X:np.array,fcluster:np.array):
    return sklearn.metrics.davies_bouldin_score(X,fcluster)
    

def silhouette_score(X:np.array,fcluster:np.array):
    silh_vals= sklearn.metrics.silhouette_samples(X,fcluster)
    silh_score = silh_vals.mean()
    silh_coefficient = [np.mean(silh_vals[fcluster==cluster]) for cluster in np.unique(fcluster)]
    return silh_score,silh_coefficient,silh_vals

def silhouette_score(X:np.array,fcluster:np.array):
    silh_vals= sklearn.metrics.silhouette_samples(X,fcluster)
    silh_score = silh_vals.mean()
    silh_coefficient = [np.mean(silh_vals[fcluster==cluster]) for cluster in np.unique(fcluster)]
    return silh_score,silh_coefficient,silh_vals


def plot_silhouette(X,fcluster,data_name="",u_cluster=None):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlim([-0.1, 1])
    if u_cluster is None:
        u_cluster = np.unique(fcluster).astype(int)
    n_clusters = len(u_cluster)
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    silh_score,silh_coeffient,sample_silhouette_values = silhouette_score(X, fcluster)
    

    y_lower = 10
    for i,cluster_name in enumerate(u_cluster):
        ith_cluster_silhouette_values = sample_silhouette_values[fcluster == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.5,
        )
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        #code for average cluster
        avg_silhouette = np.mean(ith_cluster_silhouette_values)
        ax1.plot([avg_silhouette, avg_silhouette], [y_lower, y_upper], color=color, linestyle="-", alpha=1.0,label=str(cluster_name))
        ax1.hlines(y=y_lower, xmin=avg_silhouette - 0.02, xmax=avg_silhouette + 0.02, color=color, linestyle="-", linewidth=2)
        ax1.hlines(y=y_upper, xmin=avg_silhouette - 0.02, xmax=avg_silhouette + 0.02, color=color, linestyle="-", linewidth=2)

        y_lower = y_upper + 10  
    
    ax1.axvline(x=silh_score, color="red", linestyle="--",label="all")
    ax1.set_yticks([]) 
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.legend(title="Average of Cluster")
    colors = cm.nipy_spectral(fcluster.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )
    centers = np.array([np.mean(X[cluster==fcluster,:],axis=0) for cluster in u_cluster])
    
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the data.")

    plt.suptitle(
        "Silhouette analysis embedded "+ data_name,
        fontsize=14,
        fontweight="bold",
    )

    plt.show()