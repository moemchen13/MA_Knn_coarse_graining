import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from matplotlib.patches import Ellipse
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import igraph as ig
from scipy import sparse
from tqdm import tqdm
from sklearn.preprocessing import normalize
import openTSNE

def to_transition_matrix(adj):
    #T = np.divide(adj,adj.sum(axis=1,keepdims=True),out=np.zeros_like(adj),where=adj.sum(axis=1,keepdims=True)!=0)
    T = normalize(adj,norm="l1",axis=1)
    return T

def add_salt_and_pepper_noise_vectorized(flattened_images, noise_fraction=0.1):
    flattened_images = np.array(flattened_images, dtype=np.float32)
    num_images, num_pixels = flattened_images.shape
    total_noise = int(num_pixels * noise_fraction)
    
    rng = np.random.default_rng()
    noise_mask = rng.random(size=(num_images, num_pixels))  # Uniform [0, 1)
    
    salt_threshold = noise_fraction / 2
    pepper_threshold = noise_fraction / 2
    
    noisy_images = flattened_images.copy()
    noisy_images[noise_mask < pepper_threshold] = 0
    noisy_images[noise_mask > (1 - salt_threshold)] = 255
    
    noisy_images = np.clip(noisy_images, 0, 255)
    
    return noisy_images


def generate_noise(X:np.ndarray,noise:float=0.1,X_test:np.ndarray=None,seed=42,normalized_dim:bool=True,data_name=None):
    np.random.seed(seed)
    if data_name is not None:
        if data_name == "MNIST" or data_name == "DIGITS":
            return add_salt_and_pepper_noise_vectorized(X,noise)

    N,n_dim = X.shape
    diff_per_dim = np.max(X,axis=0) - np.min(X,axis=0)
    generated_noise = np.random.randn(N,n_dim)*noise
    if normalized_dim:
        generated_noise = generated_noise*diff_per_dim/40

    if X_test is not None:
        generated_noise_test = np.random.rand(n_dim,X_test.shape[0])*diff_per_dim/40 *noise
        return generated_noise,generated_noise_test
    return X + generated_noise




def isin(state,landmarks):
    #landmarks have to be sorted if index is in list in log(n)
    index = np.searchsorted(landmarks,state,side="left")
    if index>=len(landmarks):
        return -1,False
    return index,state==landmarks[index]


def swissroll(n_circles=1,N=1000,noise=0.05,seed=None):
    #returns tuple with swissrol data and t 
    if seed is not None:
        np.random.seed(seed)
    t = n_circles*2*np.pi/2 * (1 + 2*np.random.rand(1,N))
    h = 21 * np.random.rand(1,N)
    data = np.concatenate((t*np.cos(t),h,t*np.sin(t))) + noise*np.random.randn(3,N)	
    return np.transpose(data), np.squeeze(t)

def create_T_matrices(graph,weighted=False):
    #sanity check to avoid errors
    if weighted:
        weighted = graph.is_weighted()

    N = graph.vcount()
    most_neighbors = 0
    for vertex in range(N):
        n_neighbors = len(graph.neighbors(vertex,mode="out"))
        if n_neighbors>most_neighbors:
            most_neighbors=n_neighbors
            
    T_states = np.ones((N,most_neighbors),dtype=np.int64)*-1
    T_probs = np.zeros((N,most_neighbors))

    for vertex in tqdm(range(N)):
        neighbors = graph.neighbors(vertex,mode="out")
        n_neighbors = len(neighbors)
        for i,neighbor in enumerate(neighbors):
            #using negative indices to not get negative states because rounding errors
            T_states[vertex,-(i+1)] = neighbor
            weight = 1/n_neighbors
            if weighted:
                edge_id = graph.get_eid(vertex,neighbor)
                weight = graph.es[edge_id]['weight']
            T_probs[vertex,-(i+1)] = weight
    T_probs_cumsum = T_probs.cumsum(axis=1)
    T_probs_cumsum[:,-1] = 1
    return T_probs_cumsum,T_states


def create_graph_from_2Darray(W:sparse.csr_array,directed=True,weighted=False):
    N = W.shape[0]
    if directed:
        W = W.tocoo()
        edges = [(s, t) for s, t in zip(*W.tocoo().coords)]
        graph = ig.Graph(N,edges=edges, directed=True)
        if weighted:
            graph.es['weight'] = W.data

    else: #undirected
        W = W.maximum(W.T)
        W = W.tocoo()
        edges = [(s,t) for s,t in zip(W.coords[0], W.coords[1]) if s<t] #if because we have them already included
        graph = ig.Graph(N,edges=edges, directed=False)
        if weighted:
            weights = [edge_weight for edge_weight,(s,t) in zip(W.data,zip(*W.coords)) if s<t]
            graph.es['weight'] = weights
    return graph


def get_original_landmarks(new_landmarks_indices,landmarks_ascending):
    original_landmarks = new_landmarks_indices
    for landmarks in landmarks_ascending[::-1]:
        original_landmarks = landmarks[original_landmarks]
    return original_landmarks


def plot_landmarks_labels(landmark_labels,labels,title0="landmarks per label total",title1="Fractions of landmarks per label"):
    fig,axs = plt.subplots(1,2,figsize=(10,5))
    bins = np.arange(len(labels)+1)
    counts, bin_edges = np.histogram(landmark_labels, bins=bins)
    colors = plt.cm.tab10(np.linspace(0, 1, len(counts)))
    axs[0].bar(bin_edges[:-1], counts, width=np.diff(bin_edges), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(labels)], edgecolor='black', align='center')
    axs[0].set_xlabel('Classes')
    axs[0].set_ylabel('Counts')
    axs[0].set_xticks(labels)
    axs[0].set_title(title0)
    axs[1].pie(counts,labels=labels,counterclock=False,startangle=90)
    axs[1].set_title(title1)
    plt.show()


#Handy operations for clusters
def cluster_to_fcluster(clusters):
    n_elements = 0
    for com in clusters:
        n_elements += len(com)
    fcluster = np.empty(shape=(n_elements,),dtype=int)
    for i,com in enumerate(clusters):
        fcluster[com] = i
    return fcluster


def fcluster_to_cluster(fclusters):
    if type(fclusters)==list:
        fclusters = np.array(fclusters)
    n_clusters = np.sort(np.unique(fclusters))[-1] +1
    clusters = [[] for _ in range(n_clusters)]
    for i in range(n_clusters):
        indices =  list(np.where(fclusters==i)[0])
        indices = [int(index) for index in indices]
        clusters[i] = indices
    return clusters


#plotting of graphs
def plot_graph2D_static(points,some_graph):

    edges = some_graph.es

    for i,edge in enumerate(edges):
        source = edge.source
        target = edge.target
        x_coords = [points[source, 0], points[target, 0]]
        y_coords = [points[source, 1], points[target, 1]]
        point_color = "gray"
        alpha_value = 0.2
        plt.plot(x_coords, y_coords, c=point_color, alpha=alpha_value)
    
    plt.scatter(
        points[:, 0], points[:, 1], s=5, alpha=0.5
    )
    plt.title("Visualisation of the igraph")

    plt.tight_layout()
    plt.show()


def plot_graph2D_static_above_prob(points,transition_matrix,prob_cutoff=0.05,landmarks=None):

    sym_matrix = np.maximum(transition_matrix,transition_matrix.T)
    #if not symmetric now highest connection in
    sources, targets = sym_matrix.nonzero()
    g = ig.Graph(edges=list(zip(sources, targets)), directed=False)
    edges = g.es
    for i,edge in enumerate(edges):
        
        source = edge.source
        target = edge.target
        x_coords = [points[source, 0], points[target, 0]]
        y_coords = [points[source, 1], points[target, 1]]
        point_color = "gray"
        alpha_value = 0.5#sym_matrix[source,target]
        if sym_matrix[source,target]>prob_cutoff:
            plt.plot(x_coords, y_coords, c=point_color, alpha=alpha_value)
    
    plt.scatter(
        points[:, 0], points[:, 1], s=5, alpha=0.5
    )
    if landmarks is not None:
        plt.scatter(
        points[landmarks, 0], points[landmarks, 1], s=5, alpha=1,c="red"
    )
    plt.title("Visualisation of the Transition matrix")
    plt.tight_layout()
    plt.show()


def plot_subgraph2D_static(points,some_graph,indeces_vertices):

    plt.scatter(
        points[:, 0], points[:, 1], s=5, alpha=0.5
    )

    edges = some_graph.es
    for i,edge in enumerate(edges):
        #translate to ori vertex
        source = indeces_vertices[edge.source]
        target = indeces_vertices[edge.target]
        x_coords = [points[source, 0], points[target, 0]]
        y_coords = [points[source, 1], points[target, 1]]
        point_color = "gray"
        alpha_value = 0.2
        plt.plot(x_coords, y_coords, c=point_color, alpha=alpha_value)
    
    plt.scatter(
        points[indeces_vertices, 0], points[indeces_vertices, 1],c="red", s=10, alpha=1
    )
    plt.title("Visualisation of the igraph")
    plt.tight_layout()
    plt.show()


def plot_cov_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    eigvals, eigvecs = np.linalg.eigh(cov)
    width, height = 2 * n_std * np.sqrt(eigvals)
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)


def plot_gaussian_samples(mean_clusters,covariance_clusters,all_samples):
    fig,ax=plt.subplots(figsize=(8,8))
    for i in range(mean_clusters.shape[0]):
        plot_cov_ellipse(ax, mean_clusters[i], covariance_clusters[i], n_std=2,facecolor='green', edgecolor='green', alpha=0.2)
        plt.text(mean_clusters[i,0],mean_clusters[i,1],s=i,c="green")

    ax.scatter(mean_clusters[:,0],mean_clusters[:,1],marker="x",s=3,c="red",label="Means")
    ax.scatter(all_samples[:,0],all_samples[:,1],marker=".",s=1)
    ax.legend()
    ax.set_title("Toy Data")
    plt.show()

#3D
def plot_graph3D_static_with_clusters(points,graph,clusters,show_only_clusters=True):

    fcluster = cluster_to_fcluster(clusters)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))
    cluster_colors = [colors[com] for com in fcluster]


    edges = graph.es
    angles = [(-66, 12), (30, 30), (150, 10), (210, 20)]
    fig = plt.figure(figsize=(8*len(angles), 6*len(angles)))

    for i, angle in enumerate(angles):
        ax = fig.add_subplot(1, len(angles), i + 1, projection="3d")
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2], c=cluster_colors, s=10, alpha=0.8
        )

        for edge in edges:
            source = edge.source
            target = edge.target
            x_coords = [points[source, 0], points[target, 0]]
            y_coords = [points[source, 1], points[target, 1]]
            z_coords = [points[source, 2], points[target, 2]]
            
            same_cluster = fcluster[target] == fcluster[source]
            if same_cluster:
                point_color = cluster_colors[target]
                alpha_value = 0.5
            else:
                point_color = "gray"
                if show_only_clusters:
                    alpha_value=0
                else:
                    alpha_value = 0.5
            
            ax.plot(
                x_coords, y_coords, z_coords, c=point_color, alpha=alpha_value
            )

        # Set title and viewpoint
        ax.set_title(f"View angle: azim={angle[0]}, elev={angle[1]}")
        ax.view_init(azim=angle[0], elev=angle[1])
        ax.text2D(0.8, 0.05, s="n_samples=1000", transform=ax.transAxes)
    #fig.suptitle(f"KNN walktrap communities n={len(clusters)} from different angles", fontsize=16, y=0.95)  # Adjust 'y' as needed
    #plt.subplots_adjust(hspace=0.4, top=0.85)  # Control spacing between subplots and figure title
    

    plt.tight_layout()
    plt.show()


def plot_graph3D(points, graph, point_color, title="Knn graph",show_edges=False):
    edges = graph.es

    if show_edges:
        all_edge_traces = []
        for edge in edges:
            source = edge.source
            target = edge.target
            x_coords = [points[source, 0], points[target, 0]]
            y_coords = [points[source, 1], points[target, 1]]
            z_coords = [points[source, 2], points[target, 2]]
            
            edge_color = "gray"

            all_edge_traces.append(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='lines',
                line=dict(color=edge_color, width=1),
                showlegend=False
            ))

    points_trace = go.Scatter3d(
        x=points[:, 0], 
        y=points[:, 1], 
        z=points[:, 2],
        mode='markers',
        marker=dict(
            color=point_color,
            colorscale='Viridis',
            colorbar=dict(title="Color Scale"),
            showscale=False,
            opacity=1,
            size=5
        ),
    )

    if show_edges:
        fig = go.Figure(data=all_edge_traces)
        fig.add_trace(points_trace)
    else:
        fig = go.Figure(data=points_trace)

    fig.update_layout(title=title, autosize=False, width=800, height=600,showlegend=False)
    fig.show()


def plot_graph3D_multiple_views(points,graph,colors,angles=[(-66, 12), (30, 30), (150, 10), (210, 20)]):

    edges = graph.es
    fig = plt.figure(figsize=(8*len(angles), 6*len(angles)))

    for i, angle in enumerate(angles):
        ax = fig.add_subplot(1, len(angles), i + 1, projection="3d")
        if colors is None:
            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2], c="red", s=10, alpha=0.8
                
            )
        else: 
            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2], c=colors, s=10, alpha=0.8
            )

        for edge in edges:
            source = edge.source
            target = edge.target
            x_coords = [points[source, 0], points[target, 0]]
            y_coords = [points[source, 1], points[target, 1]]
            z_coords = [points[source, 2], points[target, 2]]
            
            alpha_value = 0.5
            point_color="grey"
            
            ax.plot(
                x_coords, y_coords, z_coords, c=point_color, alpha=alpha_value
            )

        # Set title and viewpoint
        ax.set_title(f"View angle: azim={angle[0]}, elev={angle[1]}")
        ax.view_init(azim=angle[0], elev=angle[1])
        ax.text2D(0.8, 0.05, s="n_samples=1000", transform=ax.transAxes)
    
    plt.tight_layout()
    plt.show()


#sampling from gaussian
def sample_from_multivariate_gaussians(mean_clusters,covariance_clusters,n_samples):
    if type(n_samples)==int:
        n_samples = np.ones((mean_clusters.shape[0]))*n_samples
    all_samples = np.empty(shape=(n_samples.sum(),2))
    index = 0
    for i in range(mean_clusters.shape[0]):
        samples = np.random.multivariate_normal(mean=mean_clusters[i,:],cov=covariance_clusters[i,:,:],size=n_samples[i])
        all_samples[index:index+n_samples[i]]=samples
        index += n_samples[i]
    return all_samples


def create_TSNE_embedding(P,X=None):
    affinities = P
    affinities = normalize(affinities,"l1",axis=1)
    P = affinities + affinities.T
    P /= 2*affinities.shape[0]
    custom_affinity = openTSNE.affinity.PrecomputedAffinities(P)
    if X is None:
        init = openTSNE.initialization.random(n_samples=P.shape[0],n_components=2,random_state=42)
    else:
        init = openTSNE.initialization.pca(X)
    embedding = openTSNE.TSNEEmbedding(embedding=init,affinities=custom_affinity)
    embedding.optimize(n_iter=250,exaggeration=12)
    embedded_points = embedding.optimize(n_iter=500,exaggeration=1)

    return embedded_points



def generate_latex_table(df):
    latex_str = '\\begin{tabular}{|' + 'c|' * len(df.columns) + '}\n'  # Table header
    latex_str += '\\hline\n&'

    # Add column names
    latex_str += ' & '.join(df.columns) + ' \\\\\n'
    latex_str += '\\hline\n'

    # Iterate over the rows
    for idx, row in df.iterrows():
        row_str = []
        for col in df.columns:
            value = row[col]
            col_min = df[col].min()
            col_max = df[col].max()

            # Mark min and max values
            if value == col_max:
                row_str.append(f'\\textbf{{{value}}}')
            elif value == col_min:
                row_str.append(f'\\emph{{{value}}}')
            else:
                row_str.append(str(value))

        latex_str += idx + ' & ' + ' & '.join(row_str) + ' \\\\\n'
        latex_str += '\\hline\n'

    latex_str += '\\end{tabular}\n'
    return latex_str