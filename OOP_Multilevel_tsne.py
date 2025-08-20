import numpy as np
from scipy import sparse as sp
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder,normalize
from typing import Union,Tuple
import OOP_Sampling
import OOP_Connecting
from openTSNE import callbacks
import openTSNE
from igraph import Graph
from tqdm import tqdm
import matplotlib.pyplot as plt
from os.path import join
import pickle
import helper


NumericArrays = Union[np.ndarray,sp.sparray]
ArrayLike = Union[np.ndarray,sp.sparray,pd.DataFrame]


class BaseGraph:
    landmark_sampling: OOP_Sampling.BaseSampling
    connection: OOP_Connecting.BaseConnecting
    level: int
    graph: sp.sparray
    weighting:bool
    seed:int
    directed:bool
    weighted:bool
    data_name:str
    storage_dir:str

    def __init__(self):
        pass

    def save_all(self,filename:str,verbose:bool=True):
        filename = f"{filename}_la:{self.landmark_sampling.__class__.__name__}_con:{self.connection.__class__.__name__}_lv:{self.level}"
        if self.storage_dir is not None and self.storage_dir != "":
            filename = join(self.storage_dir,filename)
        with open(filename+".pkl","wb") as file:
            pickle.dump(self,file)


class Coarsened_Graph(BaseGraph):
    def __init__(self,bigger_graph:BaseGraph):
        self.bigger_graph = bigger_graph
        self.landmark_sampling:OOP_Sampling.BaseSampling = bigger_graph.landmark_sampling.new_sampling()
        self.connection:OOP_Connecting.BaseConnecting = bigger_graph.connection.new_connecting()
        self.level = bigger_graph.level +1
        self.directed = self.bigger_graph.directed
        self.weighted = self.bigger_graph.weighted
        self.data_name = self.bigger_graph.data_name
        self.seed = self.bigger_graph.seed
        self.storage_dir=self.bigger_graph.storage_dir


    def __str__(self):
        return f"Coarsened-graph at level {self.level}"
    
    def get_T(self)->sp.sparray:
        return self.connection.get_T()
    
    def get_adj(self)->sp.sparray:
        if hasattr(self.connection,"get_adj"):
            return self.connection.get_adj()
        return None
    
    def get_landmarks(self)->np.ndarray:
        return self.landmark_sampling.get_landmarks()


    def sampling_method(self)->None:
        self.landmark_sampling.sampling(self.bigger_graph.connection.get_T())

    def create_T_cumsum(self)->Tuple[np.ndarray,np.ndarray]:
        #graph = self.create_graph(self.bigger_graph.get_T(),self.directed,self.weighted)
        return self.create_T_cumsum_from_matrix(self.bigger_graph.get_T())
    
    def delete(self)->None:
        self.connection.delete()

    def delete_all(self)->None:
        self.delete()
        self.bigger_graph.delete_all()
    
    def get_I(self)->np.ndarray:
        return self.connection.get_I()


    @staticmethod
    def create_T_cumsum_from_graph(graph:Graph,weighted:bool):
    #sanity check to avoid errors
        N = graph.vcount()
        adj = sp.csr_array(graph.get_adjacency_sparse())
        n_neighbors = (adj!=0).sum(axis=1)
        most_neighbors = int(np.max((n_neighbors)))
                
        T_states = np.ones((N,most_neighbors),dtype=np.int64)*-most_neighbors
        T_probs = np.zeros((N,most_neighbors))

        #for vertex in tqdm(range(N)):
        for vertex in range(N):
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
    
    
    def create_T_cumsum_from_matrix(self,T:sp.sparray)->Tuple[np.ndarray,np.ndarray]:
        N = T.shape[0]
        T.setdiag(0)
        T.eliminate_zeros()
        T = normalize(T,"l1",axis=1).tolil()
        most_neighbors = max((T!=0).sum(axis=1))

        T_states = np.ones((N,most_neighbors),dtype=np.int64)*-most_neighbors
        T_cumsum = np.zeros((N,most_neighbors))
        for i,(row,data_row) in enumerate(zip(T.rows,T.data)):
            n = len(row)
            T_states[i,0:n] = row
            T_cumsum[i,0:n] = data_row

        T_cumsum = T_cumsum.cumsum(axis=1)
        return T_cumsum, T_states


    @staticmethod
    def create_graph(W:sp.sparray,directed:bool,weighted:bool):
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
    

    def connection_method(self):

        if isinstance(self.connection,OOP_Connecting.LandmarksToLandmarksBFS) or isinstance(self.connection,OOP_Connecting.StateToLandmarksBFS):
            graph = self.create_graph(self.bigger_graph.get_T(),self.directed,self.weighted)
            self.connection.connect_landmarks(graph,self.landmark_sampling)
        
        elif isinstance(self.connection,OOP_Connecting.StateToLandmarksRandomWalks) or isinstance(self.connection,OOP_Connecting.LandmarksToLandmarksRandomWalks):
            T_probs_cumsum,T_states = self.create_T_cumsum()
            self.connection.connect_landmarks(T_probs_cumsum,T_states,self.landmark_sampling)

        elif isinstance(self.connection,OOP_Connecting.StateToLandmarksExact):
            self.connection.connect_landmarks(self.bigger_graph.connection.get_T(),self.landmark_sampling)
        
        elif isinstance(self.connection,OOP_Connecting.Connectivity):
            graph = self.create_graph(self.bigger_graph.get_adj(),self.directed,True)
            self.connection.connect_landmarks(graph,self.get_landmarks())
            if self.connection.new_landmarks is not None:
                self.landmark_sampling.add_landmark(self.connection.new_landmarks)

        elif isinstance(self.connection,OOP_Connecting.ClusterRandomWalks):
            graph = self.create_graph(self.bigger_graph.get_T(),self.directed,self.weighted)
            T_probs_cumsum,T_states = self.create_T_cumsum()
            self.connection.connect_landmarks(graph,T_probs_cumsum,T_states,self.get_landmarks())
            if self.connection.new_landmarks is not None:
                self.landmark_sampling.add_landmark(self.connection.new_landmarks)

        else:
            raise Exception("Unrecognized Connection Method")

    def coarse_grain(self):
        self.sampling_method()
        self.connection_method()


    def create_new_level(self):
        new_level_graph = Coarsened_Graph(self)
        new_level_graph.coarse_grain()
        return new_level_graph
    
    def create_new_levels(self,n_levels:int):
        if n_levels == 0:
            return self
        else:
            new_level_graph = self.create_new_level()
            return new_level_graph.create_new_levels(n_levels-1)
        
    def return_graph_of_level(self,i:int=1):
        if i>self.level or i<1:
            raise Exception(f"The {i} level doesn't exist")
        if self.level==i:
            return self
        else:
            return self.bigger_graph.return_graph_of_level(i)
    
    def return_all_T(self)->list:
        if self.level==1:
            return [self.get_T()]
        else:
            all_Ts = self.bigger_graph.return_all_T()
            all_Ts.append(self.get.T())
            return all_Ts


    def return_all_landmarks(self)->list:
        if self.level==1:
            return [self.get_landmarks()]
        else:
            all_landmarks = self.bigger_graph.return_all_landmarks()
            all_landmarks.append(self.get_landmarks())
            return all_landmarks
        
    def return_all_embeddings(self)->list:
        if self.level==1:
            return [self.create_TSNE_embedding()] #care this will be a tuple of the embedded points and the embeddingobject
        else:
            all_embeddings = self.bigger_graph.return_all_embeddings()
            all_embeddings.append(self.create_TSNE_embedding())
            return all_embeddings
    

    def get_original_labels_of_landmarks(self,i_landmarks=None)->np.ndarray:
        og_indices = self.find_corresponding_landmarks_at_level(i_landmarks)
        return self.bigger_graph.return_labels(og_indices)

    
    def find_corresponding_landmarks_at_level(self,i_landmarks=None,level=1)->np.ndarray:
        '''
        Retrieves the index of landmarks at a certain level as well as the original labeling of the those ladnmarks
        '''
        if i_landmarks is None:
            i_landmarks = self.get_landmarks()
        else:
            i_landmarks = self.get_landmarks()[i_landmarks]

        if self.level == level:
            return i_landmarks
        
        return self.bigger_graph.find_corresponding_landmarks_at_level(i_landmarks=i_landmarks,level=level)


    def return_labels(self,indices):
        return self.bigger_graph.return_labels(indices)

    def is_discrete(self):
        return self.bigger_graph.is_discrete()


    def plot_embedding(self,cmap="viridis",file=None)->None:

        og_labels,encoded_labels = self.get_original_labels_of_landmarks()
        embedding, _ = self.create_TSNE_embedding()
        
        if self.is_discrete():
            if og_labels is None: #no_labels
                plt.scatter(embedding[:,0],embedding[:,1],s=5,alpha=0.5)
            else:
                for encoded_label,og_label in zip(np.unique(encoded_labels),np.unique(og_labels)):#discrete
                    plt.scatter(embedding[encoded_label==encoded_labels,0],embedding[encoded_label==encoded_labels,1],label=og_label,s=5,alpha=0.5,edgecolors=None)#TODO fix this function
                    plt.legend(loc="upper right",title="Classes")
        else:#continous
            plt.scatter(embedding[:,0],embedding[:,1],c=encoded_labels,cmap=cmap,s=5,alpha=0.5)


        plt.title(f"Embedding of {self.data_name} level:{self.level}, connections={self.connection.__class__.__name__}, landmarks={self.landmark_sampling.__class__.__name__}")
        plt.axis("off")
        if file is None:
            plt.show()
        else:
            plt.savefig(file)


    def plot_embedding_loss(self,cmap="viridis",file=None)->None:

        og_labels,encoded_labels = self.get_original_labels_of_landmarks()
        embedding, _,loss = self.create_TSNE_embedding(return_costs=True)
        fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
        
        if self.is_discrete():
            if og_labels is None: #no_labels
                ax[0].scatter(embedding[:,0],embedding[:,1],s=5,alpha=0.5)
            else:
                for encoded_label,og_label in zip(np.unique(encoded_labels),np.unique(og_labels)):#discrete
                    ax[0].scatter(embedding[encoded_label==encoded_labels,0],embedding[encoded_label==encoded_labels,1],label=og_label,s=5,alpha=0.5,edgecolors=None)#TODO fix this function
                    ax[0].legend(loc="upper right",title="Classes")
        else:#continous
            ax[0].scatter(embedding[:,0],embedding[:,1],c=encoded_labels,cmap=cmap,s=5,alpha=0.5)


        ax[0].set_title(f"Embedding")
        ax[0].axis("off")

        ax[1].plot(np.arange(len(loss))*10,loss)
        ax[1].axvline(250,ymin=min(loss)-1,ymax=max(loss)+1,c="red",ls="--",label="early exaggeration")
        ax[1].set_ylabel("KL-Divergence")
        ax[1].set_xlabel("Iterations")
        ax[1].legend(loc="upper right")
        ax[1].set_title("Loss of Embedding")
        if file is None:
            plt.show()
        else:
            plt.savefig(file)


    def return_embedding(self):
        og_labels,encoded_labels = self.get_original_labels_of_landmarks()
        if og_labels is None:
            print("returns og_labels as None as either not discrete or unknown")
        embedding,_ = self.create_TSNE_embedding()
        return og_labels,encoded_labels,embedding


    def create_TSNE_embedding(self,symmetrize:bool=True,random_init=False,return_costs=False,callbacks_iter=10)->NumericArrays:
        transition_matrix = self.connection.get_T()
        transition_matrix.setdiag(0)
        transition_matrix = normalize(transition_matrix,"l1",axis=1)
        og_indices = self.find_corresponding_landmarks_at_level()
        og_landmarks = self.get_datapoints(og_indices)

        P = transition_matrix
        if symmetrize:
            P = transition_matrix + transition_matrix.T
            P /= 2*transition_matrix.shape[0]

        custom_affinity = openTSNE.affinity.PrecomputedAffinities(P)
        if random_init:
            init = openTSNE.initialization.random(n_samples=P.shape[0],n_components=2,random_state=self.seed)
        else:
            init = openTSNE.initialization.pca(og_landmarks)

        if return_costs:
            class KLError(callbacks.Callback):
                def __init__(self,kl_div:list=[])->None:
                    self.kl_div = kl_div

                def __call__(self,iteration: int, error:float,embedding:openTSNE.TSNEEmbedding)->None:
                    self.kl_div.append(error)

            KL_Call = KLError()
            embedding = openTSNE.TSNEEmbedding(embedding=init,affinities=custom_affinity,callbacks=KL_Call,callbacks_every_iters=callbacks_iter)
        else:
            embedding = openTSNE.TSNEEmbedding(embedding=init,affinities=custom_affinity)
        
        embedding.optimize(n_iter=250,exaggeration=12)
        embedded_points = embedding.optimize(n_iter=500,exaggeration=1)

        if return_costs:
            costs = KL_Call.kl_div
            return embedded_points,embedding,costs

        return embedded_points, embedding

    def save_TSNE_embedding(self,filename:str,random_init:bool=False):
        filename = f"{filename}_la:{self.landmark_sampling.__class__.__name__}_con:{self.connection.__class__.__name__}_lv:{self.level}"
        embedded_points,embedding = self.create_TSNE_embedding(random_init=random_init)
        np.save(f"TSNE_lv:{self.level}_samp:{self.landmark_sampling}_con:{self.connection}_{filename}",embedded_points)

    def TSNE_to_attribute(self,random_init:bool):
        embedded_points,embedding = self.create_TSNE_embedding(random_init=random_init)
        self.TSNE = embedded_points

    def get_datapoints(self,indices)->np.ndarray:
        return self.bigger_graph.get_datapoints(indices)

    def switch_weighting(self):
        self.weighting = not self.weighting

    def switch_directed(self):
        self.directed = not self.directed


class KNNGraph(Coarsened_Graph):
    def __init__(self,
                data:NumericArrays,landmark_sampling:OOP_Sampling.BaseSampling,
                connection:OOP_Connecting.BaseConnecting,labels:ArrayLike=None,n:int=10,name:str="Root graph",
                data_name:str="unnamed",discrete_labels=True,weighted=True,directed=True,seed=42,storage_dir="",run_connect=True):
        self.data = data
        self.seed = seed
        self.discrete_labels = discrete_labels

        if not self.discrete_labels or labels is None: #continous or no labels
            self.label_encoder, self.labels = None,self._prepare_labels(labels,self.data)
        else:
            self.label_encoder, self.labels = self._encode_labels(labels)
            
        self.n_neighbors = n
        self.name = name
        self.data_name = data_name
        self.landmark_sampling = landmark_sampling
        self.connection = connection
        self.level=1
        self.weighted = weighted #generated weighted graphs
        self.directed = directed 
        self.storage_dir = storage_dir
        self.knn_graph = sp.csr_array(kneighbors_graph(self.data,n_neighbors=self.n_neighbors))
        if not self.directed:
            self.knn_graph = self.knn_graph.T +self.knn_graph
            self.knn_graph.data = np.ones_like(self.knn_graph.data)
        self.knn_graph = helper.to_transition_matrix(self.knn_graph)
        
        self.sampling_method()
        if run_connect:
            self.connection_method()

    def sampling_method(self):
        self.landmark_sampling.sampling(self.knn_graph)


    def delete_all(self)->None:
        self.delete()


    def create_T_cumsum(self)->Tuple[np.ndarray,np.ndarray]:
        #innitialised from knn grpah therefore unweighted
        #graph = self.create_graph(self.knn_graph,self.directed,weighted = False)
        return self.create_T_cumsum_from_matrix(self.knn_graph)


    def is_discrete(self)->bool:
        return self.discrete_labels


    def connection_method(self)->None:
        if isinstance(self.connection,OOP_Connecting.LandmarksToLandmarksBFS) or isinstance(self.connection,OOP_Connecting.StateToLandmarksBFS):
            graph = self.create_graph(self.knn_graph,weighted=False,directed=self.directed)#T correct
            self.connection.connect_landmarks(graph,self.landmark_sampling)
        
        elif isinstance(self.connection,OOP_Connecting.StateToLandmarksRandomWalks) or isinstance(self.connection,OOP_Connecting.LandmarksToLandmarksRandomWalks):
            T_probs_cumsum,T_states = self.create_T_cumsum()
            self.connection.connect_landmarks(T_probs_cumsum,T_states,self.landmark_sampling)

        elif isinstance(self.connection,OOP_Connecting.StateToLandmarksExact):
            T = sp.csr_array(self.create_graph(self.knn_graph,directed=self.directed,weighted=False).get_adjacency_sparse())
            T = normalize(T,"l1",axis=1)
            self.connection.connect_landmarks(T,self.landmark_sampling)
        

        elif isinstance(self.connection,OOP_Connecting.Connectivity):
            graph = self.create_graph(self.knn_graph,self.directed,True)
            self.connection.connect_landmarks(graph,self.get_landmarks())
            if self.connection.new_landmarks is not None:
                self.landmark_sampling.add_landmark(self.connection.new_landmarks)

        elif isinstance(self.connection,OOP_Connecting.ClusterRandomWalks):
            graph = self.create_graph(self.knn_graph,self.directed,True)
            T_probs_cumsum,T_states = self.create_T_cumsum()
            self.connection.connect_landmarks(graph,T_probs_cumsum,T_states,self.get_landmarks())
            if self.connection.new_landmarks is not None:
                self.landmark_sampling.add_landmark(self.connection.new_landmarks)

        else:
            raise Exception("Unrecognized Connection Method")


    def __str__(self)->str:
        return f"{self.name} with {self.data_name} data of size {self.data.shape} with labels {np.unique()} \\ You are at the root\\Choosen method for coarse-graining sampling:{self.landmark_sampling.name} connection{self.connection.name}"
    

    def get_original_labels_of_landmarks(self,i_landmarks=None)->np.ndarray:
        if i_landmarks is None:
            i_landmarks = self.get_landmarks()
        landmarks_labels = self.labels[i_landmarks]
        og_labels = None
        if self.discrete_labels:
            og_labels = self.label_encoder.inverse_transform(landmarks_labels)
        
        return og_labels,landmarks_labels


    def get_datapoints(self,indices)->np.ndarray:
        return self.data[indices]

    
    def find_corresponding_landmarks_at_level(self,i_landmarks=None,level=1)->np.ndarray:
        '''
        Retrieves the index of landmarks at a certain level as well as the original labeling of the those ladnamrks
        '''
        if i_landmarks is None:
            return self.get_landmarks()
        return self.get_landmarks()[i_landmarks]


    def _prepare_labels(self, labels:ArrayLike, data:NumericArrays)->np.ndarray:

        if labels is None:
            labels = np.zeros(data.shape[0])
        elif labels.dtype == bool:
            labels = labels.astype(int)
        if isinstance(labels,sp.sparray):
            labels= labels.todense()
        elif isinstance(labels,pd.DataFrame):
            labels = labels.select_dtypes(include=[np.number,np.bool]).to_numpy()
            labels = labels.astype(np.float64)
        labels = labels.squeeze()
        return labels


    def _encode_labels(self,labels:ArrayLike)->Tuple[LabelEncoder,np.ndarray]:
            
            if isinstance(labels,sp.sparray):
                labels = labels.todense()
            labels = labels.squeeze()#necessary for pandas.df

            label_encoder = LabelEncoder()
            label_encoder.fit(labels)
            encoded_labels = label_encoder.transform(labels)
            return label_encoder,encoded_labels


    def return_labels(self,i_indices:np.ndarray)->Tuple[ArrayLike,NumericArrays]:
        i_labels = self.labels[i_indices]
        og_labels=None
        if self.discrete_labels:
            og_labels = self.label_encoder.inverse_transform(i_labels)
        return og_labels,i_labels

"""
class HSNEGraph(KNNGraph):
    def __init__(self,
                data,landmark_sampling:OOP_Sampling.BaseSampling,
                connection:OOP_Connecting.BaseConnecting,labels=None,n:int=10,name:str="Root graph",
                data_name:str="unnamed",discrete_labels=True,weighted=True,directed=True,storage_dir="",seed=None):
        self.data = data
        self.seed = seed
        self.discrete_labels = discrete_labels

        if not self.discrete_labels or labels is None: #continous or no labels
            self.label_encoder, self.labels = None,self._prepare_labels(labels,self.data)
        else:
            self.label_encoder, self.labels = self._encode_labels(labels)
            
        self.n_neighbors = n
        self.knn_graph = sp.csr_array(kneighbors_graph(self.data,n_neighbors=self.n_neighbors))
        self.name = name
        self.data_name = data_name
        self.landmark_sampling = landmark_sampling
        self.connection = connection
        self.level=1
        self.weighted = True #generated weighted graphs
        self.directed = directed 
        self.storage_dir = storage_dir
        self.weighted_knn_graph = self.create_weighted_knn_graph(self.knn_graph)    
        self.sampling_method()
        self.connection_method()

    def create_weighted_knn_graph(self,knn_graph:sp.sparray)->sp.sparray:
        if not self.directed:
            knn_graph = knn_graph + knn_graph.T
            knn_graph[knn_graph>1] = 1

        K = knn_graph.sum(axis=1)
        distances = self.get_knn_distances(knn_graph,self.data,K)
        perplexities = self.get_perplexities(distances=distances,k=K)
        T = self.distance_to_probs(distances,perplexities,knn_graph)
        return T


    def distance_to_probs(self,distances:sp.sparray,perplexities:np.ndarray,knn_graph:sp.sparray):
        row, col = knn_graph.nonzero()
        # Expensive operation (example: square values)
        safe_exp = np.clip(distances[row, col]**2/perplexities[row],-709,709)
        distances.data = np.exp(safe_exp)
        distances = normalize(distances,"l1",axis=1)
        return distances

    
    def find_sigma(self,distances:np.ndarray,k:int,max_iter=100,tol=1e-4):
        
        size = len(distances)
        if size == 0:
            raise ValueError("Distance array cannot be empty")
        
        beta = 1.0  # Initialize beta
        min_beta, max_beta = -np.inf, np.inf
        log_perplexity = np.log(k/3)

        for iteration in range(max_iter):
            # Compute Gaussian kernel
            safe_exp = np.clip(-beta*distances,-709,709)
            exp_values = np.exp(safe_exp)
            
            sum_exp_values = np.sum(exp_values) + 1e-10  # Avoid division by zero
            probs = exp_values / sum_exp_values  # Normalize
            
            # Compute entropy
            entropy = np.sum(beta * distances * probs) / sum_exp_values + np.log(sum_exp_values)
            entropy_diff = entropy - log_perplexity  # Difference from target
            
            if np.abs(entropy_diff) < tol:
                break
            
            if entropy_diff > 0:
                min_beta = beta
                beta = beta * 2 if max_beta == np.inf else (beta + max_beta) / 2
            else:
                max_beta = beta
                beta = beta / 2 if min_beta == -np.inf else (beta + min_beta) / 2

        sigma = np.sqrt(1 / (2 * beta))
        return sigma

    
    
    def get_perplexities(self,distances:sp.sparray,k:np.ndarray)->np.ndarray:
        std = np.empty(distances.shape[0])
        for i in range(distances.shape[0]):
            idx_nonzero = distances[i].nonzero()
            k_distances = distances[i,idx_nonzero].toarray()
            std[i] = self.find_sigma(distances=k_distances,k=k[i])
        return std
    

    def get_knn_distances(self,knn_graph:sp.csr_matrix,X,K:np.ndarray)->sp.sparray:

        i=0 #runs along the col index that is equal to the value indices
        for row,k in enumerate(K):
            point_A = X[row,:]

            for _ in range(int(k)):
                point_B = X[knn_graph.indices[i],:]
                distance = np.linalg.norm(point_B - point_A)
                knn_graph.data[i] = distance
                i+=1

        return knn_graph

    def create_T_cumsum(self)->Tuple[np.ndarray,np.ndarray]:
        #graph = self.create_graph(self.weighted_knn_graph,weighted=self.weighted,directed=self.directed)
        return self.create_T_cumsum_from_matrix(self.weighted_knn_graph)
    

    def connection_method(self)->None:
        if isinstance(self.connection,OOP_Connecting.LandmarksToLandmarksBFS) or isinstance(self.connection,OOP_Connecting.StateToLandmarksBFS):
            graph = self.create_graph(self.knn_graph,weighted=False,directed=self.directed)#T correct
            self.connection.connect_landmarks(graph,self.landmark_sampling)
        
        elif isinstance(self.connection,OOP_Connecting.StateToLandmarksRandomWalks) or isinstance(self.connection,OOP_Connecting.LandmarksToLandmarksRandomWalks):
            T_probs_cumsum,T_states = self.create_T_cumsum()
            self.connection.connect_landmarks(T_probs_cumsum,T_states,self.landmark_sampling)

        elif isinstance(self.connection,OOP_Connecting.StateToLandmarksExact):
            T = self.weighted_knn_graph
            self.connection.connect_landmarks(T,self.landmark_sampling)
        
        elif isinstance(self.connection,OOP_Connecting.Connectivity):
            graph = self.create_graph(self.knn_graph,self.directed,False)
            self.connection.connect_landmarks(graph,self.get_landmarks())

        elif isinstance(self.connection,OOP_Connecting.ClusterRandomWalks):
            graph = self.create_graph(self.knn_graph,self.directed,False)
            T_probs_cumsum,T_states = self.create_T_cumsum()
            self.connection.connect_landmarks(graph,T_probs_cumsum,T_states,self.get_landmarks())

        else:
            raise Exception("Unrecognized Connection Method")
"""