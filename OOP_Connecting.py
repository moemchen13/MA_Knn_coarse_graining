from abc import ABC,abstractmethod
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spilu,spsolve,lgmres,gmres,LinearOperator,lobpcg
from typing import Union,Tuple
from tqdm import tqdm
import os
from igraph import Graph
import OOP_Sampling
import multiprocessing as mp
import datetime
from collections import deque
from sklearn.preprocessing import normalize
from functools import partial
#import PETSc 
#from petsc4py import PETSc
#import pypardiso

NumericArrays = Union[np.ndarray,sp.sparray]


class BaseConnecting(ABC):
    def __init__(self,storage_directory=".",seed:int=None,**kwargs):
        self.seed = seed
        self.storage_directory = storage_directory
        self.T = None


    def get_I(self):
        raise Exception("This method doesn't exist for this class.")


    def set_seed(self):
        if self.seed is not None:
            np.random.seed(self.seed)


    @abstractmethod
    def connect_landmarks(self)->sp.sparray:
        pass


    def save_all(self,filename,verbose:bool=False):
        class_attributes = self.__dict__.keys()
        attributes = ["I","T","f_cluster","W"]
        for attribute in attributes:
            if attribute in class_attributes:
                filename = self.save_file(filename= filename+"_"+attribute,file=getattr(self,attribute))
                if verbose:
                    print(f"{attribute} saved to {filename}")


    def save_file(self,filename:str,file:NumericArrays)->str:
        path:str = os.path.join(self.storage_directory,filename)
        if type(file) == sp.sparray:
            sp.save_npz(path,file)
        if type(file) == np.ndarray:
            np.save(path,file)
        return path


    def save_T(self,T):
        filename = self.save_file(T,"T")
        self.filename_T = filename


    def new_connecting(self):
        base_arguments = {key: value for key, value in self.__dict__.items() if key not in ["T"]}
        return self.__class__(**base_arguments)
    
    
    def load_file(self,filename)->sp.sparray:
        return sp.load_npz(os.path.join(self.storage_directory,filename))
    

    def get_T(self)->sp.sparray:
        #print("start Loading T")
        if self.T is None:
            raise Exception("Run connect first")
        return self.T


    def delete(self)->None:
        if self.T is not None:
            self.T = None


    def to_transition_matrix(self,adj:NumericArrays)->NumericArrays:
        T = normalize(adj,"l1",axis=1)
        return T
        

class StateToLandmarksBFS(BaseConnecting):
    def __init__(self,k:int=10,W:np.ndarray=None,max_nodes:int=None,**kwargs):
        super().__init__(**kwargs)
        self.W = W
        self.I = None
        self.k = k
        self.max_nodes = max_nodes

    def new_connecting(self):  
        base_arguments = {key: value for key, value in self.__dict__.items() if key not in ["T","I"]}
        if self.W is None:
            W = None
        else:
            I = self.get_I()
            W = self.W @ I
        base_arguments["W"] = W
        
        return self.__class__(**base_arguments)
    

    def save_I(self,I:sp.sparray,data_name:str="unnamed",verbose:bool=False)->None:
        filename:str = self.save_file(I,type="I",data_name=data_name)
        if verbose:
            print(f"Saved to {filename}")


    def get_I(self)->sp.sparray:
        if self.I is None:
            raise Exception("Run connect first")
        return self.I


    def delete(self)->None:
        super().delete()
        self.filename_I =None
    

    def retrieve_k_nearest_neighbors(self,graph:Graph,start_state:int,landmark_idx:dict,k:int,max_nodes_to_visit:int):
        #BFS with stop conditions of nodes to visit or k landmarks got or no more nodes to visit
        found_landmarks = 0
        checked_nodes = 0
        under_max_nodes = True
        k_nearest = []
        distances = []

        points_to_visit = deque([(start_state,0)])
        has_nodes_to_visit = True
        points_known = set([start_state])


        while found_landmarks<k and has_nodes_to_visit and under_max_nodes:
            
            point,hops = points_to_visit.popleft()
            new_points = graph.neighbors(point,mode="out")

            for new_point in new_points:
                if new_point not in points_known:
                    points_to_visit.append((new_point,hops+1))
                    points_known.add(new_point)

            if point in landmark_idx and point != start_state:
                k_nearest.append(point)
                distances.append(hops)
                found_landmarks +=1   

            has_nodes_to_visit = len(points_to_visit)>0

            if max_nodes_to_visit:
                checked_nodes =+ 1
                under_max_nodes  =checked_nodes<max_nodes_to_visit

        return np.array(k_nearest),np.array(distances)


    def get_index_landmark(self,nodes:np.ndarray,landmark_idx:dict):
        positions = np.array([landmark_idx.get(node, None) for node in nodes]) #get-1 if no is found
        return positions
    

    def connect_landmarks(self,graph:Graph,landmarks_sampling:OOP_Sampling.BaseSampling):
        landmarks = landmarks_sampling.get_landmarks()
        landmarks_idx = {val: idx for idx, val in enumerate(landmarks)}
        n_states = graph.vcount()
        n_landmarks = len(landmarks)
        I = sp.dok_array((n_states,n_landmarks))

        #for start_state in tqdm(range(n_states)):
        for start_state in range(n_states):
            k_neighbors,_ = self.retrieve_k_nearest_neighbors(graph,start_state,landmarks_idx,k=self.k,max_nodes_to_visit=self.max_nodes)
            k_neighbors = self.get_index_landmark(k_neighbors,landmarks_idx)
            I[start_state,k_neighbors] = 1

        I = self.to_transition_matrix(I)    
        I = I.tocsr()
        
        if self.W is None:
            T = I.T@I
        else:
            T = I.T*self.W[np.newaxis,:]@I

        T = self.to_transition_matrix(T)
        #delete landmarks not connected to anything
        mask = T.sum(axis=1)!=0
        T = T[mask][:,mask]
        landmarks_sampling.delete_landmarks(mask)
        I = I[:,mask]
        self.I = I
        self.T = T


class LandmarksToLandmarksBFS(BaseConnecting):
    def __init__(self,k:int=10,use_distances:bool=False,max_nodes:int=None,**kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.use_distances = use_distances
        self.max_nodes = max_nodes


    def retrieve_k_nearest_neighbors(self,graph:Graph,start_state:int,landmark_idx:dict,k:int,max_nodes_to_visit=None):
        #BFS with stop conditions of nodes to visit or k landmarks got or no more nodes to visit
        found_landmarks = 0
        checked_nodes = 0
        under_max_nodes = True
        k_nearest = []
        distances = []

        points_to_visit = deque([(start_state,0)])
        has_nodes_to_visit = True
        points_known = set([start_state])

        while found_landmarks<k and has_nodes_to_visit and under_max_nodes:
            
            point,hops = points_to_visit.popleft()
            new_points = graph.neighbors(point,mode="out")

            for new_point in new_points:
                if new_point not in points_known:
                    points_to_visit.append((new_point,hops+1))
                    points_known.add(new_point)

            if point in landmark_idx and point != start_state:
                k_nearest.append(point)
                distances.append(hops)
                found_landmarks +=1   

            has_nodes_to_visit = len(points_to_visit)>0

            if max_nodes_to_visit:
                checked_nodes =+ 1
                under_max_nodes  =checked_nodes<max_nodes_to_visit

        return np.array(k_nearest),np.array(distances)


    def connect_landmarks(self,graph:Graph,landmarks_sampling:OOP_Sampling.BaseSampling)->None:
        landmarks:np.ndarray = landmarks_sampling.get_landmarks()
        n_landmarks = len(landmarks)
        landmark_idx = {landmark: idx for idx, landmark in enumerate(landmarks) }
        adj_landmarks = sp.lil_array((n_landmarks,n_landmarks))

        #for i, start_landmark in tqdm(enumerate(landmarks), total=n_landmarks):
        for i, start_landmark in enumerate(landmarks):
            neighbors,distances = self.retrieve_k_nearest_neighbors(graph,start_landmark,landmark_idx,self.k,self.max_nodes)

            positions = np.array([landmark_idx.get(neighbor, -1) for neighbor in neighbors])
            if self.use_distances:
                adj_landmarks[i,positions] = distances
            else:
                if len(positions) !=0:
                    adj_landmarks[i,positions]= 1 / len(positions)

        if self.use_distances:
            adj_landmarks = adj_landmarks.tocsr()
            adj_landmarks.data = adj_landmarks.max()+1-adj_landmarks.data
            T_landmarks = self.to_transition_matrix(adj_landmarks.tocsr())
        else:
            T_landmarks = adj_landmarks.tocsr()
        self.T = T_landmarks  


class LandmarksToLandmarksRandomWalks(BaseConnecting):
    def __init__(self,n_walks,**kwargs):
        super().__init__(**kwargs)
        self.n_walks = n_walks


    def parallel_walks_landmarks_to_landmarks(self,T_probs_cumsum:NumericArrays,T_states:NumericArrays,landmarks:np.ndarray)->sp.sparray:
        self.set_seed()
        return parallel_walks_to_landmarks(T_probs_cumsum,T_states,landmarks,landmarks,contain_self_allowed=False,n_walks=self.n_walks)


    def connect_landmarks(self,T_probs_cumsum:NumericArrays,T_states:NumericArrays,landmark_sampling:OOP_Sampling.BaseSampling):
        landmarks = landmark_sampling.get_landmarks()
        T = self.parallel_walks_landmarks_to_landmarks(T_probs_cumsum,T_states,landmarks)
        T = self.to_transition_matrix(T)
        self.T = T


class StateToLandmarksRandomWalks(BaseConnecting):
    def __init__(self,W:np.ndarray=None,n_walks:int=100,contain_self:bool=True,**kwargs):
        super().__init__(**kwargs)
        self.W = W
        self.I = None
        self.n_walks = n_walks
        self.contain_self=contain_self


    def new_connecting(self):
        base_arguments = {key: value for key, value in self.__dict__.items() if key not in ["T","I"]}
        if self.W is None:
            W = None
        else:
            I = self.get_I()
            W = self.W @ I

        
        base_arguments["W"] = W

        return self.__class__(**base_arguments)


    def save_I(self,I:sp.sparray,data_name:str="unnamed",verbose:bool=False)->None:
        filename = self.save_file(I,kind="I",data_name=data_name)
        if verbose:
            print(f"Saved to {filename}")


    def get_I(self)->sp.sparray:
        return self.I
    

    def delete(self)->None:
        super().delete()
        self.filename_I =None
    

    def parallel_walks_states_to_landmarks(self,T_probs_cumsum,T_states,landmarks):
        self.set_seed()
        return parallel_walks_to_landmarks(T_probs_cumsum,T_states,np.arange(T_states.shape[0]),landmarks,contain_self_allowed=self.contain_self,n_walks=self.n_walks)
    


    def connect_landmarks(self,T_probs_cumsum:NumericArrays,T_states:NumericArrays,landmarks_sampling:OOP_Sampling.BaseSampling)->None:
        landmarks = landmarks_sampling.get_landmarks()
        I = self.parallel_walks_states_to_landmarks(T_probs_cumsum,T_states,landmarks)
        I = self.to_transition_matrix(I)
        I = I.tocsr()
        if self.W is None:
            T_landmarks = I.T@I
        else:
            T_landmarks = I.T*self.W[None,:]@I
        
        #self.I = I
        T_landmarks = T_landmarks.tocsr()
        T_landmarks = self.to_transition_matrix(T_landmarks)
        
        mask = T_landmarks.sum(axis=1)!=0
        T_landmarks = T_landmarks[mask][:,mask]
        I = I[:,mask]
        self.I = I
        self.T = T_landmarks
        landmarks_sampling.delete_landmarks(mask)



    
class StateToLandmarksExact(BaseConnecting):
    def __init__(self,W:np.ndarray=None,use_gambler=False,threshold_I:float=None,threshold_T:float=None,solver_name="default",**kwargs):
        super().__init__(**kwargs)
        self.W = W
        self.I = None
        self.use_gambler=use_gambler
        self.threshold_I = threshold_I
        self.threshold_T = threshold_T
        self.solver_name = solver_name

    def new_connecting(self):
        base_arguments = {key: value for key, value in self.__dict__.items() if key not in ["T","I"]}
        if self.W is None:
            W = None
        else:
            I = self.get_I()
            W = self.W @ I 
        
        base_arguments["W"] = W
        return self.__class__(**base_arguments)


    def save_I(self,I:sp.sparray,data_name:str="unnamed",verbose:bool=False)->None:
        filename = self.save_file(I,kind="I",data_name=data_name)
        if verbose:
            print(f"Saved to {filename}")


    def get_I(self)->sp.sparray:
        return self.I


    def delete(self)->None:
        super().delete()
        self.filename_I =None


    def solver(self,A:sp.sparray,b:sp.sparray):
        N = A.shape[0]
        solver_name = self.solver_name
        if solver_name is None:
            solver_name == "best"
        if solver_name == "best":
            if N<5e3:
                solver_name = "default"
            #elif N<1e5:
            #    solver_name = "SuperLU"
            else:
                solver_name = "iterative"
        
        tolerance = self.threshold_I
        if tolerance is None:
            tolerance=1e-3

        #if solver_name == "SuperLU": #direct solver but multithreading adn enhanced methods overall
        #    X = pypardiso.spsolve(A,b)

        if solver_name == "iterative" or solver_name=="gmres":
            X = solve_grmes(A,b,tolerance)
        elif solver_name == "lgmres": #iterative LU Preconditioning
            X = solve_lgrmes(A,b,tolerance) 
        #elif solver_name=="lobpcg":
        #    X = reformulation_lobpcg(A,b,tolerance)
        elif solver_name=="petsc":
            X = block_krylov_petsc(A, b, tol=1e-6, max_iter=100, solver_type="gmres")
        else:
            X = spsolve(A,b)
        return X


    def gamblers_ruin(self,T:sp.sparray,landmarks:np.ndarray,no_landmarks:np.ndarray)->sp.sparray:
        B = T[no_landmarks,:] 
        B = B[:,landmarks] #(N-n,n)

        L_U = T[no_landmarks,:] 
        L_U = L_U[:,no_landmarks] #(N-n,N-n)
        L_U = sp.csc_array((sp.eye(L_U.shape[0])-L_U).tocsc())

        I = self.solver(L_U.tocsc(),B.tocsc())
        I = I.tocsr()
        return I


    def tsne_solution(self,T:sp.sparray,landmarks:np.ndarray,no_landmarks:np.ndarray)->sp.sparray:
        N = T.shape[0]
        T = sp.csr_array(sp.eye(N)-T)
        B = T[no_landmarks,:]
        B = B[:,landmarks] #(N-n,n)

        L_U = T[no_landmarks,:]
        L_U = L_U[:,no_landmarks] #(N-n,N-n)

        I = self.solver(L_U.tocsc(),-B.tocsc())
        I = I.tocsr()
        return I
    

    def retrieve_exact_solution(self,T:sp.sparray,landmarks:np.ndarray)->None:
        N = T.shape[0]
        no_landmarks = np.setdiff1d(np.arange(N),landmarks)
        if self.use_gambler:
            I = self.gamblers_ruin(T,landmarks,no_landmarks)
        else:
            I = self.tsne_solution(T,landmarks,no_landmarks)

        def insert_landmark_rows(FA, landmarks,no_landmarks,N):
            #fill with landmarks first    
            data = [1]*len(landmarks)
            row_indices = landmarks.tolist()
            col_indices = np.arange(len(landmarks)).tolist()

            FA_coo = FA.tocoo()
            for row, column, value in zip(FA_coo.row, FA_coo.col, FA_coo.data):
                data.append(value)
                row_indices.append(no_landmarks[row])
                col_indices.append(column)

            return sp.csr_array((data, (row_indices, col_indices)), shape=(N, FA.shape[1]))

        I = insert_landmark_rows(I,landmarks,no_landmarks,N)
        if self.threshold_I is not None:
            I = self.delete_small_values(I,self.threshold_I)
        return I


    def connect_landmarks(self,T:sp.sparray,landmarks_sampling:OOP_Sampling.BaseSampling)->None:
        T = T.tocsr()
        landmarks = landmarks_sampling.get_landmarks()
        I = self.retrieve_exact_solution(T,landmarks)
        if self.W is None:
            T_landmarks = I.T@I
        else:
            T_landmarks = I.T *self.W[np.newaxis]@I
        self.I = I
        
        T_landmarks = self.to_transition_matrix(T_landmarks)
        if self.threshold_T is not None:
            T_landmarks = self.delete_small_values(T_landmarks,self.threshold_T)
            T_landmarks = self.to_transition_matrix(T_landmarks)
        
        self.T = T_landmarks
    

    @staticmethod
    def delete_small_values(some_array:sp.sparray,threshold:float)->sp.sparray:
        some_array.data = np.where(some_array.data >= threshold, some_array.data, 0)
        some_array.eliminate_zeros()
        return some_array


class Connectivity(BaseConnecting):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.f_cluster = None
        self.adj = None
        self.new_landmarks = None


    def new_connecting(self):
        base_arguments = {key: value for key, value in self.__dict__.items() if key not in ["T","f_cluster","adj"]}
        return self.__class__(**base_arguments)

    
    def connect_landmarks(self,graph:Graph,landmarks:np.ndarray):
        self.set_seed()
        f_cluster,new_landmarks = self.label_propagation(graph,landmarks)
        if new_landmarks !=[]:    
            self.new_landmarks = new_landmarks
        self.f_cluster = f_cluster
        adj = self.connectivity(f_cluster,graph)
        self.adj = adj
        T = self.to_transition_matrix(adj)
        self.T = T


    def get_adj(self)->sp.sparray:
        return self.adj

    @staticmethod
    def connectivity(f_clustering,graph):
        n_clusters = len(np.unique(f_clustering))
        adj = sp.dok_array((n_clusters,n_clusters))
        for point in range(graph.vcount()):
            clusterA = int(f_clustering[point])
            neighbors = graph.neighbors(point)
            for neighbor in neighbors:
                clusterB = int(f_clustering[neighbor])
                if clusterA != clusterB:
                    adj[clusterA,clusterB] +=1

        adj = adj.tocsr()
        return adj
    

    @staticmethod
    def label_propagation(graph:Graph,landmarks_indices:np.ndarray):
        N = graph.vcount()
        f_clustering = np.ones(N)*-1

        new_landmarks = []
        landmark_set = set(landmarks_indices)

        for comp in graph.components():
            comp_set = set(comp)
            
            if comp_set.isdisjoint(landmark_set): 
                sampled = np.random.choice(list(comp_set))
                new_landmarks.append(sampled)

        new_landmarks = [int(new_landmark) for new_landmark in new_landmarks]
        landmarks_indices = np.sort(np.concatenate([landmarks_indices,new_landmarks]))

        start_points = landmarks_indices
        n_landmarks = len(landmarks_indices)
        neighbors_to_visit = dict()
        weights_to_visit = dict()

        #initialize all clusters
        for i,start_point in enumerate(start_points):
            neighbors_to_visit[i] = [start_point]
            weights_to_visit[i] = [1]
        
        left_unclustered:int = graph.vcount()
        round_without_clustering = 0

        while(left_unclustered>0):
            #print(left_unclustered)
            clusters = np.random.permutation(np.arange(n_landmarks))
            for cluster in clusters:
                #add one node at a time per cluster
                for i in range(len(neighbors_to_visit[cluster])):
                    neighbor = int(neighbors_to_visit[cluster].pop(0))
                    weights_to_visit[cluster].pop(0)
                    if f_clustering[neighbor] == -1:
                        f_clustering[neighbor] = cluster
                        left_unclustered -= 1
                        new_neighbors_in = graph.neighbors(neighbor,mode="in")
                        new_neighbors_out = graph.neighbors(neighbor,mode="out")
                        new_neighbors = new_neighbors_in+ new_neighbors_out

                        eid_in = [graph.get_eid(new_neighbor,neighbor,directed=True) for new_neighbor in new_neighbors_in]
                        eid_out = [graph.get_eid(neighbor,new_neighbor,directed=True) for new_neighbor in new_neighbors_out]
                        new_weights = [graph.es[eid]["weight"] for eid in eid_in +new_neighbors_out]

                        #reorder to sample highest weight first
                        sorting = np.argsort(weights_to_visit[cluster] + new_weights)
                        neighbors_to_visit[cluster] = list(np.array(neighbors_to_visit[cluster] + new_neighbors)[sorting])
                        weights_to_visit[cluster] = list(np.array(weights_to_visit[cluster] + new_weights)[sorting])
                        break
        
        return f_clustering,new_landmarks 
    


class ClusterRandomWalks(BaseConnecting):
    def __init__(self,n_walks,**kwargs):
        super().__init__(**kwargs)
        self.n_walks = n_walks
        self.f_cluster = None
        self.new_landmarks = None


    def new_connecting(self):
        base_arguments = {key: value for key, value in self.__dict__.items() if key not in ["T","f_cluster"]}
        return self.__class__(**base_arguments)

    @staticmethod
    def label_propagation(graph:Graph,landmarks_indices:np.ndarray):
        N = graph.vcount()
        f_clustering = np.ones(N)*-1
        start_points = landmarks_indices
        n_landmarks = len(landmarks_indices)
        
            
        neighbors_to_visit = dict()
        #initialize all clusters
        for i,start_point in enumerate(start_points):
            neighbors_to_visit[i] = [start_point]
        
        left_unclustered:int = graph.vcount()

        new_landmarks = []
        round_without_clustering =0
        mode="out"
        while(left_unclustered>0):
            round_without_clustering +=1
            clusters = np.random.permutation(np.arange(n_landmarks))
            for cluster in clusters:
                neighbors_cluster = neighbors_to_visit[cluster]
                neighbors_to_visit[cluster] = []
                for neighbor in neighbors_cluster:
                    if f_clustering[neighbor] == -1:
                        f_clustering[neighbor] = cluster
                        left_unclustered -= 1
                        neighbors_to_visit[cluster] += graph.neighbors(neighbor,mode=mode)
                        round_without_clustering = 0
            if round_without_clustering == graph.vcount():
                if mode == "all":
                    new_landmark = np.random.choice(np.where(f_clustering==-1)[0])
                    new_landmarks.append(new_landmark)
                    n_landmarks = n_landmarks+1
                    neighbors_to_visit[n_landmarks] = [new_landmark] 
                mode="all"
        return f_clustering,new_landmarks
    
    """
    def connect_landmarks(self,graph,landmarks):
        self.set_seed()
        f_cluster = self.label_propagation(graph,landmarks)
        self.f_cluster = f_cluster
        adj = random_walk_to_cluster(graph,f_cluster,self.n_walks)
        adj = adj.tocsr()
        T = self.to_transition_matrix(adj)
        self.save_T(T)
    """
    def connect_landmarks(self,graph,transition_cumsum,transition_states,landmarks):
        self.set_seed()
        f_cluster,new_landmarks = self.label_propagation(graph,landmarks)
        if new_landmarks != []:
            self.new_landmarks = new_landmarks
        self.f_cluster = f_cluster
        adj = random_walks_to_clusters(transition_cumsum,transition_states,f_cluster,self.n_walks)
        adj = adj.tocsr()
        T = self.to_transition_matrix(adj)
        self.T = T



############## walks functions start here ###############


def random_walk_to_no_cluster_point(transition_matrix_cumsum,transition_states,start_state,cluster,f_clustering):
    state = start_state 
    i=0
    while f_clustering[state]==cluster:
        i+=1
        i_state = np.searchsorted(transition_matrix_cumsum[state, :], np.random.random(), side="left")
        state = transition_states[state,i_state]
        if i >1000:
            return None
    return f_clustering[state]


def random_walks_to_clusters(transition_cumsum,transition_states,f_clustering,n_walks=100):
    N = transition_cumsum.shape[0]
    n = len(np.unique(f_clustering))
    T_cluster = sp.dok_array((n,n))
    #for state in tqdm(range(N)):
    for state in range(N):
        cluster_state = f_clustering[state]
        for walks in range(n_walks):
            different_cluster = random_walk_to_no_cluster_point(transition_cumsum,transition_states,state,cluster_state,f_clustering)
            if different_cluster is not None:
                T_cluster[int(cluster_state),int(different_cluster)] += 1
    return T_cluster

############### RW cluster ##################


def random_walk_to_landmarks(start_state,T_probs_cumsum,T_states,landmarks,contain_self_allowed):
    current_state = start_state
    termination_okay= False
    landmark_found=False
    i=0
    while not(termination_okay and landmark_found):

        index_new_state = np.searchsorted(T_probs_cumsum[current_state,:],np.random.random(),side="left")
        current_state = T_states[current_state,index_new_state]
        
        termination_okay = True
        
        if not contain_self_allowed and current_state == start_state:
            termination_okay=False

        landmark_found=True
        end_state = landmarks.get(current_state, None) 
        if end_state is None:
            landmark_found = False
        i+=1
        if i>1000:
            return None
    return end_state

def multi_random_walks(start_state:int,T_probs_cumsum,T_states,destination_states_dict:dict,n_walks:int,contain_self_allowed:bool):
        res = np.empty((n_walks,))
        for i in range(n_walks):
            end_state = random_walk_to_landmarks(start_state,T_probs_cumsum,T_states,destination_states_dict,contain_self_allowed)
            res[i] = end_state
        res = res[~np.isnan(res)]
        columns,counts = np.unique(res,return_counts=True)
        return columns,counts


def parallel_walks_to_landmarks(T_probs_cumsum,T_states,i_start_states,i_destination_states,contain_self_allowed=False,n_walks=100):

    destination_states_dict = {num:idx for idx,num in enumerate(i_destination_states)}
    multi_walks_func = partial(multi_random_walks, T_probs_cumsum=T_probs_cumsum, 
                               T_states=T_states, destination_states_dict=destination_states_dict,
                               contain_self_allowed=contain_self_allowed, n_walks=n_walks)
    
    with mp.Pool(processes=mp.cpu_count()-1) as pool:
        results = pool.map(multi_walks_func,i_start_states)

    row_indices = []
    col_indices = []
    data_values = []
    for r,(cols, vals) in enumerate(results):
        row_indices.extend([r] * len(cols))  # Repeat the row index
        col_indices.extend(cols.astype(int))  # Store column indices
        data_values.extend(vals)  # Store values
    
    random_walks = sp.coo_array((data_values, (row_indices, col_indices)),shape=(len(i_start_states),len(i_destination_states)))
    random_walks = random_walks.tocsr()
    return random_walks


def n_random_walk_to_landmarks(start_state,T_probs_cumsum,T_states,destination_landmarks,n_walks,contain_self_allowed,result,index):
    #subfunction for next function
    n_random_walks = sp.dok_array((len(destination_landmarks),),dtype=float)
    for _ in range(n_walks):
        end_state = random_walk_to_landmarks(start_state,T_probs_cumsum,T_states,destination_landmarks,contain_self_allowed=contain_self_allowed)
        
        n_random_walks[end_state] +=1
    result[index] = n_random_walks


#def parallel_walks_to_landmarks(T_probs_cumsum,T_states,i_start_states,i_destination_states,contain_self_allowed=True,n_walks=100):
#    
#    random_walks = sp.dok_array((len(i_start_states),len(i_destination_states)))
#    i_destination_states = {num:idx for idx,num in enumerate(i_destination_states)}
#
#    for start_state in tqdm(i_start_states):
#        for _ in range(n_walks):
#            end_state = random_walk_to_landmarks(start_state,T_probs_cumsum,T_states,i_destination_states,contain_self_allowed)
#            random_walks[start_state,end_state] += 1
#    return random_walks

def random_walk_to_cluster(graph:Graph,f_clustering:np.ndarray,n_walks):
    n_cluster = len(np.unique(f_clustering))
    adj = sp.dok_array((n_cluster,n_cluster))
    N = graph.vcount()
    #for start_node in tqdm(range(N)):
    for start_node in range(N):
        start_cluster = f_clustering[start_node]
        for _ in range(n_walks):
            end_cluster = walk_to_other_cluster(start_node,f_clustering[start_node],graph,f_clustering)
            adj[int(start_cluster),int(end_cluster)] += 1

    return adj


def walk_to_other_cluster(start_node,start_cluster,graph,f_clustering):
    current_node = start_node
    while True:
        neighbors = graph.neighbors(current_node,mode="out")
        if len(neighbors)>1:
            if graph.is_weighted():
                random_index = np.random.choice(len(neighbors),p=get_outgoing_weights(graph,current_node))
            else:
                random_index = np.random.randint(0,len(neighbors))
        else:
            random_index = 0
                
        current_node = neighbors[random_index]
        end_cluster = f_clustering[current_node]
        if start_cluster != end_cluster:
            return end_cluster


def get_outgoing_weights(graph:Graph, vertex:int):
    outgoing_edges = graph.incident(vertex, mode="out")
    return [graph.es[e]["weight"] for e in outgoing_edges] 



def block_krylov_petsc(A, B, tol=1e-6, max_iter=100, solver_type="gmres"):
    """
    Solves AX = B using PETSc's Block Krylov solvers and outputs a sparse solution matrix.

    Args:
        A_scipy: (N, N) SciPy sparse matrix (should be SPD for CG)
        B_scipy: (N, n) SciPy sparse or dense right-hand side
        tol: Convergence tolerance
        max_iter: Maximum iterations
        solver_type: Krylov method ("cg", "gmres", etc.)

    Returns:
        X_scipy: (N, n) SciPy sparse solution matrix (CSR format)
    """
    N, n = B.shape
    pass
    """
    # Convert SciPy sparse matrix to PETSc AIJ format
    A_petsc = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    A_petsc.assemble()

    # Convert B to PETSc dense format (since PETSc solvers expect a dense RHS)
    if sp.issparse(B):
        B = B.toarray()  # Convert to dense
    ##B_petsc = PETSc.Mat().createDense(size=B.shape, array=B)

    # Create an empty PETSc solution matrix
    X = np.zeros((N,n),dtype=np.float64)
    ##X_petsc = PETSc.Mat().createDense(size=(N, n))

    # Create Krylov solver
    ksp = PETSc.KSP().create()
    ksp.setOperators(A_petsc)
    ksp.setType(solver_type)

    # Set tolerances
    ksp.setTolerances(rtol=tol, max_it=max_iter)

    # Use a sparse preconditioner (ILU or AMG)
    pc = ksp.getPC()
    pc.setType("jacobi")
    #pc.setType("ilu")
    ##pc.setType("hypre")  # Algebraic Multigrid (good for large sparse systems)

    # Solve the system in one step
    ###ksp.solve(B_petsc, X_petsc)

    for col in range(n):
        b_petsc = PETSc.Vec().createWithArray(B[:, col])
        x_petsc = PETSc.Vec().createWithArray(X[:, col])
        ksp.solve(b_petsc, x_petsc)
        
    # Convert PETSc dense matrix to SciPy sparse matrix
    #X_dense = X_petsc.getDenseArray()  # Extract NumPy array
    X = sp.csr_array(X_petsc)  # Convert to sparse format

    return X
    """

"""
def reformulation_lobpcg(A,b,tolerance):
    M_inv = spilu(A)
    M = LinearOperator(A.shape, matvec=M_inv.solve)
    X = np.random.randn(A.shape[0],b.shape[1])
    X,_ = lobpcg(A=A,B=b,X=X,M=M,tol=tolerance)
    X = sp.csr_array(X)
    return X
"""


def solve_grmes(A,b,tolerance):
    ilu = spilu(A, drop_tol=tolerance)
    M = LinearOperator((A.shape[0], A.shape[0]),ilu.solve())
    X = []
    #for i in tqdm(range(b.shape[1])):
    for i in range(b.shape[1]):
        x= gmres(A=A, b=b[:,i].todense(), M=M)[0]
        x = sp.csr_array(x)
        X.append(x)
    
    X = sp.vstack(X).T
    return X


def solve_lgrmes(A,b,tolerance):
    ilu = spilu(A, drop_tol=tolerance)
    M = LinearOperator((A.shape[0], A.shape[0]),ilu.solve())
    X = []
    #for i in tqdm(range(b.shape[1])):
    for i in range(b.shape[1]):
        x= lgmres(A=A, b=b[:,i].todense(), M=M)[0]
        x = sp.csr_array(x)
        X.append(x)
    
    X = sp.vstack(X).T
    return X
