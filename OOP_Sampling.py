from abc import ABC,abstractmethod
import numpy as np
from scipy import sparse as sp
from typing import Union,Tuple
from tqdm import tqdm
from igraph import Graph
from sklearn.neighbors import kneighbors_graph

NumericArrays = Union[np.ndarray,sp.sparray]


class BaseSampling(ABC):
    def __init__(self,shrinkage:float= 0.1,storage_directory=".",seed:int=None):
        self.seed = seed
        self.storage_directory = storage_directory
        if shrinkage>=1 or shrinkage <= 0:
            raise Exception("No shrinkage value")
        self.shrinkage = shrinkage
        self.landmarks = None
    

    @abstractmethod
    def sampling(self):
        """Abstract class different in each subclass"""
        pass

    def set_seed(self)->None:
        if self.seed is not None:
            np.random.seed(self.seed)
    
    def get_landmarks(self)->np.ndarray:
        if self.landmarks is None:
            raise Exception("First call sampling landmarks not yet sampled")
        return self.landmarks
        
    def new_sampling(self):
        base_arguments = {key: value for key, value in self.__dict__.items() if key != "landmarks"}
        return self.__class__(**base_arguments)
    
    def delete_landmarks(self,mask:np.ndarray):
        self.landmarks = self.landmarks[mask]
    
    def add_landmark(self,indices:np.ndarray):
        self.landmarks = np.sort(np.concatenate([self.landmarks,indices]))
    
    def save_landmarks(self,filename:str,verbose:bool=False):
        if self.landmarks is None:
            print("no Landmarks sampled")
        else:
            np.save(filename+"_landmarks",self.landmarks)
            if verbose:
                print(f"Landmarks saved to {filename}")


class RandomSampling(BaseSampling):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def sampling(self, data:NumericArrays)->np.ndarray:
        self.set_seed()
        N:int  = data.shape[0]
        n_landmarks:int = int(N*self.shrinkage)
        landmarks = np.random.choice(np.arange(N),size=n_landmarks,replace=False)
        self.landmarks = np.sort(landmarks)
    

class HighestDegreeSampling(BaseSampling):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)


    def sampling(self,adj:NumericArrays)->np.ndarray:
        N:int = adj.shape[0]
        n_landmarks:int = int(self.shrinkage*N)
        degree = adj.sum(axis=0)
        degree = np.squeeze(np.asarray(degree))
        landmarks = np.argsort(degree)[-n_landmarks:]
        self.landmarks = np.sort(landmarks)
        
    
class RandomWalksSampling(BaseSampling):
    def __init__(self,n_walks=100,n_steps=50,use_shrinkage:bool=True,walks_threshold:float=1.5,**kwargs):
        super().__init__(**kwargs)
        self.n_walks = n_walks
        self.n_steps = n_steps
        self.use_shrinkage = use_shrinkage
        self.walks_threshold = walks_threshold

    def simulate_rw_from_dist(self,T:sp.sparray):
        self.set_seed()
        N = T.shape[0]
        state_dist = np.ones(N)/N
        #for i in tqdm(range(self.n_steps)):
        for i in range(self.n_steps):
            state_dist = state_dist @ T

        state_dist /= state_dist.sum()
        walks = np.random.choice(N,size=N*self.n_walks,p=state_dist)
        end_states = np.bincount(walks,minlength=N)
        return end_states


    def sampling(self,T:sp.sparray)->np.ndarray:
        walks = self.simulate_rw_from_dist(T)
        landmarks = np.where(walks>(self.n_walks*self.walks_threshold))[0]
        if self.use_shrinkage:
            N:int = T.shape[0]
            n_landmarks:int = int(self.shrinkage*N)

            landmarks = np.argsort(walks)[-n_landmarks:]
            lowest_value = walks[landmarks[0]]
            n_landmarks_treshhold = lowest_value/self.n_walks
            
            print(f"walks_treshhold of: {round(n_landmarks_treshhold,ndigits=4)}")
        else:
            print(f"walks_threshold of {round(self.walks_threshold,ndigits=4)}")
        self.landmarks = np.sort(landmarks)


class HighDegreeExclusionNN(BaseSampling):
    def __init__(self,k=10,**kwargs):
        super().__init__(**kwargs)
        self.k = k


    def create_graph(self,W:sp.sparray):
        N = W.shape[0]
        W_coo = W.tocoo()
        
        rows_graph = []
        cols_graph = []

        current_row = 0
        k_highest_cols = dict()

        for weight,(row,col) in zip(W_coo.data,zip(W_coo.row,W_coo.col)):
            if current_row != row:
                rows_graph += [current_row]*len(k_highest_cols.keys())
                cols_graph += list(k_highest_cols.keys())
                current_row=row
                k_highest_cols = dict()
            k_highest_cols[col]=weight

            if len(k_highest_cols.keys())>self.k:
                smallest_key = min(k_highest_cols, key=k_highest_cols.get)
                del k_highest_cols[smallest_key]
                    
        edges = [(s, t) for s, t in zip(rows_graph,cols_graph)]
        graph = Graph(N,edges=edges, directed=True)
        return graph
    

    def sampling(self,adj:sp.sparray):
        N= adj.shape[0]
        n_landmarks = int(self.shrinkage*N)
        landmarks = []
        degree = adj.sum(axis=0)
        order = list(np.argsort(degree))
        G = self.create_graph(adj)
        landmarks = []
        forbidden_points = set()
        for current_point in order:
            if current_point not in forbidden_points:
                landmarks.append(current_point)
                neighbors = G.neighbors(current_point,mode="out")
                for neighbor in neighbors:
                    forbidden_points.add(neighbor)

            if n_landmarks<= len(landmarks):
                break
        self.landmarks = np.sort(landmarks)   



class RandomSamplingExclusionNN(BaseSampling):
    def __init__(self,k=10,**kwargs):
        super().__init__(**kwargs)
        self.k = k
        
    def sampling(self,adj:sp.sparray):
        N = adj.shape[0]
        n_landmarks = int(self.shrinkage*N)
        G = self.create_graph(adj)
        landmarks = []

        points = np.arange(N)
        points = np.random.permutation(points)
        forbidden_points = set()

        for sampled_point in points:
            if sampled_point not in forbidden_points:
                landmarks.append(sampled_point)
                neighbors = G.neighbors(sampled_point,mode="out")
                for neighbor in neighbors:
                    forbidden_points.add(neighbor)

            if n_landmarks<= len(landmarks):
                break
        self.landmarks = np.sort(landmarks)
    

    def create_graph(self,W:sp.sparray):
        N = W.shape[0]
        W_coo = W.tocoo()
        
        rows_graph = []
        cols_graph = []

        current_row = 0
        k_highest_cols = dict()

        for weight,(row,col) in zip(W_coo.data,zip(W_coo.row,W_coo.col)):
            if current_row != row:
                rows_graph += [current_row]*len(k_highest_cols.keys())
                cols_graph += list(k_highest_cols.keys())
                current_row=row
                k_highest_cols = dict()
            k_highest_cols[col]=weight

            if len(k_highest_cols.keys())>self.k:
                smallest_key = min(k_highest_cols, key=k_highest_cols.get)
                del k_highest_cols[smallest_key]
                    
        edges = [(s, t) for s, t in zip(rows_graph,cols_graph)]
        graph = Graph(N,edges=edges, directed=True)
        return graph


