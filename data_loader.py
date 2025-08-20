import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import numpy as np
import scipy.sparse as sp
import pandas as pd
from typing import Union
from sklearn.datasets import fetch_20newsgroups,load_digits
from sklearn.feature_extraction.text import TfidfVectorizer
#from ucimlrepo import fetch_ucirepo

NumericArrays = Union[np.ndarray,sp.sparray]

def select_dataset(kind:str="MNIST",**kwargs):
    if kind=="MNIST":
        return mnist(**kwargs)  
    elif kind=="TASIC":
        return tasic(**kwargs)
    elif kind=="FMNIST":
        return fmnist(**kwargs)
    elif kind=="CIFAR":
        return cifar(**kwargs)
    elif kind=="20NEWSGROUP":
        return twentynewsgroup(**kwargs)
    elif kind=="SWISSROLL":
        return swissroll(**kwargs)
    elif kind=="DNA":
        return dna(**kwargs)
    elif kind=="DOUBLEHELIX":
        return doublehelix(**kwargs)
    elif kind=="TORUS":
        return torus(**kwargs)
    elif kind=="DIGITS":
        print("UNCOMENT Me")
        #return digits(**kwargs)


def mnist(directory:str="",train_test_split:bool=False,**kwargs):
    MNIST = MnistDataloader(directory=directory).load_data()
    class_table = np.array(["0","1","2","3","4","5","6","7","8","9"])
    X_train,X_test = [x.reshape((x.shape[0],-1)) for x in MNIST if len(x.shape)>1]
    y_train,y_test = MNIST[1],MNIST[3]
    if train_test_split:
        return X_train,X_test,y_train,y_test,class_table
    else:
        X,y = combine_Xy([X_train,X_test],[y_train,y_test])
        return X,y,class_table
    
"""    
def digits(**kwargs):
    pen_based_recognition_of_handwritten_digits = fetch_ucirepo(id=80) 
    X = np.array(pen_based_recognition_of_handwritten_digits.data.features)
    y = np.array(pen_based_recognition_of_handwritten_digits.data.targets).squeeze()
    return X,y,np.sort(np.unique(y))
"""


def fmnist(directory:str=".",train_test_split:bool=False,**kwargs):
    class_table = np.array([
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
    ])
    if directory is not None and directory != "":
        file_path_test=join(directory,"fashion-mnist_test.csv")
        file_path_train=join(directory,"fashion-mnist_train.csv")
    df_test = pd.read_csv(file_path_test)
    y_test = df_test["label"]
    X_test = sp.csr_array(df_test)[:,1:]
    df_train = pd.read_csv(file_path_train)
    y_train = df_train["label"]
    X_train = sp.csr_array(df_train)[:,1:]
    X = [X_train,X_test]
    y = [y_train,y_test]

    X = sp.vstack(X)
    y = np.hstack(y)
    return X,y,class_table
    

def cifar(directory="",**kwargs):
    class_table = np.array([
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
    ])

    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    folder_path = join("cifar-10-python","cifar-10-batches-py")
    if directory != "" and directory is not None:
        folder_path = join(directory,folder_path)
    
    files=[]
    for i in range(1,6):
        files.append(join(folder_path,"data_batch_"+str(i) ))
    test_file = join(folder_path, "test_batch")
    files.append(test_file)
        
    X = []
    y = []
    for file in files:
        batch_dict = unpickle(file)
        X.append(batch_dict[b'data'])
        y.append(batch_dict[b'labels'])
    
    X = np.vstack(X)
    y = np.hstack(y)

    return X,y,class_table


def tasic(directory:str="",return_colors:bool=False,**kwargs):
    data_path = "tasic-pca50.npy"
    colors_path = "tasic-colors.npy"
    types_path = "tasic-ttypes.npy"
    if directory is not None and directory != "":
        data_path = join(directory,data_path)
        colors_path = join(directory,colors_path)
        types_path = join(directory,types_path)
    
    X = np.load(data_path)
    colors = np.load(colors_path)
    types = np.load(types_path)
    class_table = np.unique(types)
    u_colors = []
    y = np.zeros(types.shape,dtype=np.int16)
    for i,category in enumerate(class_table):
        y[types==category] = i
        u_colors.append(colors[np.argmax(types==category)])

    if return_colors:
        return X,y,class_table,np.array(u_colors)
    return X,y,class_table


def twentynewsgroup(train_test_split:bool=False,**kwargs):
    vectorizer = TfidfVectorizer()

    dataset_train = fetch_20newsgroups(subset="train")
    y_train = dataset_train.target
    X_train = sp.csr_array(vectorizer.fit_transform(dataset_train.data))
    class_table = dataset_train.target_names

    return X_train,y_train,class_table


###continous figures###
def swissroll(n_circles=2.5,N=5000,noise=0,seed=None,**kwargs)->tuple[np.ndarray,np.ndarray]:
    #returns tuple with swissrol data and t 
    if seed is not None:
        np.random.seed(seed)
    t = n_circles*2*np.pi/2 * (1 + 2*np.random.rand(1,N))
    h = 21 * np.random.rand(1,N)
    data = np.concatenate((t*np.cos(t),h,t*np.sin(t))) + noise*np.random.randn(3,N)	
    return np.transpose(data), np.squeeze(t)


def doublehelix(num_points=500,**kwargs)->tuple[np.ndarray,np.ndarray]:
    t = np.linspace(0, 4 * np.pi, num_points) 
    x1 = np.cos(t)
    y1 = np.sin(t)
    z1 = t 
    #TODO ADD NOISE for all below

    # Second helix (shifted by π in phase)
    x2 = np.cos(t + np.pi)
    y2 = np.sin(t + np.pi)
    z2 = t  # Same z values

    # Combine both helices
    X = np.concatenate([x1, x2])
    Y = np.concatenate([y1, y2])
    Z = np.concatenate([z1, z2])
    return np.vstack([X,Y,Z]).T,np.hstack((t,t))


def dna(num_points: int = 12000, n_per_strand: int = 800, n_connections_per_circle: int = 6, n_circles: int = 2,**kwargs)->tuple[np.ndarray,np.ndarray]:
    # Define the total range of t based on number of circles
    t_max = n_circles * 2 * np.pi  

    num_points = int((num_points - n_circles*n_connections_per_circle*n_per_strand)/2)
    t = np.linspace(0, t_max, num_points)
    x1, y1, z1 = np.cos(t), np.sin(t), t
    x2, y2, z2 = np.cos(t + np.pi), np.sin(t + np.pi), t

    X = np.column_stack((np.concatenate([x1, x2]), np.concatenate([y1, y2]), np.concatenate([z1, z2])))
    t_full = np.hstack((t, t))

    num_connections = n_circles * n_connections_per_circle
    selected_indices = np.round(np.linspace(0, len(t) - 1, num_connections)).astype(int)  

    #connecting strand points
    lambda_vals = np.linspace(0, 1, n_per_strand)
    total_interpolated = num_connections * n_per_strand

    X_con = np.empty((total_interpolated, 3))
    t_con = np.empty(total_interpolated)

    idx = 0
    for j in selected_indices:  
        for lam in lambda_vals:
            X_con[idx, 0] = (1 - lam) * x1[j] + lam * x2[j]
            X_con[idx, 1] = (1 - lam) * y1[j] + lam * y2[j]
            X_con[idx, 2] = (1 - lam) * z1[j] + lam * z2[j]
            t_con[idx] = t[j]
            idx += 1
    X = np.vstack((X, X_con))
    t_full = np.hstack((t_full, t_con))
    X[:,2] = X[:,2]*2/np.max(X[:,2])-np.min(X[:,2])-1
    return X, t_full


def torus(num_points=2500,R=1,r=0.3,**kwargs)->tuple[np.ndarray,np.ndarray]:
    #r are the small circles making the torus
    #R is the big circle of the torus
    num_points = int(np.sqrt(num_points))
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, 2 * np.pi, num_points)  
    u, v = np.meshgrid(u, v)

    X = (R + r * np.cos(v)) * np.cos(u)
    Y = (R + r * np.cos(v)) * np.sin(u)
    Z = r * np.sin(v)

    x = X.flatten()
    y = Y.flatten()
    z = Z.flatten()
    return np.vstack([x,y,z]).T,z


def combine_Xy(Xs:list[NumericArrays],ys:list[np.ndarray])->tuple[NumericArrays,np.ndarray]:
    y = np.hstack(ys)
    if type(Xs[0])==np.ndarray:
        X = np.vstack(Xs)
    elif isinstance(Xs[0],(sp.sparray,sp.csr_array)):
        X = sp.vstack(Xs)
    else:
        raise Exception(f"Unknown format: {type(Xs[0])}")
    return X,y



class MnistDataloader(object):
    def __init__(self,directory = "", train_images_path=join('train-images-idx3-ubyte','train-images.idx3-ubyte'),train_labels_path=join('train-labels-idx1-ubyte','train-labels.idx1-ubyte'),test_images_path=join('t10k-images-idx3-ubyte','t10k-images.idx3-ubyte'), test_labels_path=join('t10k-labels-idx1-ubyte','t10k-labels.idx1-ubyte')):
        self.training_images_filepath = train_images_path
        self.training_labels_filepath = train_labels_path
        self.test_images_filepath = test_images_path
        self.test_labels_filepath = test_labels_path
        if directory is not None and directory != "":
            self.training_images_filepath = join(directory,train_images_path)
            self.training_labels_filepath = join(directory,train_labels_path)
            self.test_images_filepath = join(directory,test_images_path)
            self.test_labels_filepath = join(directory,test_labels_path)
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        return x_train, y_train,x_test, y_test