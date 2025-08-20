import numpy as np
import pandas as pd
import scipy.sparse as sp
import OOP_Multilevel_tsne
import OOP_Connecting
import OOP_Sampling
import data_loader
import helper
import argparse
import os
import time

dataset_choices = ["MNIST","TASIC","FMNIST","CIFAR","20NEWSGROUP","SWISSROLL","DNA","DOUBLEHELIX","TORUS","DIGITS"]
categorical_datasets = ["MNIST","TASIC","FMNIST","CIFAR","20NEWSGROUP","DIGITS"]
connection_choices = ["EXACT","SL-BFS","LL-BFS","SL-RW","LL-RW","CLUSTERRW","CON"]
sampling_choices = ["RANDOM","RANDOMNN","HUBS","HUBSNN","RW"]


def main():
    parser = argparse.ArgumentParser(description="Process a dataset and save results.")
    parser.add_argument(
        "--dataset", type=str, required=True,default="SWISSROLL",choices=dataset_choices, help="Name of the dataset to process"
    )
    parser.add_argument(
        "--sampling",type=str, required=True, default="RANDOM",choices=sampling_choices,help="Choose selection method for landmarks"
    )
    parser.add_argument(
        "--connection",type=str, required=True, default="CON",choices=connection_choices,help="Choose connection method for landmarks"
    )
    parser.add_argument(
        "--seed", type=int, required=False, default=42, help="Set seed"
    )
    parser.add_argument(
        "--output", type=str, required=False, default=".", help="Directory to save the output files"
    )
    parser.add_argument(
        "--noise",type=float, required=False,default=0,help="use noise on data"
    )
    parser.add_argument(
        "--reduction",type=float,required=False,default=0.2,help="set reduction factor"
    )
    parser.add_argument(
        "--directed",type=bool,required=False,default=False
    )
    parser.add_argument(
        "--weighted",type=bool,required=False,default=True
    )
    parser.add_argument(
        "--directory", type=str, required=False, default=None, help="Directory to load datasets"
    )
    parser.add_argument(
        "--k",type=int, required=False,default=10,help="k for kNN-graph"
    )
    parser.add_argument(
        "--level",type=int, required=False,default=1,help="levels to shrink the graph"
    )
    parser.add_argument(
        "--random_init", type=bool,required=False,default=False, help="TSNE embedding initialized with PCA or random"
    )
    

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"Processing dataset: {args.dataset}")
    print(f"Saving results to: {args.output}")

    seed=args.seed
    data_name = args.dataset
    directory = args.directory
    storage=args.output
    reduction = args.reduction
    directed = args.directed
    weighted = args.weighted
    sampling_strat = args.sampling
    connection_strat = args.connection
    k = args.k
    level = args.level
    random_init = args.random_init
    noise = args.noise

    continous=True
    if data_name in categorical_datasets:
        continous = False

    print(data_name)
    if not continous:
        X,y,class_table = data_loader.select_dataset(kind=data_name,directory=directory)
    else:
        X,y = data_loader.select_dataset(kind=data_name,directory=directory)
    N,n_features = X.shape

    if noise>0:
        generated_noise = helper.generate_noise(X=X,noise=noise,seed=seed)
        X = X + generated_noise

    if sampling_strat =="RANDOM":
        sampling = OOP_Sampling.RandomSampling(seed=seed)
    elif sampling_strat =="RANDOMNN":
        sampling = OOP_Sampling.RandomSamplingExclusionNN(k=k,seed=seed)
    elif sampling_strat =="RW":
        sampling = OOP_Sampling.RandomWalksSampling(use_shrinkage=True,seed=seed)
    elif sampling_strat =="HUBS":
        sampling = OOP_Sampling.HighestDegreeSampling(seed=seed)
    elif sampling_strat =="HUBSNN":
        sampling = OOP_Sampling.HighDegreeExclusionNN(k=k)

    if connection_strat =="EXACT":
        threshold = None
        if N>20000:
            threshold = 1e-4
        connection = OOP_Connecting.StateToLandmarksExact(threshold_I=threshold,threshold_T=threshold,use_gambler=False,W=np.ones(N),storage_directory=storage)
    elif connection_strat =="SL-BFS":
        connection = OOP_Connecting.StateToLandmarksBFS(k=10,W=np.ones(N),storage_directory=storage)
    elif connection_strat =="SL-RW":
        connection = OOP_Connecting.StateToLandmarksRandomWalks(n_walks=100,W=np.ones(N),storage_directory=storage)
    elif connection_strat =="LL-BFS":
        connection = OOP_Connecting.LandmarksToLandmarksBFS(k=10,storage_directory=storage)
    elif connection_strat =="LL-RW":
        connection = OOP_Connecting.LandmarksToLandmarksRandomWalks(n_walks=100,storage_directory=storage)
    elif connection_strat =="CON":
        connection = OOP_Connecting.Connectivity()
    elif connection_strat =="CLUSTERRW":
        connection = OOP_Connecting.ClusterRandomWalks(n_walks=100)

    filename = f"data:{data_name}_seed:{seed}_redu:{reduction}_k:{k}_dir:{directed}_wei:{weighted}_noise:{noise}"
    print(f"Start {filename}")
    start_time = time.time()
    lv1 = OOP_Multilevel_tsne.KNNGraph(data=X,n=k,labels=y,data_name=data_name,directed=directed,weighted=weighted,landmark_sampling=sampling,connection=connection,discrete_labels=not continous,storage_dir=storage)
    print(f"LV1:{start_time - time.time()}")
    start_time = time.time()
    lv1.TSNE_to_attribute(random_init=random_init)
    print(f"LV1 TSNE:{start_time - time.time()}")

    if level>1:
        start_time = time.time()
        lv2 = lv1.create_new_level()
        print(f"LV2:{start_time - time.time()}")

        start_time = time.time()
        lv2.TSNE_to_attribute(random_init=random_init)
        print(f"LV2 TSNE:{start_time - time.time()}")
        #lv2.save_TSNE_embedding(filename=filename,random_init=random_init)
        if level==3:
            start_time = time.time()
            lv3 = lv2.create_new_level()
            print(f"LV3:{start_time - time.time()}")
            start_time = time.time()
            lv3.TSNE_to_attribute(random_init=random_init)
            print(f"LV3 TSNE:{start_time - time.time()}")
            lv3.save_all(filename=filename)
        else:
            lv2.save_all(filename=filename)
    else:
        lv1.save_all(filename=filename)


    print(f"finished:{filename}")
    

if __name__ == "__main__":
    main()


