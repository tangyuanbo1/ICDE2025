import numpy as np
import sys
sys.path.append('..')

import os
import re
import json
import numpy as np
from utils import *
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import dill
import matplotlib.pyplot as plt
from tqdm import tqdm
# print(torch.__version__)
# print(torch.version.cuda)
# use_gpu = torch.cuda.is_available()
# use_gpu
sns.set_theme(style="darkgrid")
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import argparse  


def main():  
    parser = argparse.ArgumentParser(description="description")  
    
    # Define arguments  
    parser.add_argument('-n', '--Dataset_name', type=str, help='Dataset_name', required=True)  
    
    # Parse arguments  
    args = parser.parse_args()  
    
    if args.Dataset_name:  
        Dataset_name = args.Dataset_name
        
        if Dataset_name == 'futian':
            raw_tras_name = "tra_datasetfutian5_5_"
            tra_num = 3892
        elif Dataset_name == 'porto':
            raw_tras_name = "tra_datasetporto5_5_"
            tra_num = 3100
        print("Dataset_name:", Dataset_name)
        print("--- start loading dataset ---")
        raw_data_dir = '../datasets/Vectorized_data/'
        save_dir = '../datasets/'
 
        print("--- Loading dataset finished ---")
        print("--- Start preprocessing ---")



        num_of_tra_4_train = 100000

        edge_list = []

        raw_tras_name_list=[]
        for p in  [0,0.02,0.04,0.06,0.08,0.10]:
        # for p in  [0]:
            raw_tras_name_list.append(raw_tras_name+str(tra_num)+'_p'+str(p))
        for raw_tras_name in raw_tras_name_list :
            raw_tras_sz = load_tras(save_dir+'raw_data_under_noise/'+raw_tras_name+".npy")
            print("len of raw tras when loading:",len(raw_tras_sz))
            tras = raw_tras_sz
            tras = tras[:num_of_tra_4_train] 
            print("len of raw tras after filter:",len(tras))
            
            for i in trange(len(tras)):
                tra = tras[i]
                for edge in tra:
                    if edge not in edge_list:
                        edge_list.append(edge)  


            l,n = len(edge_list),len(tras)
            print("len(edge_list):",l,"len(tras):",n)
        import pickle
        import pickle
        with open(save_dir+'edge_list'+raw_tras_name+'.pkl', 'wb') as f:
            pickle.dump(edge_list, f)
        cnt=0
        tensor_list = []
        for raw_tras_name in raw_tras_name_list:

            raw_tras_sz = load_tras(save_dir+'raw_data_under_noise/'+raw_tras_name+".npy")
            print("len of raw tras when loading:",len(raw_tras_sz))
            tras = raw_tras_sz
            # tras,tra_index_list = filter_raw_num_of_edge_6(raw_tras_sz)
            tras = tras[:num_of_tra_4_train] 
            T = np.zeros((l, n)) 
            T_half = np.zeros((l, n))
            for i in trange(len(tras)):
                tra = tras[i]
                for edge in tra:
                    T[edge_list.index(edge)][i] = 1
                    cnt+=1
                for edge in tra[:int(len(tra)/2)]:
                    T_half[edge_list.index(edge)][i] = 1

            T_tensor=torch.from_numpy(T[:len(edge_list)])#.cuda()
            tensor_path = save_dir+'Vectorized_data/T_tensor'+raw_tras_name+'.pth'
            torch.save(T_tensor, tensor_path)
            tensor_list.append(T_tensor)
            print(T_tensor.shape,T_tensor.sum())
        print("--- preprocessing done ---")
            
    else:  
        print("Dataset_name information is not provided.")  

if __name__ == '__main__':  
    main()




