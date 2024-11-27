import argparse  
import torch
import sys
sys.path.append('..')
from utils import *
from train import *
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
import json
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
import torch
from tensorboardX import SummaryWriter
from tqdm import trange

from IPython.display import clear_output
import matplotlib.pyplot as plt


import torch
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision import transforms




def main():  
    parser = argparse.ArgumentParser(description="description")  
    
    # Define arguments  
    parser.add_argument('-n', '--Dataset_name', type=str, help='Dataset_name', required=True)  
    parser.add_argument('-p', '--p_noise', type=str, help='p_noise', required=True)  
    
    # Parse arguments  
    args = parser.parse_args()  
    
    # Use arguments  
    if args.p_noise:  
        p_noise = args.p_noise
        print("p_noise:", p_noise)  
    else:  
        print("p_noise information is not provided.") 
    
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
        p=0
        big_matrix_real_data = torch.load(raw_data_dir+'T_tensor'+raw_tras_name+str(tra_num)+'_p'+str(p)+'.pth').float().cuda()#.cpu().numpy()
        print("--- loading dataset finished ---")
        print("--- start training ---")
        train(raw_data_dir,raw_tras_name,tra_num,big_matrix_real_data,p_noise)

    else:  
        print("Dataset_name information is not provided.")  

if __name__ == '__main__':  
    main()