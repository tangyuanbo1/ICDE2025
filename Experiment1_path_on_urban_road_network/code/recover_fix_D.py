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


sys.path.append('..')

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


def recover(raw_data_dir,raw_tras_name,tra_num,big_matrix_real_data,p_noise):

    jsd_res = []
    L1_res = []
    for p in [p_noise]:
    # p=0.03
        temp_jsd_res = []
        temp_L1_res = []
        print("-------------------------------------------------------------p:",p)
        tensor_diff = torch.load(raw_data_dir+'T_tensor'+raw_tras_name+str(tra_num)+'_p'+str(p)+'.pth').cuda().float()
        big_matrix_real_data.shape,tensor_diff.shape
        noise_start=0  
        use_plot=0
        for la_ratio in [0.3]:
            for mu_value1 in [20]:
                T_tensor = tensor_diff
                writer = SummaryWriter('/root/tf-logs')
                import copy
                R_tensor = copy.deepcopy(T_tensor)
                T_tensor = T_tensor.float().cuda()
                R_tensor = R_tensor.float().cuda()
                D_tensor = torch.eye(tensor_diff.shape[0]).float().cuda()
                n_ratio=1
                
                lambda_value=0.5
                D_tensor = torch.load(raw_data_dir+'decomposed/D_tensor_p'+str(p)+'_n'+str(n_ratio)+'la'+str(lambda_value)+'.pth').cuda()
                R_tensor.requires_grad=True
                # D_tensor.requires_grad=True

                relu = torch.nn.ReLU()
                opt  = torch.optim.Adam([R_tensor], lr=0.009*(1/la_ratio))
                epoch_num=3010

                diff_norm_num=1

                recon_data = D_tensor@R_tensor
                temp_tensor = ( recon_data- T_tensor).cuda()
                diff_with_real = round(float(torch.norm(big_matrix_real_data-recon_data,1)),2)
                dict_para,lambda_value,mu_value1,binary_para=la_ratio*0.1,la_ratio*0.1*60,10,0
                a = dict_para*torch.sum(torch.max(R_tensor,1).values).cuda()
                b = lambda_value*(torch.norm(R_tensor,1)/recon_data.shape[1]).cuda()
                c =  mu_value1*(torch.norm((temp_tensor),diff_norm_num))/recon_data.shape[1]
                
                print(diff_with_real,
                                # round(float(torch.norm(big_matrix_real_data-D_tensor@R_tensor,1)),2),
                                round(float(torch.norm(T_tensor-D_tensor@R_tensor,1)),2),
                                'dict_size:',round(a.item()/dict_para,2),'avg_repr_:',round(b.item()/lambda_value,2),'noise',recon_data[noise_start:].sum().item(),'data',recon_data[:noise_start].sum().item())
                
                for i in trange(epoch_num):
                    recon_data = D_tensor@R_tensor
                    # recon_data.data=torch.minimum(recon_data, torch.ones(recon_data.size()).cuda()) 
                    diff_with_real = round(float(torch.norm(big_matrix_real_data-recon_data,1)),2)
                    given_diff_with_real = round(float(torch.norm(T_tensor-recon_data,1)),2)

                    temp_tensor = ( recon_data- T_tensor).cuda()
                    
                    a = dict_para*torch.sum(torch.max(R_tensor,1).values).cuda()
                    b = lambda_value*(torch.norm(R_tensor,1)/recon_data.shape[1]).cuda()
                    c =  mu_value1*(torch.norm((temp_tensor),diff_norm_num))/recon_data.shape[1]
                    binary_loss = binary_para* (torch.abs(0.5 - torch.abs(R_tensor - 0.5)).sum()/recon_data.shape[1] +torch.abs(0.5 - torch.abs(D_tensor - 0.5)).sum()/recon_data.shape[1])#10*(torch.abs(R_tensor-0.5*torch.ones(R_tensor.size()).cuda()).mean())

                    res=a+b+c+binary_loss
                    res.backward()
                    opt.step()
                    mean = 0.0
                    std = 0.001


                    R_tensor.data=torch.maximum(R_tensor, torch.zeros(R_tensor.size()).cuda()) 
                    R_tensor.data=torch.minimum(R_tensor, torch.ones(R_tensor.size()).cuda()) 

                    writer.add_scalar("all_loss",res.cpu().data.numpy().round(1).item(),i) 
                    writer.add_scalar("dict_size",a.cpu().data.numpy().round(1).item(),i) 
                    writer.add_scalar("representation_cost",b.cpu().data.numpy().round(1).item(),i) 
                    writer.add_scalar("reconstruction loss",c.cpu().data.numpy().round(1).item(),i)
                    writer.add_scalar("binary loss",binary_loss.cpu().data.numpy().round(1).item(),i)
                    writer.add_scalar("diff_with_real",diff_with_real,i)
                    
                    R_tensor.grad.zero_()
                    # D_tensor.grad.zero_()
                    if i%300==1:
                        given_diff_with_real = round(float(torch.norm(T_tensor-recon_data,1)),2)
                        L1_norm_real = torch.norm(big_matrix_real_data,1)
                        L1_diff_given_with_real = round(float(torch.norm(T_tensor-big_matrix_real_data,1)),2)
                        L1_diff_recover_with_real = round(float(torch.norm(recon_data-big_matrix_real_data,1)),2)
                        L1_diff_given_with_recover = round(float(torch.norm(recon_data-T_tensor,1)),2)
                        # print("L1:",L1_norm_real,L1_diff_given_with_real,L1_diff_recover_with_real,L1_diff_given_with_recover)
                        print("difference percent between noisy data and real data:",(L1_diff_given_with_real/L1_norm_real).item())
                        print("difference percent between recover data and real data:",(L1_diff_recover_with_real/L1_norm_real).item())
                        # print(diff_with_real,given_diff_with_real,
                        #         # round(float(torch.norm(big_matrix_real_data-D_tensor@R_tensor,1)),2),
                        #         round(float(torch.norm(T_tensor-D_tensor@R_tensor,1)),2),
                        #         'dict_size:',round(a.item()/dict_para,2),'avg_repr_:',round(b.item()/lambda_value,2),'noise',round(recon_data[noise_start:].sum().item()),'data',round(recon_data[:noise_start].sum().item()))

                    # clear_output(wait=True)
                    if use_plot==1:
                        plt.figure(figsize=(15, 5))
                        plt.subplot(1, 3, 1)
                        plt.imshow(D_tensor.detach().cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
                        plt.title('D_tensor 1'+str(i))
                        plt.subplot(1, 3, 2)
                        plt.imshow(R_tensor.detach().cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
                        plt.title('R_tensor 2')
                        plt.subplot(1, 3, 3)
                        plt.imshow(recon_data.detach().cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
                        plt.title('recon_data 3')

                        from matplotlib.ticker import MultipleLocator
                        plt.gca().xaxis.set_major_locator(MultipleLocator(15))
                        # plt.grid()
                        
                        plt.show()
                    if i%1000==1:
                        conditional_prob_matrix_recover_data = calculate_joint_probabilities_torch((D_tensor@R_tensor).detach().T)
                        # big_matrix_real_data = torch.load(raw_data_dir+"T_tensor_31238.pth").float().T.cpu().numpy()
                        conditional_prob_matrix_real_data = calculate_joint_probabilities_torch(big_matrix_real_data.T)
                        conditional_prob_matrix_noisy_data = calculate_joint_probabilities_torch(T_tensor.T)
                        # print('jsd score:',compute_weighted_jsd_score(conditional_prob_matrix_recover_data,conditional_prob_matrix_real_data,big_matrix_real_data))
                        # print('jsd score:',compute_weighted_jsd_score(conditional_prob_matrix_noisy_data,conditional_prob_matrix_real_data,big_matrix_real_data))
                        
                        # print('jsd score between recover data and real data:',compute_weighted_jsd_score(conditional_prob_matrix_recover_data,conditional_prob_matrix_real_data,big_matrix_real_data))
                        # print('jsd score between noisy data and real data:',compute_weighted_jsd_score(conditional_prob_matrix_noisy_data,conditional_prob_matrix_real_data,big_matrix_real_data))

                        # print('L1 norm between recover data and real data:',torch.norm(torch.tensor(conditional_prob_matrix_real_data-conditional_prob_matrix_recover_data),1))
                        # print('L1 norm between noisy data and real data:',torch.norm(torch.tensor(conditional_prob_matrix_real_data-conditional_prob_matrix_noisy_data),1))
                        temp_jsd_res.append(compute_weighted_jsd_score(conditional_prob_matrix_recover_data,conditional_prob_matrix_real_data,big_matrix_real_data))
                        temp_L1_res.append(torch.norm(torch.tensor(conditional_prob_matrix_real_data-conditional_prob_matrix_recover_data),1))
                jsd_res.append(temp_jsd_res)
                L1_res.append(temp_L1_res)
                writer.close()
                    
                n_ratio=1
                tensor_path = raw_data_dir+'/decomposed/D_tensor_p'+str(p)+'_n'+str(n_ratio)+'la'+str(la_ratio)+'.pth'
                torch.save(D_tensor, tensor_path)
                tensor_path = raw_data_dir+'/decomposed/R_tensor_p'+str(p)+'_n'+str(n_ratio)+'la'+str(la_ratio)+'.pth'
                torch.save(R_tensor, tensor_path)
                


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
        print("Dataset_name information is not provided.") 
    
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
        recover(raw_data_dir,raw_tras_name,tra_num,big_matrix_real_data,p_noise)

    else:  
        print("Dataset_name information is not provided.")  

if __name__ == '__main__':  
    main()