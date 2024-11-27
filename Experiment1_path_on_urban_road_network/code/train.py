import argparse  
import torch
import sys
sys.path.append('..')
from utils import *
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
from tqdm import *

import torch
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision import transforms

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import special

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import special
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from tqdm import *

class NegativeBinomialVAE(nn.Module):
    def __init__(self, arch, lr=1e-3, random_seed=None):
        super(NegativeBinomialVAE, self).__init__()

        self.decoder_arch = arch[::-1]
        # print(self.decoder_arch)
        self.decoder_arch[0]=int(self.decoder_arch[0]/2)
        self.encoder_arch = arch
        self.lr = lr
        self.random_seed = random_seed

        self.encoder_weights, self.encoder_biases = self._construct_weights(self.encoder_arch)
        self.decoder_weights_r, self.decoder_biases_r = self._construct_weights(self.decoder_arch)
        self.decoder_weights_p, self.decoder_biases_p = self._construct_weights(self.decoder_arch)

    def _construct_weights(self, architecture):
        weight_list = []
        bias_list = []
        for i, (d_in, d_out) in enumerate(zip(architecture[:-1], architecture[1:])):
            # print(d_in, d_out)
            weight = nn.Parameter(torch.Tensor(d_in, d_out).float())
            bias = nn.Parameter(torch.Tensor(d_out).float())
            nn.init.xavier_uniform_(weight)
            nn.init.normal_(bias, std=0.001)
            weight_list.append(weight)
            bias_list.append(bias)
        return nn.ParameterList(weight_list), nn.ParameterList(bias_list)

    def _encoder_pass(self, input_ph, keep_prob_ph):
        mu_z, std_z, kl = None, None, None
        h = nn.functional.normalize(input_ph, p=2, dim=1)
        h = nn.functional.dropout(h, p=1-keep_prob_ph, training=True)
        for i, (w, b) in enumerate(zip(self.encoder_weights, self.encoder_biases)):
            h = torch.matmul(h.float(), w.float()) + b.float()
            if i != len(self.encoder_weights) - 1:
                h = torch.tanh(h)
            else:
                # print(h.shape)
                mu_z = h[:, :self.encoder_arch[-1]//2]
                logvar_q = h[:, self.encoder_arch[-1]//2:]
                std_z = torch.exp(0.5 * logvar_q)
                kl = torch.sum(0.5 * (-logvar_q + torch.exp(logvar_q) + mu_z ** 2 - 1), dim=1)
        return mu_z, std_z, kl

    def _decoder_pass_r(self, z):
        h_r = z
        for i, (w, b) in enumerate(zip(self.decoder_weights_r, self.decoder_biases_r)):
            h_r = torch.matmul(h_r.float(), w.float()) + b.float()
            if i != len(self.decoder_weights_r) - 1:
                h_r = torch.tanh(h_r)
        return h_r

    def _decoder_pass_p(self, z):
        h_p = z
        for i, (w, b) in enumerate(zip(self.decoder_weights_p, self.decoder_biases_p)):
            h_p = torch.matmul(h_p.float(), w.float()) + b.float()
            if i != len(self.decoder_weights_p) - 1:
                h_p = torch.tanh(h_p)
        return h_p

    def forward(self, input_ph, keep_prob_ph, is_training_ph, anneal_ph):
        mu_z, std_z, kl = self._encoder_pass(input_ph, keep_prob_ph)
        epsilon = torch.randn_like(std_z)
        z = mu_z + is_training_ph * epsilon * std_z
        h_r = self._decoder_pass_r(z)
        h_p = self._decoder_pass_p(z)
        return h_r, h_p, kl

    def negative_elbo_loss(self, h_r, h_p, input_ph, anneal_ph,kl):
        ll = self._log_likelihood(h_r, h_p, input_ph)
        neg_ll = -torch.mean(torch.sum(ll, dim=-1))
        neg_elbo = neg_ll + anneal_ph * torch.mean(kl)
        return neg_elbo

    def _log_likelihood(self, h_r, h_p, input_ph):
        temp = torch.exp(-torch.mul(torch.exp(h_r), torch.log(torch.exp(h_p) + 1)))
        temp = torch.clamp(temp, 1e-5, 1 - 1e-5)
        ll = torch.mul(input_ph, torch.log(1 - temp)) + torch.mul(1 - input_ph, torch.log(temp))
        return ll

    def get_predictive_rate(self, h_r, h_p, test_data):
        l_prime = 1 - np.power(special.expit(-h_p), np.exp(h_r))
        return l_prime



def train_binary_variational_auto_encoder(raw_data_dir,raw_tras_name,tra_num,big_matrix_real_data,p_noise):

    for n_ratio in [1]:
        for p_noise in [p_noise]:

            best_loss = None
            patience_counter = 0
            T_tensor = torch.load('/root/code_data/datasets/Vectorized_data/decomposed/R_tensor_p'+str(p_noise)+'_n1la0.5.pth').detach(). cuda().T.float()

            data_tensor =T_tensor
            data_tensor.shape
            data_tensor = data_tensor.cuda()#torch.load(pth_file_path)
            dataset = TensorDataset(data_tensor)
            data_loader = DataLoader(dataset, batch_size=4096, shuffle=True)
            if torch.cuda.is_available():
                device = torch.device("cuda")  
                print("CUDA is available. Model will run on GPU.")
            else:
                device = torch.device("cpu")  
                print("CUDA is not available. Model will run on CPU.")

            arch = [data_tensor.shape[1], 512  , 512,256,256]
            model = NegativeBinomialVAE(arch)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=model.lr)
            best_loss = None
            patience_counter = 0
            
            epochs = 10000
            patience = 2500  
            min_delta = 0.01  
            for epoch in trange(epochs):
                model.train()
                total_loss = 0
                for batch_idx, (data,) in enumerate(data_loader):
                    input_ph = Variable(data)
                    keep_prob_ph = 0.9  
                    is_training_ph = 1  
                    anneal_ph = 1.0  

                    optimizer.zero_grad()
                    h_r, h_p, kl = model(input_ph, keep_prob_ph, is_training_ph, anneal_ph)
                    loss = model.negative_elbo_loss(h_r, h_p, input_ph, anneal_ph,kl)
    
                    l1_lambda=0.0001
                    l1_regularization = torch.tensor(0.).cuda()
                    for param in model.parameters():
                        l1_regularization += torch.norm(param, 1)
                    loss += l1_regularization * l1_lambda

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                if epoch%1000==0:
                    print(f'Epoch: {epoch+1}, Loss: {total_loss/len(data_loader)}')
                if best_loss is None or total_loss < best_loss - min_delta:
                    best_loss = total_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered. Stopping training.")
                    break



            from scipy.stats import nbinom
            import numpy as np


            dimension = 128
            sample_size = 6000  
            
            binary_res_list = []
            for i in trange(sample_size):
                samples = np.random.normal(0, 1, (1, dimension))
                # print(samples)
                samples_tensor = torch.tensor(samples, dtype=torch.float32).cuda()
                sample_p = special.expit(-1*model._decoder_pass_p(samples_tensor).cpu() .detach().numpy())#, np.exp(h_r)
                sample_r =  np.exp(model._decoder_pass_r(samples_tensor).cpu() .detach().numpy())
                r = sample_r#.detach().numpy()  
                p = sample_p#.detach().numpy()

                samples = []

                for pi, ri in zip(p, r):
                    # print(pi,ri)
                    sample = nbinom.rvs(ri, pi)
                    samples.append(sample)

                binary_res = np.array([1 if x >= 1 else 0 for x in sample])
                # print(binary_res.sum())
                binary_res_list.append(binary_res)
            big_matrix = np.vstack(binary_res_list)

            print(big_matrix.shape)
            print(big_matrix.sum()/sample_size)

            D_tensor = torch.load('/root/code_data/datasets/Vectorized_data/decomposed/D_tensor_p'+str(p_noise)+'_n1la0.5.pth').cpu()
            generated_data = (D_tensor@(torch.tensor(big_matrix).float().T)).detach().T
            # generated_data.shape

            conditional_prob_matrix_recover_data = calculate_joint_probabilities_torch(generated_data )
            conditional_prob_matrix_real_data = calculate_joint_probabilities_torch(big_matrix_real_data.cpu().detach().numpy().T)
            # conditional_prob_matrix_noisy_data = calculate_joint_probabilities_torch(T_tensor.T)
            print('jsd score between recover data and real data:',compute_weighted_jsd_score(conditional_prob_matrix_recover_data,conditional_prob_matrix_real_data,big_matrix_real_data))


def train(raw_data_dir,raw_tras_name,tra_num,big_matrix_real_data,p_noise):
    jsd_res = []
    L1_res = []
    diff_given = []
    dict_size = []
    D_list = []
    R_list = []

    for p in [p_noise]:
        print("-------------------------------------------------------------p:",p)
        tensor_diff = torch.load(raw_data_dir+'T_tensor'+raw_tras_name+str(tra_num)+'_p'+str(p)+'.pth').cuda().float()
    
        noise_start=0  
        use_plot=0
        for la_ratio in [0.5]:
            for mu_value1 in [20]:
                temp_jsd_res = []
                temp_L1_res = []
                temp_diff_given = []
                temp_dict_size = []
                T_tensor = tensor_diff
                writer = SummaryWriter('/root/tf-logs')
                import copy
                R_tensor = copy.deepcopy(T_tensor)
                T_tensor = T_tensor.float().cuda()
                R_tensor = R_tensor.float().cuda()
                D_tensor = torch.eye(tensor_diff.shape[0]).float().cuda()
                R_zeros = torch.zeros(R_tensor.size()).cuda()
                R_ones = torch.ones(R_tensor.size()).cuda()
                D_zeros = torch.zeros(D_tensor.size()).cuda()
                D_ones = torch.ones(D_tensor.size()).cuda()
                R_tensor.requires_grad=True
                D_tensor.requires_grad=True

                relu = torch.nn.ReLU()
                opt  = torch.optim.Adam([R_tensor,D_tensor], lr=0.009*(1/la_ratio))
                epoch_num=3000

                diff_norm_num=1

                recon_data = D_tensor@R_tensor
                temp_tensor = ( recon_data- T_tensor).cuda()
                diff_with_real = round(float(torch.norm(big_matrix_real_data-recon_data,1)),2)
                dict_para,lambda_value,mu_value1,binary_para=la_ratio*0.1*0.2,la_ratio*0.1*240,10,0
                a = dict_para*torch.sum(torch.max(R_tensor,1).values).cuda()
                b = lambda_value*(torch.norm(R_tensor,1)/recon_data.shape[1]).cuda()
                c =  mu_value1*(torch.norm((temp_tensor),diff_norm_num))/recon_data.shape[1]
            
                print('diff between given and recon data:',torch.norm((big_matrix_real_data-D_tensor@R_tensor),1),
                            # round(float(torch.norm(big_matrix_real_data-D_tensor@R_tensor,1)),2),
                            round(float(torch.norm(T_tensor-D_tensor@R_tensor,1)),2),
                            'dict_size:',round(a.item()/dict_para,2),'avg_repr_:',round(b.item()/lambda_value,2),'noise',recon_data[noise_start:].sum().item(),'data',recon_data[:noise_start].sum().item())
            
                for i in trange(epoch_num):
                    recon_data = D_tensor@R_tensor
                    diff_with_real = round(float(torch.norm(big_matrix_real_data-recon_data,1)),2)
                    diff_with_given = round(float(torch.norm(T_tensor-recon_data,1)),2)

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
                
                    R_tensor.data=torch.maximum(R_tensor, R_zeros) 
                    R_tensor.data=torch.minimum(R_tensor, R_ones) 
                    D_tensor.data=torch.maximum(D_tensor, D_zeros) 
                    D_tensor.data=torch.minimum(D_tensor, D_ones) 

                    writer.add_scalar("all_loss",res.cpu().data.numpy().round(1).item(),i) 
                    writer.add_scalar("dict_size",a.cpu().data.numpy().round(1).item(),i) 
                    writer.add_scalar("representation_cost",b.cpu().data.numpy().round(1).item(),i) 
                    writer.add_scalar("reconstruction loss",c.cpu().data.numpy().round(1).item(),i)
                    writer.add_scalar("binary loss",binary_loss.cpu().data.numpy().round(1).item(),i)
                    writer.add_scalar("diff_with_real",diff_with_real,i)
                    # writer.add_scalar("uncover edges",e.cpu().data.numpy().round(1).item(),i)
                    # writer.add_scalar("overlap edges",f.cpu().data.numpy().round(1).item(),i)
                    R_tensor.grad.zero_()
                    D_tensor.grad.zero_()
                    if i%300==1:
                        print(   round(float(torch.norm(T_tensor-D_tensor@R_tensor,1)),2),
                            'dict_size:',round(a.item()/dict_para,2),'avg_repr_:',round(b.item()/lambda_value,2),'noise',round(recon_data[noise_start:].sum().item()),'data',round(recon_data[:noise_start].sum().item()))

                    # clear_output(wait=True)
                    if use_plot==1:
                        plt.figure(figsize=(15, 5))
                        plt.subplot(1, 3, 1)
                        plt.imshow(D_tensor.detach().cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
                        plt.title('D_tensor 1'+str(i))
                        # plt.colorbar()  
                        plt.subplot(1, 3, 2)
                        plt.imshow(R_tensor.detach().cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
                        plt.title('R_tensor 2')
                        # plt.colorbar()  
                        plt.subplot(1, 3, 3)
                        plt.imshow(recon_data.detach().cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
                        plt.title('recon_data 3')

                        from matplotlib.ticker import MultipleLocator
                        plt.gca().xaxis.set_major_locator(MultipleLocator(15))
                        # plt.grid()
                        plt.show()
                    if i%300==1:
                        print('diff between real and recon data:',torch.norm((big_matrix_real_data-D_tensor@R_tensor),1))
                        conditional_prob_matrix_recover_data = calculate_joint_probabilities_torch((D_tensor@R_tensor).detach().T)
                        conditional_prob_matrix_real_data = calculate_joint_probabilities_torch(big_matrix_real_data.T)
                        conditional_prob_matrix_noisy_data = calculate_joint_probabilities_torch(T_tensor.T)
                        print('jsd score between recover data and real data:',compute_weighted_jsd_score(conditional_prob_matrix_recover_data,conditional_prob_matrix_real_data,big_matrix_real_data))
                        print('jsd score between noisy data and real data:',compute_weighted_jsd_score(conditional_prob_matrix_noisy_data,conditional_prob_matrix_real_data,big_matrix_real_data))
                        print('L1 norm between recover data and real data:',torch.norm(torch.tensor(conditional_prob_matrix_real_data-conditional_prob_matrix_recover_data),1))
                        print('L1 norm between noisy data and real data:',torch.norm(torch.tensor(conditional_prob_matrix_real_data-conditional_prob_matrix_noisy_data),1))
                        temp_jsd_res.append(compute_weighted_jsd_score(conditional_prob_matrix_recover_data,conditional_prob_matrix_real_data,big_matrix_real_data))
                        temp_L1_res.append(torch.norm(torch.tensor(conditional_prob_matrix_real_data-conditional_prob_matrix_recover_data),1))
                        temp_diff_given.append(diff_with_given)
                        temp_dict_size.append(round(a.item()/dict_para,2))
                        D_list.append(D_tensor.data.cpu().numpy())
                        R_list.append(R_tensor.data.cpu().numpy())
                jsd_res.append(temp_jsd_res)
                L1_res.append(temp_L1_res)
                diff_given.append(temp_diff_given)
                dict_size.append(temp_dict_size)
                writer.close()
                
                n_ratio=1
                tensor_path = raw_data_dir+'/decomposed/D_tensor_p'+str(p)+'_n'+str(n_ratio)+'la'+str(la_ratio)+'.pth'
                torch.save(D_tensor, tensor_path)
                tensor_path = raw_data_dir+'/decomposed/R_tensor_p'+str(p)+'_n'+str(n_ratio)+'la'+str(la_ratio)+'.pth'
                torch.save(R_tensor, tensor_path)

    for i in range(len(D_list)):
        np.save(raw_data_dir+'decomposed/D_list'+raw_tras_name+str(i)+'.npy',D_list[i])
    train_binary_variational_auto_encoder(raw_data_dir,raw_tras_name,tra_num,big_matrix_real_data,p_noise)
