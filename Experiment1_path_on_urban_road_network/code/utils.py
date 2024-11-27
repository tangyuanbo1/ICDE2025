
import os
from pathlib import Path
import re
from matplotlib import pyplot as plt  
import json
import numpy as np
import networkx as nx
import math
matrix_size = 5
path_num  = 300


def calculate_joint_probabilities_torch(matrix):
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix)
    
    matrix = matrix.to('cuda')
    
    k = matrix.shape[1]  
    occurrences = torch.sum(matrix, dim=0, dtype=torch.float32)
    # print(occurrences.shape)
    both_occurrences_matrix = torch.matmul(matrix.T.float(), matrix.float())
    conditional_prob_matrix = both_occurrences_matrix/matrix.shape[0] #/ occurrences[:, None]
    # conditional_prob_matrix = both_occurrences_matrix/ occurrences[:, None]
    conditional_prob_matrix[torch.isnan(conditional_prob_matrix)] = 0
    
    conditional_prob_matrix[torch.isinf(conditional_prob_matrix)] = 0
    torch.diagonal(conditional_prob_matrix).fill_(1)

    # big_matrix_real_data.T.shape
    for i in range(matrix.shape[1]):
        if matrix[:,i].mean()<=0.001:
        # if matrix[:,i].sum()==0:
            conditional_prob_matrix[i,i]=0
            # print(i)


    conditional_prob_matrix = conditional_prob_matrix.to('cpu').numpy()
    
    return conditional_prob_matrix



def compute_weighted_jsd_score(m1, m2, real_data):
    vector = real_data.sum(axis=0)
    vector = vector[:, np.newaxis]
    # print(vector.shape)
    p1 = m1 + 1e-5
    p2 = m2 + 1e-5
    
    term1 = p1 * np.log2(2 * p1 / (p1 + p2))
    term2 = p2 * np.log2(2 * p2 / (p1 + p2))

    divergence = np.sum(((term1 + term2) / 2 ))
    edge_space_size = m1.shape[0]
    return divergence/edge_space_size



def compute_diff(matrix,real_data):
    vector = real_data.sum(axis=0)#.shape
    vector = vector[:, np.newaxis]
    abs_sum = np.sum(np.abs(matrix))
    # abs_sum = np.sum(np.abs(matrix))/3661#/vector.sum()
    square_sum = np.sum(np.square(matrix))
    print("Abs Sum:", abs_sum)
    print("Square Sum:", square_sum)
    return [abs_sum,square_sum]




#-----------------------preprocess--------------------------#
import math
def getDistance(latA, lonA, latB, lonB):
    ra = 6378140  
    rb = 6356755  
    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = math.radians(latA)
    radLonA = math.radians(lonA)
    radLatB = math.radians(latB)
    radLonB = math.radians(lonB)
 
    pA = math.atan(rb / ra * math.tan(radLatA))
    pB = math.atan(rb / ra * math.tan(radLatB))
    x = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB) * math.cos(radLonA - radLonB))
    c1 = (math.sin(x) - x) * (math.sin(pA) + math.sin(pB)) ** 2 / math.cos(x / 2) ** 2
    c2 = (math.sin(x) + x) * (math.sin(pA) - math.sin(pB)) ** 2 / math.sin(x / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (x + dr)
    distance = round(distance / 1000, 4)
    return distance,f'{distance}km'

def union_multi_grids(list_of_bounds):
    a = pd.DataFrame(list_of_bounds,columns = ["w",'s','e','n'])
    return [a["w"].min(),a["s"].min(),a["e"].max(),a["n"].max()]




def generate_subset_using_bound(raw_data,bound,start_or_end=0):
    if start_or_end==0:
        res_subset = raw_data[raw_data["slon"]>bound[0]]
        res_subset = res_subset[res_subset["slon"]<bound[2]]
        res_subset = res_subset[res_subset["slat"]>bound[1]]
        res_subset = res_subset[res_subset["slat"]<bound[3]]
    if start_or_end==1:
        res_subset = raw_data[raw_data["elon"]>bound[0]]
        res_subset = res_subset[res_subset["elon"]<bound[2]]
        res_subset = res_subset[res_subset["elat"]>bound[1]]
        res_subset = res_subset[res_subset["elat"]<bound[3]]
    if start_or_end==2:
        res_subset = raw_data[raw_data["slon"]>bound[0]]
        res_subset = res_subset[res_subset["slon"]<bound[2]]
        res_subset = res_subset[res_subset["slat"]>bound[1]]
        res_subset = res_subset[res_subset["slat"]<bound[3]]
        res_subset = res_subset[res_subset["elon"]>bound[0]]
        res_subset = res_subset[res_subset["elon"]<bound[2]]
        res_subset = res_subset[res_subset["elat"]>bound[1]]
        res_subset = res_subset[res_subset["elat"]<bound[3]]
    return res_subset

    
#-----------------------preprocess end--------------------------#


def load_tras(path='generated_trajectory_dataset_highspeedway1.npy'):
    c=np.load(path,allow_pickle=True)
    return c.tolist()
def filter_raw_num_of_edge_6(raw_tras_sz):
    all_tras = []
    tra_index = 0
    tra_index_list = []
    for raw_tra in raw_tras_sz:
        if 1:
            tmp_tra = []
            for edge in raw_tra:
                tmp_tra.append(tuple(edge))
            all_tras.append(tmp_tra)
            tra_index_list.append(tra_index)
        tra_index+=1
    print("len of raw tras after filter edges more than 6:",len(all_tras))
    print("the first trajectory:")
    print(all_tras[0])
    # tras = all_tras[:int(len(all_tras)*0.9)]
    # tras_test = all_tras[int(len(all_tras)*0.9):]
    print("split tras and tras_test")
    tras = all_tras[:]
    tras_test = all_tras[:100]
    print("tras chosen:",len(tras))
    return tras,tra_index_list

def generate_all_p(t,min=3,max=1000):
    res=[]
    for i in range(len(t)):
        for j in range(i+1,len(t)+1):
            if ((j-i)>min) and ((j-i)<max):
                res.append(t[i:j])
    return res

from torch.optim.lr_scheduler import StepLR
import torch
from tqdm import trange

def report_performance(D_tensor,R_tensor,T_tensor):
    temp_tensor = (D_tensor@R_tensor - T_tensor).cuda()
    relu = torch.nn.ReLU()
    a=torch.sum(torch.max(R_tensor,1).values).cpu().data.numpy().round(1).item()
    b=torch.norm(R_tensor,1).cpu().data.numpy().round(1).item()
    c=(torch.norm(R_tensor,1).cpu().data.numpy().round(1).item())/(R_tensor.size()[1])
    d=T_tensor.sum().cpu().data.numpy().round(1).item()
    e=torch.norm(relu(-1*temp_tensor),1).cuda().cpu().data.numpy().round(2).item()
    f=torch.norm(relu(temp_tensor),1).cuda().cpu().data.numpy().round(4).item()
    g=((torch.norm(relu(-1*temp_tensor),1).cuda())/(T_tensor.sum())).cpu().data.numpy().round(4).item()
    h=((torch.norm(relu(temp_tensor),1).cuda())/(T_tensor.sum())).cpu().data.numpy().round(2).item()
    print("dict_size",a) 
    print("representation_cost",b)
    print("avg representation_cost",c) 
    print("num of all edges:",d)
    print("num of uncover edges:",e)
    print("num of overlap edges:",f)
    print("uncover ratio:",g)
    print("overlap ratio:",h)
    return [a,b,c,d,e,f,g,h]
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
# if_subset=1 means we just care about representation cost and reconstruction error
def R_train(initial_R,D_tensor,T_tensor,if_subset,num_of_subset,lambda_value,mu_value1,mu_value2,mu_value3,mu_value4,epoch_num,LR,gamma_value):
    print("-------------------------------------------------lambda_value=",lambda_value)
    writer = SummaryWriter('/root/tf-logs')
    R_tensor = initial_R.float().cuda()
    R_tensor.requires_grad=True
    
    # scheduler_1 = StepLR(opt, step_size=100, gamma=gamma_value)
    diff_norm_num=1
    start_opt_rounding_ratio = 0.3
    relu = torch.nn.ReLU()
    loss_his = [[],[],[],[],[],[],[]]
    real_mu_value1 = torch.ones(T_tensor.size()).cuda()
    real_mu_value3 = torch.ones(1).cuda()
    

    real_mu_value1.requires_grad=True
    real_mu_value3.requires_grad=True
    opt  = torch.optim.Adam([R_tensor,real_mu_value1,real_mu_value3], lr=LR)
    # -------------------------start iteration------------------------------------- #
    for i in trange(epoch_num):
        temp_tensor = (D_tensor@R_tensor - T_tensor).cuda()
         
        a = torch.sum(torch.max(R_tensor,1).values*(1-if_subset)).cuda()
        b = lambda_value*torch.norm(R_tensor,1).cuda()
        
        # mu = 0 invalidate these two terms
        if mu_value1!=0:
            c =  mu_value1*(torch.norm((temp_tensor)*(temp_tensor),diff_norm_num)) 
            # c = (real_mu_value1.mul(temp_tensor)).sum()
        else:
            c = torch.zeros(1).cuda()
        if mu_value2!=0:
            d =  mu_value2*( torch.norm(D_tensor@(R_tensor*(torch.ones(R_tensor.size()).cuda()-R_tensor)),diff_norm_num))
        else:
            d = torch.zeros(1).cuda()        
        # change the constraints
        if mu_value3!=0:
            e = mu_value3*torch.norm(relu(-1*temp_tensor),1).cuda()
        else:
            e = torch.zeros(1).cuda()
        if mu_value4!=0:
            f = mu_value4*torch.norm(relu(1*temp_tensor),1).cuda()
        else:
            f = torch.zeros(1).cuda()   
            
 
        # if i <epoch_num*0.1:
        res = (a+b+c+d+e+f)
        # else:
            # res = (a+b+e+f)
        res.backward()
        # if i<1:
        #     print(i,"a""loss:",res.data)

        opt.step()
        R_tensor.data=torch.maximum(R_tensor, torch.zeros(R_tensor.size()).cuda()) 
        R_tensor.data=torch.minimum(R_tensor, torch.ones(R_tensor.size()).cuda()) 
        
        # R_tensor=torch.clamp(R_tensor,0,1)
        
        # scheduler_1.step()
        R_tensor.grad.zero_()
        loss_his[0].append(res.cpu().data.numpy().round(1))
        loss_his[1].append(a.cpu().data.numpy().round(1))
        loss_his[2].append(torch.norm(R_tensor,1).cpu().data.numpy().round(1) )
        # loss_his[3].append(c.item())
        # loss_his[4].append(d.item())
        loss_his[5].append(e.cpu().data.numpy().round(1))
        loss_his[6].append(f.cpu().data.numpy().round(1))
        
        writer.add_scalar("all_loss",res.cpu().data.numpy().round(1).item(),i)  
        writer.add_scalar("dict_size",a.cpu().data.numpy().round(1).item(),i)  
        writer.add_scalar("representation_cost",torch.norm(R_tensor,1).cpu().data.numpy().round(1) .item(),i)  
        writer.add_scalar("c",c.cpu().data.numpy().round(1).item(),i)  
        writer.add_scalar("d",d.cpu().data.numpy().round(1).item(),i)  
        writer.add_scalar("e",e.cpu().data.numpy().round(1).item(),i)  
        writer.add_scalar("f",f.cpu().data.numpy().round(1).item(),i)  
        writer.add_scalar("real_mu_value1",real_mu_value1.sum().cpu().data.numpy().round(1).item(),i)  
        writer.add_scalar("real_mu_value3",real_mu_value3.cpu().data.numpy().round(1).item(),i)  
        writer.add_scalar("num of uncover edges:",torch.norm(relu(-1*temp_tensor),1).cuda().cpu().data.numpy().round(2).item(),i)
        writer.add_scalar("num of overlap edges:",torch.norm(relu(temp_tensor),1).cuda().cpu().data.numpy().round(2).item(),i)
    

    writer.close()
    return loss_his,R_tensor,real_mu_value1

def R_train_nn(initial_R,T_tensor,lambda_value,epoch_num,LR,mu,size_of_hidden,dimension_of_d):
    print("-------------------------------------------------lambda_value=",lambda_value)
    writer = SummaryWriter('/root/tf-logs')
    
    

    # MLP_matrix_1 = 0.01*torch.rand( T_tensor.size()[1],size_of_hidden).cuda()
    # MLP_matrix_2 = 0.01*torch.rand( size_of_hidden,dimension_of_d).cuda()

    MLP_matrix_1 = torch.eye( size_of_hidden).cuda()
    MLP_matrix_2 = torch.eye( size_of_hidden).cuda()

    R_tensor =  0.01*torch.rand( dimension_of_d,T_tensor.size()[1]).cuda()

    R_tensor.requires_grad=True
    MLP_matrix_1.requires_grad=True
    MLP_matrix_2.requires_grad=True

    
    diff_norm_num=1
    # scheduler_1 = StepLR(opt, step_size=100, gamma=gamma_value)
    # start_opt_rounding_ratio = 0.3
    relu = torch.nn.ReLU()
    loss_his = [[],[],[],[],[],[],[]]

    opt  = torch.optim.Adam([R_tensor,MLP_matrix_1,MLP_matrix_2], lr=LR)
    # opt  = torch.optim.Adam([R_tensor], lr=LR)
    # -------------------------start iteration------------------------------------- #
    for i in trange(epoch_num):
        temp_tensor1 = relu(T_tensor@MLP_matrix_1).cuda()
        D_tensor = (temp_tensor1@MLP_matrix_2).cuda()
        diff_tensor = (D_tensor@R_tensor - T_tensor).cuda()
         
        a = torch.sum(torch.max(R_tensor,1).values).cuda()
        b = lambda_value*torch.norm(R_tensor,1).cuda()
        c =  mu*(torch.norm((diff_tensor)*(diff_tensor),diff_norm_num)) 
        # res = (a+b+c)
        res=a+b+c
        res.backward()
        opt.step()
        R_tensor.data=torch.maximum(R_tensor, torch.zeros(R_tensor.size()).cuda()) 
        # R_tensor.data=torch.minimum(R_tensor, torch.ones(R_tensor.size()).cuda()) 

        R_tensor.grad.zero_()
        MLP_matrix_1.grad.zero_()
        MLP_matrix_2.grad.zero_()

        loss_his[0].append(res.cpu().data.numpy().round(1))
        loss_his[1].append(a.cpu().data.numpy().round(1))
        loss_his[2].append(b.cpu().data.numpy().round(1) )
        loss_his[3].append(c.cpu().data.numpy().round(1) )
        loss_his[4].append(MLP_matrix_1.sum().cpu().data.numpy().round(1) )
        loss_his[5].append(MLP_matrix_2.sum().cpu().data.numpy().round(1) )

        
        writer.add_scalar("all_loss",res.cpu().data.numpy().round(1).item(),i) 
        writer.add_scalar("dict_size",a.cpu().data.numpy().round(1).item(),i) 
        writer.add_scalar("representation_cost",b.cpu().data.numpy().round(1) .item(),i) 
        writer.add_scalar("c",c.cpu().data.numpy().round(1).item(),i) 
        writer.add_scalar("num of uncover edges:",torch.norm(relu(-1*diff_tensor),1).cuda().cpu().data.numpy().round(2).item(),i)
        writer.add_scalar("num of overlap edges:",torch.norm(relu(diff_tensor),1).cuda().cpu().data.numpy().round(2).item(),i)
        writer.add_scalar("MLP_matrix_1:",MLP_matrix_1.sum().cpu().data.numpy().round(1),i)
        writer.add_scalar("MLP_matrix_2:",MLP_matrix_2.sum().cpu().data.numpy().round(1),i)
        writer.add_scalar("D:",D_tensor.sum().cpu().data.numpy().round(1),i)
    

    writer.close()
    return loss_his,R_tensor,D_tensor

#-----------------------transform--------------------------#
# tra edge pathlet 
# str
def transform_pathlet_2_continueedges(path):
    res = []
    # print(path)
    for edge in path:
        res.append(edge[0])
    res.append(edge[1])
    return tuple(res)
def transform_tra_2_continue_paths(tra,unorder_path_list):
    res = []
    temp_path_list=unorder_path_list
    current_edge_index = 0
    while current_edge_index<len(tra):
        current_edge = str(tra[current_edge_index])
        for path in temp_path_list:
            if current_edge in path:
                res.append(transform_pathlet_2_continueedges(strpath2rawpath(path)))
                temp_path_list.remove(path)
                break
        current_edge_index+=1
                
    return res

def strpath2rawpath(strpath):
    stredge_list = path_str2list(strpath)
    res = []
    for str_edge in stredge_list:
        res.append(stredge2edge(str_edge))
    return res 
def stredge2edge(stredge):
    str_edge = stredge.replace("(","")
    str_edge = str_edge.replace(")","")
    str_edge = str_edge.replace("[","")
    str_edge = str_edge.replace("]","")
    start_point = int(str_edge.split(",")[0])
    end_point = int(str_edge.split(",")[1])
    return (start_point,end_point)

def path_str2list(temp_str):
    res = []
    find_loc = [temp_str.find("("),temp_str.find(")")]
    while (find_loc[0]!=-1 and find_loc[1]!=-1 ):
        # print(temp_str)
        edge_str = temp_str[find_loc[0]:find_loc[1]+1]
        # print(edge_str.find(")"))
        # print(len(edge_str))
        # if edge_str[-1]!=")":
        #     edge_str=edge_str[:-1]
        res.append(edge_str)
        temp_str=temp_str[find_loc[1]+1:]
        
        find_loc = [temp_str.find("("),temp_str.find(")")]
    return res
# path_str2list(path)
# transform_pathlet_2_continueedges(path_list[0])
# (1169591930, 8192576285, 1169591930)
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
def plot_edge(edges_p,source_path):
    pathdf = pd.DataFrame(source_path,columns = ['u'])
    pathdf['v'] = pathdf['u'].shift(-1)
    pathdf = pathdf[-pathdf['v'].isnull()]
    pathgdf = pd.merge(pathdf,edges_p.reset_index())
    pathgdf = gpd.GeoDataFrame(pathgdf)
    # pathgdf.plot()
    pathgdf.crs = {'init':'epsg:2416'}
    pathgdf_4326_ = pathgdf.to_crs(4326)
    return pathgdf_4326_




# ----------------------------DP----------------------------------------

def plot_tra(tra,matrix_size):
    plt.figure(figsize=(15, 5))
    ax = plt.gca()
    ax.margins(0.11)
    G = nx.grid_2d_graph(matrix_size*3, matrix_size)
    pos = dict( (n, n) for n in G.nodes() )
    nx.draw_networkx_nodes(G, pos, node_size=50, alpha=0.3, node_color="tab:orange")
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=4, edge_color="tab:orange")
    startpoint_of_highspeedway=(matrix_size-1,matrix_size//2)
    endpoint_of_highspeedway=(matrix_size*2,matrix_size//2)
    for i in range(matrix_size):
        for j in range(matrix_size):
            nx.draw_networkx_nodes(G, pos, node_size=30, alpha=0.6, nodelist=[(i,j)], node_color="tab:blue")
            nx.draw_networkx_nodes(G, pos, node_size=30, alpha=0.6, nodelist=[(i+matrix_size*2,j)], node_color="tab:green")
    for i in range(len(tra)-1):
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=4, edgelist=[(tra[i],tra[i+1])],edge_color="tab:blue")
        nx.draw_networkx_nodes(G, pos, node_size=100, alpha=0.5, nodelist=[tra[i]], node_color="tab:blue")
    nx.draw_networkx_nodes(G, pos, node_size=100, alpha=0.8, nodelist=[tra[-1]], node_color="tab:green")

def plot_high_speed_way(matrix_size,matrix_num=5,ax_before=None):
    if ax_before==None:
        plt.figure(figsize=(25, 5))
        ax = plt.gca()
        ax.margins(0.11)
    else:
        ax = ax_before
    startpoint_of_highspeedway=[(matrix_size-1,matrix_size//2),(3*matrix_size-1,matrix_size//2)]
    endpoint_of_highspeedway=[(matrix_size*2,matrix_size//2),(matrix_size*4,matrix_size//2)]
    G = nx.grid_2d_graph(matrix_size*matrix_num, matrix_size)
    pos = dict( (n, n) for n in G.nodes() )
    # nx.draw_networkx_nodes(G, pos, node_size=50, alpha=0.9, nodelist=[1], node_color="tab:red")
    # nx.draw_networkx_nodes(G, pos, node_size=50, alpha=0.9, nodelist=[2], node_color="tab:red")
    # nx.draw_networkx_edges(G, pos, alpha=0.5, width=8, edgelist=[(1,1)],edge_color="tab:purple") 
    nx.draw_networkx_nodes(G, pos, node_size=50, alpha=0.9, nodelist=[startpoint_of_highspeedway[0]], node_color="tab:red")
    nx.draw_networkx_nodes(G, pos, node_size=50, alpha=0.9, nodelist=[endpoint_of_highspeedway[0]], node_color="tab:red")
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=8, edgelist=[(startpoint_of_highspeedway[0],endpoint_of_highspeedway[0])],edge_color="tab:purple")
    nx.draw_networkx_nodes(G, pos, node_size=50, alpha=0.9, nodelist=[startpoint_of_highspeedway[1]], node_color="tab:red")
    nx.draw_networkx_nodes(G, pos, node_size=50, alpha=0.9, nodelist=[endpoint_of_highspeedway[1]], node_color="tab:red")
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=8, edgelist=[(startpoint_of_highspeedway[1],endpoint_of_highspeedway[1])],edge_color="tab:purple")
        
def plot_tra_new(tra,matrix_size,matrix_num=5,ax_before=None):
    if ax_before==None:
        plt.figure(figsize=(25, 5))
        ax = plt.gca()
        ax.margins(0.11)
    else:
        ax = ax_before
    G = nx.grid_2d_graph(matrix_size*matrix_num, matrix_size)
    pos = dict( (n, n) for n in G.nodes() )
    nx.draw_networkx_nodes(G, pos, node_size=50, alpha=0.3, node_color="tab:orange")
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=4, edge_color="tab:orange")
    startpoint_of_highspeedway=(matrix_size-1,matrix_size//2)
    endpoint_of_highspeedway=(matrix_size*2,matrix_size//2)
    for i in range(matrix_size):
        for j in range(matrix_size):
            nx.draw_networkx_nodes(G, pos, node_size=30, alpha=0.6, nodelist=[(i,j)], node_color="tab:blue")
            nx.draw_networkx_nodes(G, pos, node_size=30, alpha=0.6, nodelist=[(i+matrix_size*2,j)], node_color="tab:green")
            nx.draw_networkx_nodes(G, pos, node_size=30, alpha=0.6, nodelist=[(i+matrix_size*4,j)], node_color="tab:green")

    for i in range(len(tra)-1):
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=4, edgelist=[(tra[i],tra[i+1])],edge_color="tab:blue")
        nx.draw_networkx_nodes(G, pos, node_size=100, alpha=0.5, nodelist=[tra[i]], node_color="tab:blue")
    nx.draw_networkx_nodes(G, pos, node_size=100, alpha=0.8, nodelist=[tra[-1]], node_color="tab:green")
    return ax



def transform_tra(raw_tra,matrix_size):
    tra_step1 = []
    tra_step2 = []
    for point in raw_tra:
        tra_step1.append(point[0]*matrix_size*5+point[1])
    for i in range(len(tra_step1)-1):
    # for point in tra_step1:
        if (tra_step1[i]!=tra_step1[i+1]):
            tra_step2.append((tra_step1[i],tra_step1[i+1]))
    return tra_step2

def compute_loss(D,D_optimal,var_lambda):
    loss = len(D)
    for tra_key in D_optimal.keys():
        loss += var_lambda*len(D_optimal[tra_key])
    return loss
def compute_represent_loss(D_optimal):
    loss = 0
    for tra_key in D_optimal.keys():
        loss += len(D_optimal[tra_key])
    return loss

def count_optimal2(D_optimal,path1,path2):
    count = 0
    for tra_key in D_optimal.keys():
        if path1 in D_optimal[tra_key] and path2 in D_optimal[tra_key]:
            count+=1
    return count

def count_optimal(D_optimal,path1,path2):
    count = [0,0,0]
    for tra_key in D_optimal.keys():
        if path1 in D_optimal[tra_key] and path2 in D_optimal[tra_key]:
            count[0]+=1
        if path1 in D_optimal[tra_key] and path2 not in D_optimal[tra_key]:
            count[1]+=1
        if path1 not in D_optimal[tra_key] and path2 in D_optimal[tra_key]:
            count[2]+=1
    return count

def compute_g(D_optimal,path1,path2,var_lambda):
    count = count_optimal(D_optimal,path1,path2)
    if count[0]==0:
        return [-100,-100,-100,-100]
    res = [0,0,0,0]
    res[0] = count[0]*var_lambda-1
    for i in range(1,3):
        if count[i]!=0:
            res[i] = -100
        else:
            res[i] = count[0]*var_lambda
    if count[1]!=0 or count[2]!=0:
        res[3] = -100
    else:
        res[3] = count[0]*var_lambda+1
    return res
