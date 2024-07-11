# %%
# import os
# os.getcwd()
# os.chdir('/home/chtung/project_MC')

import numpy as np
import time
from tqdm import tqdm, trange
from WLM import WLChain
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat

# %%
def eval_sq_SHE(kappa, epsilon, chain, qq, rr, lm=[(0,0),(2,0),(2,2),(2,-2)], n_sample = 20, n_merge = 4, rayleigh=False, calculate_g_r=False, real=True):
    ## generate spectra of three different grids
    S_q_lm_list = []
    g_r_lm_list = []
    for i, grid in enumerate(['SC','RB','RT']):
        print(grid)
        S_q_lm_i = np.zeros([len(qq),len(lm)])
        g_r_lm_i = np.zeros([len(rr),len(lm)])
        for j in trange(n_sample):
            # chain = WLChain(N_backbone,a_backbone,lambda_backbone,unit_C)
            chain.grid = grid
            chain.apply_SA = 0
            chain.d_exc = 0.1
            chain.f = 0.0
            chain.kappa = kappa
            chain.epsilon = epsilon

            chain.chain_grid()
            chain.affine()
            N = chain.N
            chain_box = chain.box
            
            chain.scatter_direct_SHE(qq, rr, lm, n_merge=n_merge, calculate_g_r=calculate_g_r, real=real)
            S_q_lm_i = S_q_lm_i + chain.S_q_lm
            g_r_lm_i = g_r_lm_i + chain.g_r_lm
        S_q_lm_list.append(S_q_lm_i/n_sample) # Append the S(Q) of given grid type
        g_r_lm_list.append(g_r_lm_i/n_sample) # Append the g(r) of given grid type

    ## Rayleigh chain
    if rayleigh:
        grid = 'KP'
        print(grid)
        S_q_lm_i = np.zeros([len(qq),len(lm)])
        g_r_lm_i = np.zeros([len(rr),len(lm)])
        chain.a = kappa
        for j in trange(n_sample):
            chain.chain()
            chain.affine()
            chain.scatter_direct_SHE(qq, rr, lm, n_merge=n_merge, calculate_g_r=calculate_g_r, real=real)
            S_q_lm_i = S_q_lm_i + chain.S_q_lm 
            g_r_lm_i = g_r_lm_i + chain.g_r_lm 
        S_q_lm_list.append(S_q_lm_i/n_sample) # Append the S(Q) of given grid type
        g_r_lm_list.append(g_r_lm_i/n_sample) # Append the g(r) of given grid type

    if calculate_g_r:
        return np.array(S_q_lm_list), np.array(g_r_lm_list)
    else:
        return np.array(S_q_lm_list)

# %%
## Chain parameters
# Coordinate of C atoms in each unit
# unit_C = load('b_c.dat')';
unit_C = np.zeros((3,1))

# Degree of polymerization
N_backbone = 5000

# Chain stiffness (placeholder)
a_backbone = 1

# Unit persistence
lambda_backbone = 1

# Affine deformation matrix
F = np.array([[1/np.sqrt(2),0,0],[0,1/np.sqrt(2),0],[0,0,2]])

# Call WLChain class
chain = WLChain(N_backbone,a_backbone,lambda_backbone,unit_C)
chain.apply_SA = 1
chain.d_exc = 0.1
chain.f = 0.0
chain.F = F

kappa_list = np.array([50])
epsilon_list = [0]

n_q = 41
n_r = 51
qq = (np.logspace(-1,3,n_q))/N_backbone
rr = (np.arange(n_r))*2

lm=[(0,0),(2,0),(4,0),(6,0)]

parameters_list = []
S_q_lm_list_param = []
g_r_lm_list_param = []
for kappa in kappa_list:
    for epsilon in epsilon_list:
        parameters_list.append([kappa, epsilon])
        n_sample = 512
        
        # Chain stiffness
        chain.kappa = kappa
        chain.epsilon = epsilon

        S_q_lm_list, g_r_lm_list= eval_sq_SHE(kappa, epsilon, chain, qq, rr, lm,
                                 n_sample = n_sample, n_merge = 1, rayleigh=True, calculate_g_r=True, real=True)
        S_q_lm_list_param.append(S_q_lm_list)
        g_r_lm_list_param.append(g_r_lm_list)
        
        S_q_lm_list_param = np.array(S_q_lm_list_param)
        g_r_lm_list_param = np.array(g_r_lm_list_param)
        mdic = {"S_q_lm_list_param":S_q_lm_list_param, "g_r_lm_list_param":g_r_lm_list_param, "qq":qq, "rr":rr, "parameters_list":parameters_list,"lm":lm}
        savemat("sq_lm_zz_1.0_{}_{}_{}.mat".format(n_sample,kappa,epsilon),mdic)
