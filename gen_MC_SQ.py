# import os
# os.getcwd()
# os.chdir('/home/chtung/project_MC')

import numpy as np
import time
from tqdm import tqdm, trange
from WLM import WLChain
# import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat

#%% Chain parameters
# Coordinate of C atoms in each unit
# unit_C = load('b_c.dat')';
unit_C = np.zeros((3,1))

# Degree of polymerization
N_backbone = 5000

# Chain stiffness (placeholder)
a_backbone = 1

# Unit persistence
lambda_backbone = 1

#%% Generate scattering function associated with given parameters
def eval_sq(kappa, epsilon, chain, qq, n_sample = 20, n_merge = 4, rayleigh=False):
    ## generate spectra of three different grids
    S_q_list = []

    chain_Cc_list = []
    for i, grid in enumerate(['SC','RB','RT']):
        S_q_i = np.zeros_like(qq)
        E_total = 0
        print(grid)
        for j in trange(n_sample):
            # chain = WLChain(N_backbone,a_backbone,lambda_backbone,unit_C)
            chain.grid = grid
            chain.apply_SA = 0
            chain.d_exc = 0.1
            chain.f = 0.0
            chain.kappa = kappa
            chain.epsilon = epsilon


            chain.chain_grid()
            N = chain.N
            chain_box = chain.box
            
            chain.scatter_direct(qq,n_merge=n_merge)
            S_q_i = S_q_i + chain.S_q

        #     E_j = np.sum(chain.E_list)
        #     E_total+=E_j
        # E_total = E_total/n_sample
        # print(E_total)

        qq = chain.qq    
        S_q_list.append(S_q_i/n_sample) # Append the S(Q) of given grid type

#%% Assign kappa and epsilon, then calculate S(Q)
kappa_list = [5,15,45,135]
epsilon_list = [0]

# Call WLChain class
chain = WLChain(N_backbone,a_backbone,lambda_backbone,unit_C)
chain.apply_SA = 1
chain.d_exc = 0.1
chain.f = 0.0

# Q points, QL/2pi = 0.1 to 1000
n_q = 101
qq = 2*np.pi*(np.logspace(-1,3,n_q))/N_backbone

parameters_list = []
S_q_list_param = []
for kappa in kappa_list:
    for epsilon in epsilon_list:
        parameters_list.append([kappa, epsilon])

        # Chain stiffness
        chain.kappa = kappa
        chain.epsilon = epsilon

        S_q_list_grid = eval_sq(kappa, epsilon, chain, qq, n_sample = 10, n_merge = 1, rayleigh=True)
        S_q_list_param.append(S_q_list_grid)

S_q_list_param = np.array(S_q_list_param)
mdic = {"S_q_list_param":S_q_list_param, "qq":qq, "parameters_list":parameters_list}
savemat("S_q_list_param.mat",mdic)