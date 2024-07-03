# import os
# os.getcwd()
# os.chdir('/home/chtung/project_MC')

import numpy as np
import time
from tqdm import tqdm, trange
from WLM import WLChain
# import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import sys

def eval_sq(kappa, epsilon, chain, qq, n_sample = 20, n_merge = 4, rayleigh=False):
    ## generate spectra of three different grids
    S_q_list = []
    S_q_2D_list = []

    chain_Cc_list = []
    for i, grid in enumerate(['SC','RB','RT']):
        S_q_i = np.zeros_like(qq)
        qq_2D = np.concatenate((-np.flip(qq), np.array([0.0]), qq))
        S_q_2D_i = np.zeros([len(qq_2D),len(qq_2D),3])
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
            
            chain.scatter_direct_aniso(qq,n_merge=n_merge)
            S_q_i = S_q_i + chain.S_q
            S_q_2D_i = S_q_2D_i + chain.S_q_2D

        #     E_j = np.sum(chain.E_list)
        #     E_total+=E_j
        # E_total = E_total/n_sample
        # print(E_total)

        qq = chain.qq    
        qq_2D = chain.qq_2D
        S_q_list.append(S_q_i/n_sample) # Append the S(Q) of given grid type
        S_q_2D_list.append(S_q_2D_i/n_sample) # Append the 2D S(Q) of given grid type

    ## Rayleigh chain
    if rayleigh:
        S_q_i = np.zeros_like(qq)
        qq_2D = np.concatenate((-np.flip(qq), np.array([0.0]), qq))
        S_q_2D_i = np.zeros([len(qq_2D),len(qq_2D),3])
        chain.a = kappa
        for j in trange(n_sample):
            chain.chain()
            chain.affine()
            chain.scatter_direct_aniso(qq,n_merge=n_merge)
            S_q_i = S_q_i + chain.S_q
            S_q_2D_i = S_q_2D_i + chain.S_q_2D
        S_q_list.append(S_q_i/n_sample) # Append the S(Q) of given grid type
        S_q_2D_list.append(S_q_2D_i/n_sample) # Append the 2D S(Q) of given grid type

    return np.array(S_q_list), np.array(S_q_2D_list)

if __name__ == '__main__':
    input_params = sys.argv[2:8]
    # -i keppa epsilon N_backbone n_sample n_merge filename
    print("-i keppa={} epsilon={} N_backbone={} n_sample={} n_merge={} filename={}".format(*input_params))
    print(input_params)
    #%% Assign kappa and epsilon, then calculate S(Q)
    kappa_list = [float(input_params[0])]
    epsilon_list = [float(input_params[1])]

    #%% Chain parameters
    # Coordinate of C atoms in each unit
    # unit_C = load('b_c.dat')';
    unit_C = np.zeros((3,1))

    # Degree of polymerization
    N_backbone = int(input_params[2])

    # Chain stiffness (placeholder)
    a_backbone = 1

    # Unit persistence
    lambda_backbone = 1

    # Affine deformation matrix
    F = np.array([[1,0,1],[0,1,0],[0,0,1]])

    # Call WLChain class
    chain = WLChain(N_backbone,a_backbone,lambda_backbone,unit_C)
    chain.apply_SA = 0
    chain.d_exc = 0.1
    chain.f = 0.0
    chain.F = F

    # Q points, QL/2pi = 0.1 to 1000
    n_q = 25
    qq = 2*np.pi*(np.linspace(4,100,n_q))/N_backbone

    parameters_list = []
    S_q_list_param = []
    S_q_2D_list_param = []
    for kappa in kappa_list:
        for epsilon in epsilon_list:
            parameters_list.append([kappa, epsilon])

            # Chain stiffness
            chain.kappa = kappa
            chain.epsilon = epsilon

            time_start = time.time()
            S_q_list_grid, S_q_2D_list_grid = eval_sq(kappa, epsilon, chain, qq, n_sample = int(input_params[3]), n_merge = int(input_params[4]), rayleigh=True)

            time_end = time.time()
            print("time used: {}".format(time_end-time_start))

            S_q_list_param.append(S_q_list_grid)
            S_q_2D_list_param.append(S_q_2D_list_grid)
    
    S_q_list_param = np.array(S_q_list_param)
    S_q_2D_list_param = np.array(S_q_2D_list_param)
    mdic = {"S_q_list_param":S_q_list_param, "S_q_2D_list_param":S_q_2D_list_param, "qq":qq, "parameters_list":parameters_list}
    savemat(input_params[5]+"_{:}_{:}.mat".format(float(input_params[0]),float(input_params[1])),mdic)