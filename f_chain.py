# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:27:38 2021
chain
@author: CHTUNG
"""
import numpy as np
import f_rotation
rotation = f_rotation.rotation
rotation_dihedral = f_rotation.rotation_dihedral
# rotation_stretched = f_rotation.rotation_stretched


def chain_Rayleigh(N, a, lambda_seg, unit_C, apply_SA=1, d_exc=1):
    """
    Modelling the polymer chain as a semi-flexible rod.
    
    Assuming the bending energy is propotional to the square of bending angle
        e = 1/2 * a * theta^2
        
    the partition function: 
        Z = exp(-e/kT)
        
    probability distribution of theta:
        p(theta) = Z(theta)sin(theta) / integral(Z(theta)sin(theta)) from 0 to pi
                 = exp(-a*theta^2/2kT)*sin(theta)
                 
    for theta << 1, p(theta) can be approximated by:
        exp(-a*theta^2/2kT)*(theta) (Rayleigh distribution).
        
    The CDF of Rayleigh distribution is:
        1-exp(-theta^2/2a^2)
        
    and its inverse function:
        sqrt(-2/a ln(1-X)).
    -------------------------------------------------------
    Args:
    N: int
        Number of segments
        
    a: float
        chain stiffness, persistence length
    
    lambda_seg: float
        segment length
    
    unit_C: 3*n float array
        repetive units in each segment
        
    apply_SA: boolean
        apply self avoiding check
        
    d_exc: float
        minimum interparticle distance of the self avoiding chain
    """
    d2_exc = d_exc**2
    i_diameter = int(np.ceil(np.pi/2*d_exc/lambda_seg)) 
    # Check for sphere overlap was done for points 
    # separated by more than pi*d_exc/2 along the contour
       
    n = np.zeros((3,N))
    l = np.zeros((3,N))
    lc = np.zeros((3,N))
    #B = np.zeros((3,3))
    #C = np.zeros((3,3))
    #D = np.zeros((3,3))
    R = np.zeros((3,3))
    O = np.zeros((3,3,N))
    
    abort = 1
    while abort==1:
        abort = 0
        for i in range(N):
            if i==0:
                # n[:,i] = [1,0,0]
                n_0 = np.random.rand(3)
                n[:,i] = n_0/np.linalg.norm(n_0)
                l[:,i] = n[:,i]
                #B = np.eye(3)
                #C = np.eye(3)
                #D = np.eye(3)
                R = np.eye(3)
                O[:,:,i] = R
            else:
                R = rotation(O[:,:,i-1],a)
                
                O[:,:,i] = R@O[:,:,i-1]
                # O[:,:,i] = O[:,:,i]/np.sqrt(np.sum(O[:,:,i]**2,axis=0))
                n[:,i] = O[:,0,i].reshape((3))
                # n[:,i] = n[:,i]/np.sqrt(np.sum(n[:,i]**2))
                l[:,i] = l[:,i-1] + n[:,i]
                
                if i<i_diameter:
                    continue
                
                #%% check self avoiding
                # if apply_SA:
                SA = 0
                
                n_retry = -1
                while (SA == 0) & (n_retry < 100):
                    n_retry += 1
                    
                    # if n_retry > 100:
                    #     abort = 1
                    #     print('abort')
                    #     break
                        
                    d2_uv_min = np.min(np.sum((l[:,:i-i_diameter+1].T-l[:,i].T)**2,axis=1))
                    # d1_uv_min = np.min(np.max(np.abs(l[:,:i-1].T-l[:,i].T),axis=1))
                    # print(d1_uv_min)
                    
                    if d2_uv_min<d2_exc:
                    # if d1_uv_min<d_exc:
                        print('retry ({:d})'.format(n_retry+1))
                        # n_retry+=1
                        R = rotation(O[:,:,i-1],a)
            
                        O[:,:,i] = R@O[:,:,i-1]
                        # O[:,:,i] = O[:,:,i]/np.sqrt(np.sum(O[:,:,i]**2,axis=0))
                        n[:,i] = O[:,1,i].reshape((3))
                        # n[:,i] = n[:,i]/np.sqrt(np.sum(n[:,i]**2))
                        l[:,i] = l[:,i-1] + n[:,i]
                    else:
                        if n_retry!=0:
                            print('retry (end)')
                        break
                    
                if n_retry >= 100:
                    abort = 1
                    print('abort')
                    break
        
    lc = l*lambda_seg

    #%% map unimer
    #C
    nC = unit_C.shape[1]
    m_backbone_C = np.zeros((3,nC,N))
    for j in range(N):
        for k in range(nC):
            m_backbone_C[:,k,j] = O[:,:,j]@unit_C[:,k] + lc[:,j] + np.array([0,0,0])
    
    Cc = np.reshape(m_backbone_C,(3,N*nC))
    
    # print(n_retry)
    return lc, Cc, O, n

#%%
def chain_Rayleigh_woSA(N, a, lambda_seg, unit_C, apply_SA=0, d_exc=1):
    # d2_exc = d_exc**2
    i_diameter = int(np.ceil(np.pi/2*d_exc/lambda_seg)) 
    # Check for sphere overlap was done for points 
    # separated by more than pi*d_exc/2 along the contour
       
    n = np.zeros((3,N))
    l = np.zeros((3,N))
    lc = np.zeros((3,N))
    #B = np.zeros((3,3))
    #C = np.zeros((3,3))
    #D = np.zeros((3,3))
    R = np.zeros((3,3))
    O = np.zeros((3,3,N))
    
    abort = 1
    while abort==1:
        abort = 0
        for i in range(N):
            if i==0:
                # n[:,i] = [1,0,0]
                n_0 = np.random.rand(3)
                n[:,i] = n_0/np.linalg.norm(n_0)
                l[:,i] = n[:,i]
                #B = np.eye(3)
                #C = np.eye(3)
                #D = np.eye(3)
                R = np.eye(3)
                O[:,:,i] = R
            else:
                R = rotation(O[:,:,i-1],a)
                
                O[:,:,i] = R@O[:,:,i-1]
                # O[:,:,i] = O[:,:,i]/np.sqrt(np.sum(O[:,:,i]**2,axis=0))
                n[:,i] = O[:,0,i].reshape((3))
                # n[:,i] = n[:,i]/np.sqrt(np.sum(n[:,i]**2))
                l[:,i] = l[:,i-1] + n[:,i]
                
                if i<i_diameter:
                    continue
        
    lc = l*lambda_seg

    #%% map unimer
    #C
    nC = unit_C.shape[1]
    m_backbone_C = np.zeros((3,nC,N))
    for j in range(N):
        for k in range(nC):
            m_backbone_C[:,k,j] = O[:,:,j]@unit_C[:,k] + lc[:,j] + np.array([0,0,0])
    
    Cc = np.reshape(m_backbone_C,(3,N*nC))
    
    # print(n_retry)
    return lc, Cc, O, n

#%%
def chain_fix_val_free_rot(N, a, lambda_seg, unit_C, apply_SA=1, d_exc=1):
    """
    Modelling the polymer chain by fixed-valence-free-rotation model
    
    theta is fixed
    persistence:
        2*l_p = l_b*(1+cos(theta))/(1-cos(theta)) 
              = l_b*cot^2(theta/2)
        theta = 2*arctan(1/sqrt(2*l_p))
    -------------------------------------------------------
    Args:
    N: int
        Number of segments
        
    a: float
        chain stiffness, persistence length
    
    lambda_seg: float
        segment length
    
    unit_C: 3*n float array
        repetive units in each segment
        
    apply_SA: boolean
        apply self avoiding check
        
    d_exc: float
        minimum interparticle distance of the self avoiding chain
    """
    d2_exc = d_exc**2
    i_diameter = int(np.ceil(np.pi/2*d_exc/lambda_seg)) 
    # Check for sphere overlap was done for points 
    # separated by more than pi*d_exc/2 along the contour
       
    n = np.zeros((3,N))
    l = np.zeros((3,N))
    lc = np.zeros((3,N))
    #B = np.zeros((3,3))
    #C = np.zeros((3,3))
    #D = np.zeros((3,3))
    R = np.zeros((3,3))
    O = np.zeros((3,3,N))
    
    abort = 1
    while abort==1:
        abort = 0
        for i in range(N):
            if i==0:
                # n[:,i] = [1,0,0]
                n_0 = np.random.rand(3)
                n[:,i] = n_0/np.linalg.norm(n_0)
                l[:,i] = n[:,i]
                #B = np.eye(3)
                #C = np.eye(3)
                #D = np.eye(3)
                R = np.eye(3)
                O[:,:,i] = R
            else:
                R = rotation_dihedral(O[:,:,i-1],a)
                
                O[:,:,i] = R@O[:,:,i-1]
                # O[:,:,i] = O[:,:,i]/np.sqrt(np.sum(O[:,:,i]**2,axis=0))
                n[:,i] = O[:,0,i].reshape((3))
                # n[:,i] = n[:,i]/np.sqrt(np.sum(n[:,i]**2))
                l[:,i] = l[:,i-1] + n[:,i]
                
                if i<i_diameter:
                    continue
                
                #%% check self avoiding
                # if apply_SA:
                SA = 0
                
                n_retry = -1
                while (SA == 0) & (n_retry < 100):
                    n_retry += 1
                    
                    # if n_retry > 100:
                    #     abort = 1
                    #     print('abort')
                    #     break
                        
                    d2_uv_min = np.min(np.sum((l[:,:i-i_diameter+1].T-l[:,i].T)**2,axis=1))
                    # d1_uv_min = np.min(np.max(np.abs(l[:,:i-1].T-l[:,i].T),axis=1))
                    # print(d1_uv_min)
                    
                    if d2_uv_min<d2_exc:
                    # if d1_uv_min<d_exc:
                        print('retry ({:d})'.format(n_retry+1))
                        # n_retry+=1
                        R = rotation_dihedral(O[:,:,i-1],a)
            
                        O[:,:,i] = R@O[:,:,i-1]
                        # O[:,:,i] = O[:,:,i]/np.sqrt(np.sum(O[:,:,i]**2,axis=0))
                        n[:,i] = O[:,1,i].reshape((3))
                        # n[:,i] = n[:,i]/np.sqrt(np.sum(n[:,i]**2))
                        l[:,i] = l[:,i-1] + n[:,i]
                    else:
                        if n_retry!=0:
                            print('retry (end)')
                        break
                    
                if n_retry >= 100:
                    abort = 1
                    print('abort')
                    break
        
    lc = l*lambda_seg
    
    #%% map unimer
    #C
    nC = unit_C.shape[1]
    m_backbone_C = np.zeros((3,nC,N))
    for j in range(N):
        for k in range(nC):
            m_backbone_C[:,k,j] = O[:,:,j]@unit_C[:,k] + lc[:,j] + np.array([0,0,0])
    
    Cc = np.reshape(m_backbone_C,(3,N*nC))
    
    # print(n_retry)
    return lc, Cc, O, n

#%%
def chain_fix_val_free_rot_woSA(N, a, lambda_seg, unit_C, apply_SA=0, d_exc=1):
    # d2_exc = d_exc**2
    i_diameter = int(np.ceil(np.pi/2*d_exc/lambda_seg)) 
    # Check for sphere overlap was done for points 
    # separated by more than pi*d_exc/2 along the contour
       
    n = np.zeros((3,N))
    l = np.zeros((3,N))
    lc = np.zeros((3,N))
    #B = np.zeros((3,3))
    #C = np.zeros((3,3))
    #D = np.zeros((3,3))
    R = np.zeros((3,3))
    O = np.zeros((3,3,N))
    
    abort = 1
    while abort==1:
        abort = 0
        for i in range(N):
            if i==0:
                # n[:,i] = [1,0,0]
                n_0 = np.random.rand(3)
                n[:,i] = n_0/np.linalg.norm(n_0)
                l[:,i] = n[:,i]
                #B = np.eye(3)
                #C = np.eye(3)
                #D = np.eye(3)
                R = np.eye(3)
                O[:,:,i] = R
            else:
                R = rotation_dihedral(O[:,:,i-1],a)
                
                O[:,:,i] = R@O[:,:,i-1]
                # O[:,:,i] = O[:,:,i]/np.sqrt(np.sum(O[:,:,i]**2,axis=0))
                n[:,i] = O[:,0,i].reshape((3))
                # n[:,i] = n[:,i]/np.sqrt(np.sum(n[:,i]**2))
                l[:,i] = l[:,i-1] + n[:,i]
                
                if i<i_diameter:
                    continue
        
    lc = l*lambda_seg
    
    #%% map unimer
    #C
    nC = unit_C.shape[1]
    m_backbone_C = np.zeros((3,nC,N))
    for j in range(N):
        for k in range(nC):
            m_backbone_C[:,k,j] = O[:,:,j]@unit_C[:,k] + lc[:,j] + np.array([0,0,0])
    
    Cc = np.reshape(m_backbone_C,(3,N*nC))
    
    # print(n_retry)
    return lc, Cc, O, n

#%% 
phi = (1+np.sqrt(5))/2
def grid_coords(grid):
    if grid=='SC':
        # 6 orientations following the arrangement of standard dice
        # 1/6 for +x/-x; 2/5 for +y/-y; 3/4 for +z/-z
        r_n = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,-1],[0,-1,0],[-1,0,0]])
        r_opp = np.array([5,4,3,2,1,0])
    
    if grid=='RB':
        # rhombic dodecahedron
        r_n = np.array([[0,1,1],[0,-1,-1],[0,1,-1],[0,-1,1],
                        [1,0,1],[-1,0,-1],[1,0,-1],[-1,0,1],
                        [1,1,0],[-1,-1,0],[1,-1,0],[-1,1,0]])/np.sqrt(2)
        r_opp = np.array([1,0,3,2,
                          5,4,7,6,
                          9,8,11,10])
        
    if grid=='RT':
        phi = (1+np.sqrt(5))/2
        # rhombic triacontahedron
        r_n = np.array([
                       [phi,0,0],[0,phi,0],[0,0,phi],
                       [-phi,0,0],[0,-phi,0],[0,0,-phi],
                       [1/2,phi/2,phi**2/2],[phi**2/2,1/2,phi/2],[phi/2,phi**2/2,1/2],
                       [-1/2,phi/2,phi**2/2],[-phi**2/2,1/2,phi/2],[-phi/2,phi**2/2,1/2],
                       [1/2,-phi/2,phi**2/2],[phi**2/2,-1/2,phi/2],[phi/2,-phi**2/2,1/2],
                       [1/2,phi/2,-phi**2/2],[phi**2/2,1/2,-phi/2],[phi/2,phi**2/2,-1/2],
                       [-1/2,-phi/2,phi**2/2],[-phi**2/2,-1/2,phi/2],[-phi/2,-phi**2/2,1/2],
                       [1/2,-phi/2,-phi**2/2],[phi**2/2,-1/2,-phi/2],[phi/2,-phi**2/2,-1/2],
                       [-1/2,phi/2,-phi**2/2],[-phi**2/2,1/2,-phi/2],[-phi/2,phi**2/2,-1/2],
                       [-1/2,-phi/2,-phi**2/2],[-phi**2/2,-1/2,-phi/2],[-phi/2,-phi**2/2,-1/2]
                       ])
        r_n = np.array([points/np.sqrt(np.sum(points**2)) for points in r_n])
        r_opp = np.array([3,4,5,
                          0,1,2,
                          27,28,29,
                          21,22,23,
                          24,25,26,
                          18,19,20,
                          15,16,17,
                          9,10,11,
                          12,13,14,
                          6,7,8])
        
    if grid=='TH':
        # tetrahedron
        r_n = np.array([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]])
        r_n = np.array([points/np.sqrt(np.sum(points**2)) for points in r_n])
        r_opp = np.array([])

    if grid=='OH':
        # tetrahedron
        r_n = np.array([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1],[1,-1,-1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]])
        r_n = np.array([points/np.sqrt(np.sum(points**2)) for points in r_n])
        r_opp = np.array([7,4,5,6,3,2,1,0])

    if grid=='OHTH':
        # tetrahedron
        r_n = np.array([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1],[1,-1,-1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]])
        r_n = np.array([points/np.sqrt(np.sum(points**2)) for points in r_n])
        cos_ij = r_n@r_n.T
        r_opp = np.array([[x[0] for x in np.argwhere(cos_ij[i,:]<0)]+[i] for i in range(len(cos_ij))])
        
    return r_n, r_opp

def kappa_eff(l_p, grid, nonreverse=True):
    r_n, _ = grid_coords(grid)
    cos_i1 = r_n@r_n[0].T
    cos_i1[cos_i1>1]=1
    cos_i1[cos_i1<-1]=-1
    theta_i1 = np.arccos(cos_i1)

    def f_average(x, kappa):
        # x(theta) is a fumction of theta
        x_ave = np.sum(x*np.exp(-kappa/2*theta_i1**2))/np.sum(np.exp(-kappa/2*theta_i1**2))
        return x_ave

    def f_average_nonrev(x, kappa):
        # x(theta) is a fumction of theta
        x_ave = np.sum((x*np.exp(-kappa/2*theta_i1**2))[cos_i1>-1])/np.sum(np.exp(-kappa/2*theta_i1**2)[cos_i1>-1])
        return x_ave
    
    # get the relation between bending energy and chain persistence
    kappa_list = np.logspace(-2,2,200)
    if nonreverse:
        f_ave = f_average_nonrev
    else:
        f_ave = f_average

    cos_ave = np.array([f_ave(cos_i1, k) for k in kappa_list])
    sigma = 1e-18
    lp_ave_list = -1/(np.log(cos_ave)-sigma)

    # obtain the corresponding kappa_eff from interpolation
    i_finite = np.isfinite(lp_ave_list)
    # f_interp = interpolate.PchipInterpolator(lp_ave_list, kappa_list)
    # kap_eff = f_interp(l_p)
    kap_eff = np.interp(l_p, lp_ave_list[i_finite], kappa_list[i_finite])

    return kap_eff

def lp_kappa(kappa, grid, nonreverse=True):
    r_n, _ = grid_coords(grid)
    cos_i1 = r_n@r_n[0].T
    cos_i1[cos_i1>1]=1
    cos_i1[cos_i1<-1]=-1
    theta_i1 = np.arccos(cos_i1)

    def f_average(x, kappa):
        # x(theta) is a fumction of theta
        x_ave = np.sum(x*np.exp(-kappa/2*theta_i1**2))/np.sum(np.exp(-kappa/2*theta_i1**2))
        return x_ave

    def f_average_nonrev(x, kappa):
        # x(theta) is a fumction of theta
        x_ave = np.sum((x*np.exp(-kappa/2*theta_i1**2))[cos_i1>-1])/np.sum(np.exp(-kappa/2*theta_i1**2)[cos_i1>-1])
        return x_ave
    
    # get the relation between bending energy and chain persistence
    kappa_list = np.logspace(-2,2,200)
    if nonreverse:
        f_ave = f_average_nonrev
    else:
        f_ave = f_average

    sigma = 1e-18

    # obtain the corresponding kappa_eff from interpolation
    lp_kappa = -1/(np.log(f_ave(cos_i1, kappa))-sigma)

    return lp_kappa


import random
def chain_grid(N, kappa, epsilon, lambda_seg, apply_SA=1, d_exc=1, grid='SC', original_kappa=False):
    """
    Monte Carlo simulations of lattice models for polymer chains.
    
    kappa: energy for kink, kappa = a
    
    epsilon:  force coupling to the extension
    
    e = 1/2 * a * theta(r_ij)^2 + f * r_ij dot X
        
    the partition function: 
        Z = exp(-e/kT)
        
    probability distribution of theta:
        p(r_ij) = Z(r_ij) / integral(Z(r_ij))
                 
    calculate the probability of all possible walks
    choose r_ij according to the probability
    -------------------------------------------------------
    Args:
    N: int
        Number of segments
        
    a: float
        chain stiffness, persistence length
    
    lambda_seg: float
        segment length
    
    unit_C: 3*n float array
        repetive units in each segment
        
    apply_SA: boolean
        apply self avoiding check
        
    d_exc: float
        minimum interparticle distance of the self avoiding chain
    """
    d2_exc = d_exc**2
    i_diameter = int(np.ceil(np.pi/2*d_exc/lambda_seg))
    
    # coordinate 
    r_n, r_opp = grid_coords(grid)
    
    cos_ij = r_n@r_n.T
    cos_ij[cos_ij>1]=1
    cos_ij[cos_ij<-1]=-1
    sin_ij2 = 1-cos_ij**2
    
    n = np.zeros((3,N))
    n_i = np.zeros((N)).astype(int)
    l = np.zeros((3,N))
    
    # energy
    Z = np.zeros((len(r_n),len(r_n)))
    EZ = np.zeros((len(r_n),len(r_n)))
    for i in range(len(r_n)):
        # E_phi = kappa/2*(sin_ij2[iz,:])
        # E_x = -epsilon*(cos_ij[0,:])
        kap_eff = kappa_eff(kappa,grid)
        if original_kappa:
            kap_eff = kappa
        E_phi = kap_eff/2*((np.arccos(cos_ij[i,:]))**2)
        E_x = -epsilon*(r_n[:,0])
        
        E = E_phi + E_x
        z_i = np.exp(-E)
        z_i[r_opp[i]] = 0     
        
        z_i = z_i/np.sum(z_i)
        Z[:,i] = z_i
        
        Ez_i = E
        EZ[:,i] = Ez_i

    E_list = []
    i_array = np.arange(len(r_n))
    abort = 1
    while abort==1:
        abort = 0
        for i in range(N):
            if i==0:
                n_i[i] = random.choice(i_array)
                E_list.append(0)
                n[:,i] = r_n[n_i[i],:]
                l[:,i] = n[:,i]
                
            else:
                z = Z[:,n_i[i-1]]
                n_i[i] = random.choices(i_array,weights=z)[0]
                E_list.append(EZ[n_i[i],n_i[i-1]])
                            
                n[:,i] = r_n[n_i[i],:]
                l[:,i] = l[:,i-1] + n[:,i]
                
                if i<i_diameter:
                        continue
                
                #%% check self avoiding
                # if apply_SA:
                SA = 0
                
                n_retry = -1
                while (SA == 0) & (n_retry < 100):
                    n_retry += 1
                    
                    # if n_retry > 100:
                    #     abort = 1
                    #     print('abort')
                    #     break
                        
                    d2_uv_min = np.min(np.sum((l[:,:i-i_diameter+1].T-l[:,i].T)**2,axis=1))
                    # d1_uv_min = np.min(np.max(np.abs(l[:,:i-1].T-l[:,i].T),axis=1))
                    # print(d1_uv_min)
                    
                    if d2_uv_min<d2_exc:
                    # if d1_uv_min<d_exc:
                        print('retry ({:d})'.format(n_retry+1))
                        # n_retry+=1
                        z = Z[:,n_i[i-1]]
                        n_i[i] = random.choices(i_array,weights=z)[0]
                        E_list.append(EZ[n_i[i],n_i[i-1]])
                        n[:,i] = r_n[n_i[i],:]
                        l[:,i] = l[:,i-1] + n[:,i]
                        
                    else:
                        if n_retry!=0:
                            print('retry (end)')
                        break
                    
                if n_retry >= 100:
                    abort = 1
                    print('abort')
                    break
                    
    lc = l*lambda_seg
    Cc = lc
    
    return lc, Cc, n, Z, E_list

def chain_grid_woSA(N, kappa, epsilon, lambda_seg, apply_SA=1, d_exc=0, grid='SC', original_kappa=False):
    # d2_exc = d_exc**2
    i_diameter = int(np.ceil(np.pi/2*d_exc/lambda_seg))
    
    # coordinate 
    r_n, r_opp = grid_coords(grid)
    
    cos_ij = r_n@r_n.T
    cos_ij[cos_ij>1]=1
    cos_ij[cos_ij<-1]=-1
    sin_ij2 = 1-cos_ij**2
    
    n = np.zeros((3,N))
    n_i = np.zeros((N)).astype(int)
    l = np.zeros((3,N))
    
    # energy
    Z = np.zeros((len(r_n),len(r_n)))
    EZ = np.zeros((len(r_n),len(r_n)))
    for i in range(len(r_n)):
        # E_phi = kappa/2*(sin_ij2[iz,:])
        # E_x = -epsilon*(cos_ij[0,:])
        kap_eff = kappa_eff(kappa,grid)
        if original_kappa:
            kap_eff = kappa
        E_phi = kap_eff/2*((np.arccos(cos_ij[i,:]))**2)
        E_x = -epsilon*(r_n[:,0])
        
        E = E_phi + E_x
        z_i = np.exp(-E)
        z_i[r_opp[i]] = 0     
        
        z_i = z_i/np.sum(z_i)
        Z[:,i] = z_i
        
        Ez_i = E
        EZ[:,i] = Ez_i

    E_list = []
    i_array = np.arange(len(r_n))
    abort = 1
    while abort==1:
        abort = 0
        for i in range(N):
            if i==0:
                n_i[i] = random.choice(i_array)
                E_list.append(0)
                n[:,i] = r_n[n_i[i],:]
                l[:,i] = n[:,i]
                
            else:
                z = Z[:,n_i[i-1]]
                n_i[i] = random.choices(i_array,weights=z)[0]
                E_list.append(EZ[n_i[i],n_i[i-1]])
                            
                n[:,i] = r_n[n_i[i],:]
                l[:,i] = l[:,i-1] + n[:,i]
                
                if i<i_diameter:
                        continue
                    
    lc = l*lambda_seg
    Cc = lc
    
    return lc, Cc, n, Z, E_list

def chain_grid_shear(N, kappa, epsilon, lambda_seg, apply_SA=1, d_exc=1, grid='SC', original_kappa=False):
    """
    Monte Carlo simulations of lattice models for polymer chains.
    
    kappa: energy for kink
    
    epsilon:  force coupling to the extension
    
    e = 1/2 * a * theta(r_ij)^2 + f * r_ij dot X
        
    the partition function: 
        Z = exp(-e/kT)
        
    probability distribution of theta:
        p(r_ij) = Z(r_ij) / integral(Z(r_ij))
                 
    calculate the probability of all possible walks
    choose r_ij according to the probability
    -------------------------------------------------------
    Args:
    N: int
        Number of segments
        
    a: float
        chain stiffness, persistence length
    
    lambda_seg: float
        segment length
    
    unit_C: 3*n float array
        repetive units in each segment
        
    apply_SA: boolean
        apply self avoiding check
        
    d_exc: float
        minimum interparticle distance of the self avoiding chain
    """
    d2_exc = d_exc**2
    i_diameter = int(np.ceil(np.pi/2*d_exc/lambda_seg))
    
    # coordinate 
    r_n, r_opp = grid_coords(grid)
    
    cos_ij = r_n@r_n.T
    cos_ij[cos_ij>1]=1
    cos_ij[cos_ij<-1]=-1
    sin_ij2 = 1-cos_ij**2
    
    n = np.zeros((3,N))
    n_i = np.zeros((N)).astype(int)
    l = np.zeros((3,N))
    
    abort = 1
    while abort==1:
        abort = 0
        for i in range(N):
            if i==0:
                n_i[i] = random.choice(np.arange(len(r_n)))
                # n_i[i] = 0
                n[:,i] = r_n[n_i[i],:]
                l[:,i] = n[:,i]
                
            else:
                # energy
                Z = np.zeros((len(r_n),len(r_n)))
                for iz in range(len(r_n)):
                    # E_phi = kappa/2*(sin_ij2[iz,:])
                    # E_x = -epsilon*(cos_ij[0,:])*l[1,i-1]
                    kap_eff = kappa_eff(kappa,grid)
                    if original_kappa:
                        kap_eff = kappa
                    E_phi = kap_eff/2*((np.arccos(cos_ij[i,:]))**2)
                    E_x = -epsilon*(r_n[:,0])*l[1,i-1]
                    
                    E = E_phi + E_x
                    
                    z_i = np.exp(-E)
                    z_i[r_opp[iz]] = 0     
                    
                    z_i = z_i/np.sum(z_i)
                    Z[:,iz] = z_i
                    
                z = Z[:,n_i[i-1]]
                
                n_i[i] = random.choices(np.arange(len(r_n)),weights=z)[0]
                            
                n[:,i] = r_n[n_i[i],:]
                l[:,i] = l[:,i-1] + n[:,i]
                
                if i<i_diameter:
                        continue
                
                #%% check self avoiding
                # if apply_SA:
                SA = 0
                
                n_retry = -1
                while (SA == 0) & (n_retry < 100):
                    n_retry += 1
                    
                    # if n_retry > 100:
                    #     abort = 1
                    #     print('abort')
                    #     break
                        
                    d2_uv_min = np.min(np.sum((l[:,:i-i_diameter+1].T-l[:,i].T)**2,axis=1))
                    # d1_uv_min = np.min(np.max(np.abs(l[:,:i-1].T-l[:,i].T),axis=1))
                    # print(d1_uv_min)
                    
                    if d2_uv_min<d2_exc:
                    # if d1_uv_min<d_exc:
                        print('retry ({:d})'.format(n_retry+1))
                        # n_retry+=1
                        z = Z[:,n_i[i-1]]
                        n_i[i] = random.choices(np.arange(len(r_n)),weights=z)[0]
                                    
                        n[:,i] = r_n[n_i[i],:]
                        l[:,i] = l[:,i-1] + n[:,i]
                        
                    else:
                        if n_retry!=0:
                            print('retry (end)')
                        break
                    
                if n_retry >= 100:
                    abort = 1
                    print('abort')
                    break
                    
    lc = l*lambda_seg
    Cc = lc
    
    return lc, Cc, n, Z

def chain_grid_shear_woSA(N, kappa, epsilon, lambda_seg, apply_SA=1, d_exc=1, grid='SC', original_kappa=False):
    """
    Monte Carlo simulations of lattice models for polymer chains.
    
    kappa: energy for kink
    
    epsilon:  force coupling to the extension
    
    e = 1/2 * a * theta(r_ij)^2 + f * r_ij dot X
        
    the partition function: 
        Z = exp(-e/kT)
        
    probability distribution of theta:
        p(r_ij) = Z(r_ij) / integral(Z(r_ij))
                 
    calculate the probability of all possible walks
    choose r_ij according to the probability
    -------------------------------------------------------
    Args:
    N: int
        Number of segments
        
    a: float
        chain stiffness, persistence length
    
    lambda_seg: float
        segment length
    
    unit_C: 3*n float array
        repetive units in each segment
        
    apply_SA: boolean
        apply self avoiding check
        
    d_exc: float
        minimum interparticle distance of the self avoiding chain
    """
    # d2_exc = d_exc**2
    i_diameter = int(np.ceil(np.pi/2*d_exc/lambda_seg))
    
    # coordinate 
    r_n, r_opp = grid_coords(grid)
    
    cos_ij = r_n@r_n.T
    cos_ij[cos_ij>1]=1
    cos_ij[cos_ij<-1]=-1
    sin_ij2 = 1-cos_ij**2
    
    n = np.zeros((3,N))
    n_i = np.zeros((N)).astype(int)
    l = np.zeros((3,N))
    
    abort = 1
    while abort==1:
        abort = 0
        for i in range(N):
            if i==0:
                n_i[i] = random.choice(np.arange(len(r_n)))
                # n_i[i] = 0
                n[:,i] = r_n[n_i[i],:]
                l[:,i] = n[:,i]
                
            else:
                # energy
                Z = np.zeros((len(r_n),len(r_n)))
                for iz in range(len(r_n)):
                    # E_phi = kappa/2*(sin_ij2[iz,:])
                    # E_x = -epsilon*(cos_ij[0,:])*l[1,i-1]
                    kap_eff = kappa_eff(kappa,grid)
                    if original_kappa:
                        kap_eff = kappa
                    E_phi = kap_eff/2*((np.arccos(cos_ij[i,:]))**2)
                    E_x = -epsilon*(r_n[:,0])*l[1,i-1]
                    
                    E = E_phi + E_x
                    z_i = np.exp(-E)
                    z_i[r_opp[iz]] = 0     
                    
                    z_i = z_i/np.sum(z_i)
                    Z[:,iz] = z_i
                    
                z = Z[:,n_i[i-1]]
                
                n_i[i] = random.choices(np.arange(len(r_n)),weights=z)[0]
                            
                n[:,i] = r_n[n_i[i],:]
                l[:,i] = l[:,i-1] + n[:,i]
                
                if i<i_diameter:
                        continue
                
    lc = l*lambda_seg
    Cc = lc
    
    return lc, Cc, n, Z

#%% block co-polymer
def chain_Rayleigh_block(N, a_s, f, lambda_seg, unit_C, apply_SA=1, d_exc=np.array([1,1])):
    """
    Modelling the polymer chain as a semi-flexible rod.
    
    Assuming the bending energy is propotional to the square of bending angle
        e = 1/2 * a * theta^2
        
    the partition function: 
        Z = exp(-e/kT)
        
    probability distribution of theta:
        p(theta) = Z(theta)sin(theta) / integral(Z(theta)sin(theta)) from 0 to pi
                 = exp(-a*theta^2/2kT)*sin(theta)
                 
    for theta << 1, p(theta) can be approximated by:
        exp(-a*theta^2/2kT)*(theta) (Rayleigh distribution).
        
    The CDF of Rayleigh distribution is:
        1-exp(-theta^2/2a^2)
        
    and its inverse function:
        sqrt(-2/a ln(1-X)).
    -------------------------------------------------------
    Args:
    N: int
        Number of segments
        
    a_s: 2*1 float
        chain stiffness, persistence length of the blocks
        
    f: float
        ratio of two blocks, 0<f<1
    
    lambda_seg: float
        segment length
    
    unit_C: 3*n float array
        repetive units in each segment
        
    apply_SA: boolean
        apply self avoiding check
        
    d_exc: float
        minimum interparticle distance of the two blocks in the self avoiding chain 
    """
    d2_exc_s = d_exc**2
    i_diameter_s = (np.ceil(np.pi/2*d_exc/lambda_seg)).astype(int)
    # print(i_diameter_s)
    a = a_s[0]
    d2_exc = d2_exc_s[0]
    i_diameter = i_diameter_s[0]
    # Check for sphere overlap was done for points 
    # separated by more than pi*d_exc/2 along the contour
       
    n = np.zeros((3,N))
    l = np.zeros((3,N))
    lc = np.zeros((3,N))
    #B = np.zeros((3,3))
    #C = np.zeros((3,3))
    #D = np.zeros((3,3))
    R = np.zeros((3,3))
    O = np.zeros((3,3,N))
    
    N1 = int(N*f)
    
    abort = 1
    while abort==1:
        abort = 0
        a = a_s[0]
        d2_exc = d2_exc_s[0]
        i_diameter = i_diameter_s[0]
        
        i_seg = 0
        for i in range(N):
            i_seg += 1
            
            if i_seg>N1:
                a = a_s[1]
                d2_exc = d2_exc_s[1]
                i_diameter = i_diameter_s[1]
                
            
            if i==0:
                n[:,i] = [1,0,0]
                l[:,i] = n[:,i]
                #B = np.eye(3)
                #C = np.eye(3)
                #D = np.eye(3)
                R = np.eye(3)
                O[:,:,i] = R
            else:
                R = rotation(O[:,:,i-1],a)
                
                O[:,:,i] = R@O[:,:,i-1]
                # O[:,:,i] = O[:,:,i]/np.sqrt(np.sum(O[:,:,i]**2,axis=0))
                n[:,i] = O[:,0,i].reshape((3))
                # n[:,i] = n[:,i]/np.sqrt(np.sum(n[:,i]**2))
                l[:,i] = l[:,i-1] + n[:,i]
                
                if i<i_diameter:
                    continue
                
                #%% check self avoiding
                # if apply_SA:
                SA = 0
                
                n_retry = -1
                while (SA == 0) & (n_retry < 100):
                    n_retry += 1
                    
                    # if n_retry > 100:
                    #     abort = 1
                    #     print('abort')
                    #     break
                        
                    d2_uv_min = np.min(np.sum((l[:,:i-i_diameter+1].T-l[:,i].T)**2,axis=1))
                    # d1_uv_min = np.min(np.max(np.abs(l[:,:i-1].T-l[:,i].T),axis=1))
                    # print(d1_uv_min)
                    
                    if d2_uv_min<d2_exc:
                    # if d1_uv_min<d_exc:
                        print('retry ({:d})'.format(n_retry+1))
                        # n_retry+=1
                        R = rotation(O[:,:,i-1],a)
            
                        O[:,:,i] = R@O[:,:,i-1]
                        # O[:,:,i] = O[:,:,i]/np.sqrt(np.sum(O[:,:,i]**2,axis=0))
                        n[:,i] = O[:,1,i].reshape((3))
                        # n[:,i] = n[:,i]/np.sqrt(np.sum(n[:,i]**2))
                        l[:,i] = l[:,i-1] + n[:,i]
                    else:
                        if n_retry!=0:
                            print('retry (end)')
                        break
                    
                if n_retry >= 100:
                    abort = 1
                    print('abort')
                    i_seg = 0
                    break
        
    lc = l*lambda_seg

    #%% map unimer
    #C
    nC = unit_C.shape[1]
    m_backbone_C = np.zeros((3,nC,N))
    for j in range(N):
        for k in range(nC):
            m_backbone_C[:,k,j] = O[:,:,j]@unit_C[:,k] + lc[:,j] + np.array([0,0,0])
    
    Cc = np.reshape(m_backbone_C,(3,N*nC))
    
    # print(n_retry)
    return lc, Cc, O, n, N1

def chain_Rayleigh_block_woSA(N, a_s, f, lambda_seg, unit_C, apply_SA=0, d_exc=np.array([1,1])):
    """
    Modelling the polymer chain as a semi-flexible rod.
    
    Assuming the bending energy is propotional to the square of bending angle
        e = 1/2 * a * theta^2
        
    the partition function: 
        Z = exp(-e/kT)
        
    probability distribution of theta:
        p(theta) = Z(theta)sin(theta) / integral(Z(theta)sin(theta)) from 0 to pi
                 = exp(-a*theta^2/2kT)*sin(theta)
                 
    for theta << 1, p(theta) can be approximated by:
        exp(-a*theta^2/2kT)*(theta) (Rayleigh distribution).
        
    The CDF of Rayleigh distribution is:
        1-exp(-theta^2/2a^2)
        
    and its inverse function:
        sqrt(-2/a ln(1-X)).
    -------------------------------------------------------
    Args:
    N: int
        Number of segments
        
    a_s: 2*1 float
        chain stiffness, persistence length of the blocks
        
    f: float
        ratio of two blocks, 0<f<1
    
    lambda_seg: float
        segment length
    
    unit_C: 3*n float array
        repetive units in each segment
        
    apply_SA: boolean
        apply self avoiding check
        
    d_exc: float
        minimum interparticle distance of the two blocks in the self avoiding chain 
    """
    # d2_exc_s = d_exc**2
    i_diameter_s = (np.ceil(np.pi/2*d_exc/lambda_seg)).astype(int)
    # print(i_diameter_s)
    a = a_s[0]
    # d2_exc = d2_exc_s[0]
    i_diameter = i_diameter_s[0]
    # Check for sphere overlap was done for points 
    # separated by more than pi*d_exc/2 along the contour
       
    n = np.zeros((3,N))
    l = np.zeros((3,N))
    lc = np.zeros((3,N))
    #B = np.zeros((3,3))
    #C = np.zeros((3,3))
    #D = np.zeros((3,3))
    R = np.zeros((3,3))
    O = np.zeros((3,3,N))
    
    N1 = int(N*f)
    
    abort = 1
    while abort==1:
        abort = 0
        a = a_s[0]
        # d2_exc = d2_exc_s[0]
        i_diameter = i_diameter_s[0]
        
        i_seg = 0
        for i in range(N):
            i_seg += 1
            
            if i_seg>N1:
                a = a_s[1]
                # d2_exc = d2_exc_s[1]
                i_diameter = i_diameter_s[1]
                
            
            if i==0:
                n[:,i] = [1,0,0]
                l[:,i] = n[:,i]
                #B = np.eye(3)
                #C = np.eye(3)
                #D = np.eye(3)
                R = np.eye(3)
                O[:,:,i] = R
            else:
                R = rotation(O[:,:,i-1],a)
                
                O[:,:,i] = R@O[:,:,i-1]
                # O[:,:,i] = O[:,:,i]/np.sqrt(np.sum(O[:,:,i]**2,axis=0))
                n[:,i] = O[:,0,i].reshape((3))
                # n[:,i] = n[:,i]/np.sqrt(np.sum(n[:,i]**2))
                l[:,i] = l[:,i-1] + n[:,i]
                
                if i<i_diameter:
                    continue
                       
    lc = l*lambda_seg

    #%% map unimer
    #C
    nC = unit_C.shape[1]
    m_backbone_C = np.zeros((3,nC,N))
    for j in range(N):
        for k in range(nC):
            m_backbone_C[:,k,j] = O[:,:,j]@unit_C[:,k] + lc[:,j] + np.array([0,0,0])
    
    Cc = np.reshape(m_backbone_C,(3,N*nC))
    
    # print(n_retry)
    return lc, Cc, O, n, N1

#%%
# def chain_stretched(N, a, lambda_seg, unit_C, apply_SA=1, d_exc=1):
#     d2_exc = d_exc**2
#     i_diameter = int(np.ceil(np.pi/2*d_exc/lambda_seg)) 
#     # Check for sphere overlap was done for points 
#     # separated by more than pi*d_exc/2 along the contour
       
#     n = np.zeros((3,N))
#     l = np.zeros((3,N))
#     lc = np.zeros((3,N))
#     #B = np.zeros((3,3))
#     #C = np.zeros((3,3))
#     #D = np.zeros((3,3))
#     R = np.zeros((3,3))
#     O = np.zeros((3,3,N))
    
#     abort = 1
#     while abort==1:
#         abort = 0
#         for i in range(N):
#             if i==0:
#                 n[:,i] = [1,0,0]
#                 l[:,i] = n[:,i]
#                 #B = np.eye(3)
#                 #C = np.eye(3)
#                 #D = np.eye(3)
#                 R = np.eye(3)
#                 O[:,:,i] = R
#             else:
#                 R = rotation(O[:,:,i-1],a)
                
#                 O[:,:,i] = R@O[:,:,i-1]
#                 # O[:,:,i] = O[:,:,i]/np.sqrt(np.sum(O[:,:,i]**2,axis=0))
#                 n[:,i] = O[:,0,i].reshape((3))
#                 # n[:,i] = n[:,i]/np.sqrt(np.sum(n[:,i]**2))
#                 l[:,i] = l[:,i-1] + n[:,i]
                
#                 if i<i_diameter:
#                     continue
                
#                 #%% check self avoiding
#                 if apply_SA:
#                     SA = 0
                    
#                     n_retry = -1
#                     while (SA == 0) & (n_retry < 100):
#                         n_retry += 1
                        
#                         # if n_retry > 100:
#                         #     abort = 1
#                         #     print('abort')
#                         #     break
                            
#                         d2_uv_min = np.min(np.sum((l[:,:i-i_diameter+1].T-l[:,i].T)**2,axis=1))
#                         # d1_uv_min = np.min(np.max(np.abs(l[:,:i-1].T-l[:,i].T),axis=1))
#                         # print(d1_uv_min)
                        
#                         if d2_uv_min<d2_exc:
#                         # if d1_uv_min<d_exc:
#                             print('retry ({:d})'.format(n_retry+1))
#                             # n_retry+=1
#                             R = rotation(O[:,:,i-1],a)
                
#                             O[:,:,i] = R@O[:,:,i-1]
#                             # O[:,:,i] = O[:,:,i]/np.sqrt(np.sum(O[:,:,i]**2,axis=0))
#                             n[:,i] = O[:,1,i].reshape((3))
#                             # n[:,i] = n[:,i]/np.sqrt(np.sum(n[:,i]**2))
#                             l[:,i] = l[:,i-1] + n[:,i]
#                         else:
#                             if n_retry!=0:
#                                 print('retry (end)')
#                             break
                        
#                     if n_retry >= 100:
#                         abort = 1
#                         print('abort')
#                         break
        
#     lc = l*lambda_seg

#     #%% map unimer
#     #C
#     nC = unit_C.shape[1]
#     m_backbone_C = np.zeros((3,nC,N))
#     for j in range(N):
#         for k in range(nC):
#             m_backbone_C[:,k,j] = O[:,:,j]@unit_C[:,k] + lc[:,j] + np.array([0,0,0])
    
#     Cc = np.reshape(m_backbone_C,(3,N*nC))
    
#     # print(n_retry)
#     return lc, Cc, O, n