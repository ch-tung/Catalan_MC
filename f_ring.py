# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:27:38 2021
chain
@author: CHTUNG
"""
import numpy as np
from scipy import interpolate
# import quaternion

def ring_harmonic(N,n_harmonics,sigma):
    c_ring = np.zeros((3,N+1))
    # c_ring_deriv = np.zeros((3,N+1))
    
    theta = np.arange(N+1)/N*2*np.pi
    
    for i in range(3):
        phi_i = 2*np.pi*np.random.rand(1)
        
        weight = np.exp(-(np.arange(n_harmonics)+1)**2/sigma**2/2)
        weight = weight/np.sqrt(np.sum(weight**2))
        coeff_c_i = np.random.rand(n_harmonics)*weight
        coeff_s_i = np.random.rand(n_harmonics)*weight
        
        harmonics_c_i = np.cos(np.outer(theta,(np.arange(n_harmonics)+1)) + phi_i)*coeff_c_i
        harmonics_s_i = np.sin(np.outer(theta,(np.arange(n_harmonics)+1)) + phi_i)*coeff_s_i
        
        harmonics_i = harmonics_c_i + harmonics_s_i
        # harmonics_i_deriv = harmonics_s_i*(np.arange(n_harmonics)+1) + -harmonics_c_i*(np.arange(n_harmonics)+1)
        
        c_ring[i,:] = np.sum(harmonics_i,axis=1)
        # c_ring_deriv[i,:] = np.sum(harmonics_i_deriv,axis=1)
    
    arc_segment = np.sqrt(np.sum((c_ring[:,:-1]-c_ring[:,1:])**2,axis=0))
    
    # arc_segment = np.sqrt(np.sum(c_ring_deriv**2,axis=0))
    arc_sum = np.sum(arc_segment)
    
    arc_cum = np.zeros(N+1)
    for i in range(N+1):
        if i==0:
            arc_cum[i] = 0
            
        else:
            arc_cum[i] = arc_cum[i-1] + arc_segment[i-1]
            
    f_arc = interpolate.interp1d(arc_cum, theta, kind='quadratic', fill_value='extrapolate')
    
    arc_seq = np.arange(N+1)/(N+1)*arc_sum
    
    # print(arc_cum)
    # print(arc_seq)
    
    theta_interpolate = f_arc(arc_seq)
    
    f_ring = interpolate.interp1d(theta, c_ring, kind='quadratic', fill_value='extrapolate')
    Cc = f_ring(theta_interpolate)/arc_sum*(N+1)
    
    return Cc

# def ring_q(N,lmbda):
#     list_u = []
#     list_v = []

#     for i in range(N):
#         U = np.random.rand(2)
#         u_re = np.sqrt(-np.log(U[0]))*np.cos(2*np.pi*U[1])
#         u_im = np.sqrt(-np.log(U[0]))*np.sin(2*np.pi*U[1])
#         u = np.quaternion(u_re,u_im,0,0)
#         list_u.append(u)
#         V = np.random.rand(2)
#         v_re = np.sqrt(-np.log(V[0]))*np.cos(2*np.pi*V[1])
#         v_im = np.sqrt(-np.log(V[0]))*np.sin(2*np.pi*V[1])
#         v = np.quaternion(v_re,v_im,0,0)
#         list_v.append(v)
    
#     uu = np.array(list_u)
#     vv = np.array(list_v)
    
#     vv2 = vv - uu*np.sum(np.conjugate(uu)*vv)/np.sum(np.conjugate(uu)*uu)
#     uu = uu/np.sqrt(np.sum(np.conjugate(uu)*uu))
#     vv2 = vv2/np.sqrt(np.sum(np.conjugate(vv2)*vv2))
    
#     h = uu + vv2*np.quaternion(0,0,1,0)
    
#     h_Hopf = np.conjugate(h)*np.quaternion(0,1,0,0)*h
    
#     n = quaternion.as_float_array(h_Hopf)[:,1:]*N
    
#     l = np.zeros((N+1,3))
#     for i in range(N+1):
#         if i==0:
#             l[i,:] = 0
            
#         else:
#             l[i,:] = l[i-1,:] + n[i-1,:]
    
#     Cc = l.T*lmbda
    
#     return Cc