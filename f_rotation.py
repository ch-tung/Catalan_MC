# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:26:12 2021
rotation
@author: CHTUNG
"""
import numpy as np

def rotation(O,a):
    phi_q = 2*(np.random.rand(1)-0.5)*np.pi
    delta = 0
    theta_q = np.sqrt(-np.log(1-np.random.rand(1)*(1-delta))*2/a)/2
    # ----------------------------------------
    # theta = 2*theta_q = sqrt(-ln(1-X)/a)
    # theta_q is half of the rotation angle
    # where X is a random variable in [0,1]
    #       a is the persistence length
    # ----------------------------------------
    
    # if theta_q>np.pi/3*2:
    #     theta_q = np.array([np.pi/3*2])/2
    #print(theta_q)
    sin_theta_q = np.sin(theta_q)
    
    vq = O[:,1]*np.cos(phi_q) + O[:,2]*np.sin(phi_q)
    # vq = vq/np.sqrt(np.sum(vq**2))
    qr = np.cos(theta_q);
    qi = vq[0]*sin_theta_q;
    qj = vq[1]*sin_theta_q;
    qk = vq[2]*sin_theta_q;
    # nq = np.sqrt(qr**2 + qi**2 + qj**2 + qk**2)
    # qr = qr/nq
    # qi = qi/nq
    # qj = qj/nq
    # qk = qk/nq
    
    qij = qi*qj
    qjk = qj*qk
    qik = qi*qk
    qir = qi*qr
    qjr = qj*qr
    qkr = qk*qr
    qii = qi*qi
    qjj = qj*qj
    qkk = qk*qk
    
    Rq = np.array([[1-2*(qjj+qkk), 2*(qij+qkr), 2*(qik-qjr)],
                   [2*(qij-qkr), 1-2*(qii+qkk), 2*(qjk+qir)],
                   [2*(qik+qjr), 2*(qjk-qir), 1-2*(qii+qjj)]])
    
    R = Rq[:,:,0]
    
    # Re-orthogonalize
    Rx = R[:,0]
    Ry = R[:,1]
    err = np.dot(Rx,Ry)
    Rx_ort = Rx-(err/2)*Ry
    Ry_ort = Ry-(err/2)*Rx
    #Rz_ort = np.cross(Rx_ort,Ry_ort)
    Rx_new = 0.5*(3-np.dot(Rx_ort,Rx_ort))*Rx_ort
    Ry_new = 0.5*(3-np.dot(Ry_ort,Ry_ort))*Ry_ort
    Rz_new = np.cross(Rx_new,Ry_new)
    #Rz_new = 0.5*(3-np.dot(Rz_ort,Rz_ort))*Rz_ort
    R = np.array([Rx_new, Ry_new, Rz_new]).T
    
    return R

#%%
def rotation_dihedral(O,a):
    phi_q = 2*(np.random.rand(1)-0.5)*np.pi
    theta_q = np.arctan(1/np.sqrt(a*2))
    # ----------------------------------------
    # theta = 2*theta_q
    # 2*a = [1+cos(theta)]/[1-cos(theta)]
    # where a is the persistence length
    # theta_q is half of the rotation angle
    # ----------------------------------------
    
    # if theta_q>np.pi/3*2:
    #     theta_q = np.array([np.pi/3*2])/2
    #print(theta_q)
    sin_theta_q = np.sin(theta_q)
    
    vq = O[:,1]*np.cos(phi_q) + O[:,2]*np.sin(phi_q)
    # vq = vq/np.sqrt(np.sum(vq**2))
    qr = np.cos(theta_q);
    qi = vq[0]*sin_theta_q;
    qj = vq[1]*sin_theta_q;
    qk = vq[2]*sin_theta_q;
    # nq = np.sqrt(qr**2 + qi**2 + qj**2 + qk**2)
    # qr = qr/nq
    # qi = qi/nq
    # qj = qj/nq
    # qk = qk/nq
    
    qij = qi*qj
    qjk = qj*qk
    qik = qi*qk
    qir = qi*qr
    qjr = qj*qr
    qkr = qk*qr
    qii = qi*qi
    qjj = qj*qj
    qkk = qk*qk
    
    Rq = np.array([[1-2*(qjj+qkk), 2*(qij+qkr), 2*(qik-qjr)],
                   [2*(qij-qkr), 1-2*(qii+qkk), 2*(qjk+qir)],
                   [2*(qik+qjr), 2*(qjk-qir), 1-2*(qii+qjj)]])
    
    R = Rq
    
    # Re-orthogonalize
    Rx = R[:,0]
    Ry = R[:,1]
    err = np.dot(Rx,Ry)
    Rx_ort = Rx-(err/2)*Ry
    Ry_ort = Ry-(err/2)*Rx
    #Rz_ort = np.cross(Rx_ort,Ry_ort)
    Rx_new = 0.5*(3-np.dot(Rx_ort,Rx_ort))*Rx_ort
    Ry_new = 0.5*(3-np.dot(Ry_ort,Ry_ort))*Ry_ort
    Rz_new = np.cross(Rx_new,Ry_new)
    #Rz_new = 0.5*(3-np.dot(Rz_ort,Rz_ort))*Rz_ort
    R = np.array([Rx_new, Ry_new, Rz_new]).T
    
    return R

#%%
# def rotation_stretched(O,a):
#     # quaternion
#     phi_q = 2*(np.random.rand(1)-0.5)*np.pi
#     delta = 0
#     theta_q = np.sqrt(-np.log(1-np.random.rand(1)*(1-delta))*2/a)/2
#     # ----------------------------------------
#     # theta = 2*theta_q = sqrt(-ln(1-X)/a)
#     # where X is a random variable in [0,1]
#     #       a is the persistence length
#     # ----------------------------------------
    
#     # if theta_q>np.pi/3*2:
#     #     theta_q = np.array([np.pi/3*2])/2
#     #print(theta_q)
#     sin_theta_q = np.sin(theta_q)
    
#     vq = O[:,1]*np.cos(phi_q) + O[:,2]*np.sin(phi_q)
#     # vq = vq/np.sqrt(np.sum(vq**2))
#     qr = np.cos(theta_q);
#     qi = vq[0]*sin_theta_q;
#     qj = vq[1]*sin_theta_q;
#     qk = vq[2]*sin_theta_q;
#     # nq = np.sqrt(qr**2 + qi**2 + qj**2 + qk**2)
#     # qr = qr/nq
#     # qi = qi/nq
#     # qj = qj/nq
#     # qk = qk/nq
    
#     qij = qi*qj
#     qjk = qj*qk
#     qik = qi*qk
#     qir = qi*qr
#     qjr = qj*qr
#     qkr = qk*qr
#     qii = qi*qi
#     qjj = qj*qj
#     qkk = qk*qk
    
#     Rq = np.array([[1-2*(qjj+qkk), 2*(qij+qkr), 2*(qik-qjr)],
#                    [2*(qij-qkr), 1-2*(qii+qkk), 2*(qjk+qir)],
#                    [2*(qik+qjr), 2*(qjk-qir), 1-2*(qii+qjj)]])
    
#     R = Rq[:,:,0]
    
#     # Re-orthogonalize
#     Rx = R[:,0]
#     Ry = R[:,1]
#     err = np.dot(Rx,Ry)
#     Rx_ort = Rx-(err/2)*Ry
#     Ry_ort = Ry-(err/2)*Rx
#     #Rz_ort = np.cross(Rx_ort,Ry_ort)
#     Rx_new = 0.5*(3-np.dot(Rx_ort,Rx_ort))*Rx_ort
#     Ry_new = 0.5*(3-np.dot(Ry_ort,Ry_ort))*Ry_ort
#     Rz_new = np.cross(Rx_new,Ry_new)
#     #Rz_new = 0.5*(3-np.dot(Rz_ort,Rz_ort))*Rz_ort
#     R = np.array([Rx_new, Ry_new, Rz_new]).T
    
#     return R