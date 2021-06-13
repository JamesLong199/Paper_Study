# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 16:53:02 2021

@author: jianz
"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time

from numpy.linalg import norm

from basic_functions import (generate_rand_vec,
                             circ_conv_1d,
                             generate_circ_matrix)



# linear operation A(Z)
def op_A(Z,A,B):
    # input
    # -- Z: real matrix      KxN
    # -- A: complex matrix   LxN
    # -- B: complex matrix   LxK
    # output
    # -- vec: vector of length L
    K,N = Z.shape
    L = A.shape[0]
    assert A.shape == (L,N) and B.shape == (L,K)
    vec = np.zeros((L,))
    for i in range(L):
        vec[i] = B[i,:] @ Z @ np.conjugate(A[i,:])
    
    return vec

# linear operation A*(z):
def op_A_star(z,A,B):
    # input
    # -- z: vector           L
    # -- A: complex matrix   LxN
    # -- B: complex matrix   LxK
    # output
    # -- mat: matrix         KxN
    L,N,K = z.shape[0],A.shape[1],B.shape[1]
    assert z.size == L
    z = z.reshape(L,)
    mat = np.zeros((K,N))
    for i in range(L):
        mat = mat + z[i]*np.outer(np.conjugate(B[i,:]),np.conjugate(A[i,:]))
    
    return mat

# derivative of G(z)    
def d_G0(z):
    # input
    # -- z: vector
    return (np.max(z-1,0))**2

# derivate of Gh
def d_Gh_func(rho,mu,d,L,B,h):
    Sum = 0
    for i in range(L):
        Sum += d_G0(L*norm(B[i,:]*h)**2)*np.outer(np.conjugate(B[i,:]),
                                                  np.conjugate(B[i,:])) @ h
        
    d_Gh = rho/(2*d) * (d_G0(norm(h)**2 / (2*d))*h + L/(4*mu**2)*Sum)
    return d_Gh
    

if __name__ == '__main__':
    L = 64
    K = 15
    N = 15
    h = generate_rand_vec(K)
    f = np.zeros((L,),dtype=complex)
    f[:K] = h
    x = generate_rand_vec(N)
    C = np.random.normal(0,1,(L,N))
    g = C @ x
    y = circ_conv_1d(f,g)
    
    
    # complex Gaussian matrix A 
    A = np.zeros((L,N),dtype=complex)
    A.real = np.random.normal(0,0.5,(L,N))
    A.imag = np.random.normal(0,0.5,(L,N))
    
    # normalized DFT matrix
    F = scipy.linalg.dft(L) / np.sqrt(L)
 
    # matrix B: first K columns of F
    B = F[:,0:K]
    B_re,B_im = B.real,B.imag
    
    
    # algorithm 1: find initial value
    A_star_y = op_A_star(y,A,B)      # KxN
    U,S,Vh = np.linalg.svd(A_star_y) # KxK,KxN,NxN
    
    d = S[0]
    h0 = U[0,:]    # K
    x0 = Vh[:,0]   # N
    sqrt_d = np.sqrt(d)
    mu = 6*np.sqrt(L/(K+N)) / np.log(L)    
    
    h0_re,h0_im = h0.real,h0.imag
    x0_re,x0_im = x0.real,x0.imag    
    
    u_re_cvx = cp.Variable(shape=(K,))
    u_im_cvx = cp.Variable(shape=(K,))

    obj = cp.Minimize(cp.sum((u_re_cvx-sqrt_d*h0_re)**2+
                      (u_im_cvx-sqrt_d*h0_im)**2))
    
    constraint = [cp.norm(((B_re*u_re_cvx-B_im*u_im_cvx)**2 + 
    (B_re*u_im_cvx+B_im*u_re_cvx)**2),'inf')*L <= 4*d*(mu**2)]
    
    prob = cp.Problem(obj,constraint)
    prob.solve()
    
    u0 = np.zeros_like(h0,dtype=complex)              # estimate h
    u0.real,u0.imag = u_re_cvx.value,u_im_cvx.value
    v0 = sqrt_d*x0                                    # estimate x
    
    # algorithm 2: gradient descent
    step = 0.01
    u,v = u0,v0
    diff_u,diff_v = 10000,10000
    # converge when relative diff <= 0.01
    while diff_u > 0.01 or diff_v > 0.01:
        
        f_ = np.zeros((L,),dtype=complex)
        f_[:K] = u
        g = C @ v
        y_ = generate_circ_matrix(f) @ g
        rho = d**2
        
        d_Fh = op_A_star((op_A(np.outer(u,v),A,B)-y_),A,B) @ v
        d_Fx = np.conjugate(op_A_star((op_A(np.outer(u,v),A,B)-y_),A,B)) @ u
        
        d_Gh = d_Gh_func(rho,mu,d,L,B,u)
        d_Gx = rho/(2*d) * d_G0(norm(v)**2 / (2*d))*v
        
        u_nxt = u - step*(d_Fh + d_Gh)
        v_nxt = v - step*(d_Fx + d_Gx)
        
        diff_u = norm(u_nxt-u) / norm(u)
        diff_v = norm(v_nxt-v) / norm(v)
        
        print('diff_u:{:.4f} diff_v{:.4f}'.format(diff_u,diff_v))
        
        u = u_nxt
        v = v_nxt
    
    f_opt = np.zeros((L,),dtype=complex)
    f_opt[:K] = u
    g_opt = C @ v
    y_opt = generate_circ_matrix(f_opt) @ g_opt
    
    err = norm(y_opt-y) / norm(y)
    print('relative error:',err)
    
    
    
    
    
    
    
    
    
    
    
    