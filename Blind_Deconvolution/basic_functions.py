# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 19:45:55 2021

@author: jianz
"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time

# generate deterministic matrix B
def generate_orthonormal_mat(L,K,random=True,diffuse=True):
    # input
    # -- L: length of observation
    # -- K: dimension of subspace
    # -- random: if select columns of identity randomly
    # -- diffuse: if matrix should be diffuse
    # output
    # -- B: orthonormal matrix B
    assert isinstance(L,int) and isinstance(K,int)
    identity = np.eye(L)
    if random == True:
        ind = np.random.choice(L,K)
        B = identity[:,ind]
    else:
        B = identity[:,0:K]
        
    # for experiment with violating diffuse condition
    # exponential energy distribution
    if diffuse == False:
        coef = np.zeros((K,))
        for i in range(K):
            if i+1 == K:
                coef[i] = 1 / 2**(K-1)
            else:
                coef[i] = 1 / 2**(i+1)
        rand_coef = np.random.choice(coef,coef.size,replace=False)
        B = B*rand_coef
        
    return B    

# generate deterministic matrix C
def generate_rand_mat(L,N):
    # input
    # -- L: length of observation
    # -- N: dimension of subspace
    # output
    # -- C: random matrix C
    assert isinstance(L,int) and isinstance(N,int)
    std = np.sqrt(1/L)
    C = np.random.normal(0,std,(L,N))
    return C

# generate standard Gaussian vectors with independent entries
def generate_rand_vec(length):
    # input
    # -- length: length of the output vector
    assert isinstance(length,int)
    return np.random.normal(0,1,(length,))


# circular convolution for 1d vector
# https://www.py4u.net/discuss/165472
def circ_conv_1d(signal,kernel):
    # input
    # -- signal: 1d numpy array of length L
    # -- kernel: 1d numpy array of length L
    # output
    # -- y: 1d numpy array of length L
    assert signal.shape == kernel.shape
    y = np.real(np.fft.ifft(np.fft.fft(signal)*np.fft.fft(kernel)))
    return y

# generate noise vector z
def generate_noise(X,SNR,length):
    # input
    # -- X: ground truth outer product
    # -- SNR: signal to noise ratio, in unit dB
    # -- length: length of observation L
    # output
    # -- z: noise vector satisfying given X and SNR
    X_F_2 = (np.linalg.norm(X,ord='fro'))**2
    # use ||z||^2 to approximate sigma^2
    sig = np.sqrt(X_F_2 / (10**(SNR/10)))
    return sig,np.random.normal(0,sig,(length,))

# generate circular convolution matrix of a vector
# assume no padding needed
def generate_circ_matrix(vec):
    # input
    # -- vec: vector of length L
    # output
    # -- circ: circulant matrix
    L = vec.shape[0]
    assert vec.size == L
    vec = vec.reshape(L,)
    circ = np.zeros((L,L))
    prev = vec
    for i in range(L):
        if i == 0:
            circ[:,i] = vec
        else:
            circ[1:,i] = prev[0:L-1]
            circ[0,i] = prev[-1]
            prev = circ[:,i]
    
    return circ
    

# calculate relative error of outer product estimate
def relative_error(X,X_opt):
    # input
    # -- X: ground truth
    # -- X_opt: algorithm output
    nom = np.linalg.norm((X_opt-X),ord='fro')
    denom = np.linalg.norm(X,ord='fro')
    # print('relative error: {:.3f}'.format(nom/denom))
    return nom/denom 

# solve convex optimization problem 
def solve_cvx(B,C,y,sig=None):
    # input
    # -- B: deterministic subspace matrix
    # -- C: random codebook matrix
    # -- y: convolution of x and w vector
    # -- sig: standard deviation of noise vector
    # output
    # -- X_opt: minimizer of the convex 
    assert B.shape[0] == C.shape[0]
    L,K,N = B.shape[0],B.shape[1],C.shape[1]
    
    # normalized DFT matrix
    F = scipy.linalg.dft(L) / np.sqrt(L)
    F_re = F.real
    F_im = F.imag
    
    C_hat_re = F_re @ C
    C_hat_im = F_im @ C
    B_hat_re = F_re @ B
    B_hat_im = F_im @ B
    
    # first try: naive approach
    A_re_list = []
    A_im_list = []
    for i in range(N):
        delta_re = C_hat_re[:,i].reshape(-1,1) * np.sqrt(L)
        delta_im = C_hat_im[:,i].reshape(-1,1) * np.sqrt(L)

        A_re_list.append(delta_re*B_hat_re - delta_im*B_hat_im)
        A_im_list.append(delta_re*B_hat_im + delta_im*B_hat_re)
    
    A_re = np.hstack(A_re_list)
    A_im = np.hstack(A_im_list)
    assert A_re.shape == (L,K*N)
    
    # second try: numpy broadcasting
    # A_re = (C_hat_re[:,:,None]*B_hat_re[:,None,:] 
    #         - C_hat_im[:,:,None]*B_hat_im[:,None,:]).reshape(L,-1)
    # A_im = (C_hat_re[:,:,None]*B_hat_im[:,None,:] 
    #         + C_hat_im[:,:,None]*B_hat_re[:,None,:]).reshape(L,-1)
    
    y_hat_re = (F @ y).real
    y_hat_im = (F @ y).imag
    
    A_re_cvx = cp.Parameter(shape=A_re.shape,value=A_re)
    A_im_cvx = cp.Parameter(shape=A_im.shape,value=A_im)
    X_cvx = cp.Variable(shape=(K,N))
    
    obj = cp.Minimize(cp.norm(X_cvx,"nuc"))
    
    # noiseless case
    if sig is None:
        constraint = [A_re_cvx @ cp.vec(X_cvx.T) == y_hat_re,
                      A_im_cvx @ cp.vec(X_cvx.T) == y_hat_im]
    # noisy case
    else:
        delta = np.sqrt(L+np.sqrt(4*L))*sig
        delta_cvx = cp.Parameter(shape=delta.shape,value=delta)
        constraint = [
            cp.norm(y_hat_re-A_re_cvx @ cp.vec(X_cvx.T))**2 + 
            cp.norm(y_hat_im-A_im_cvx @ cp.vec(X_cvx.T))**2 <= delta_cvx**2
        ]
    
    prob = cp.Problem(obj,constraint)
    prob.solve()
    X_opt = X_cvx.value
    
    return X_opt