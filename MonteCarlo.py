# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 00:12:02 2021

@author: jianz
"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time

from basic_functions import (generate_orthonormal_mat,
generate_rand_mat,
generate_rand_vec,
circ_conv_1d,
generate_noise,
generate_circ_matrix,
relative_error,
solve_cvx)

# Monte Carlo runs of the experiment
def MonteCarlo_noiseless(K,N,L,rep):
    # input
    # -- K: subspace dimension of w
    # -- N: subspace dimension of x
    # -- L: number of observation
    # -- rep: number of trials per (K,N)
    # output
    # -- img: plot of experiment result
    
    img = np.zeros((N,K))
    
    for k in range(2,K):
        for n in range(2,N):
            # if k%3 == 0 and n%3 == 0:
            #     print('k={},n={}'.format(k,n))
            # B,C are fixed
            avg_err = 0
            
            for i in range(rep):
                B = generate_orthonormal_mat(L,K)
                C = generate_rand_mat(L,N)
                h = generate_rand_vec(K)
                m = generate_rand_vec(N)
                
                w = B @ h
                x = C @ m
                y = circ_conv_1d(w,x)
                X = np.kron(m,h).reshape(K,N)
                X_opt = solve_cvx(B,C,y)
                # sig,z = generate_noise(X,60,L)
                # X_opt = solve_cvx(B,C,y,sig)
                
                err = relative_error(X,X_opt)
                avg_err += err
                
            avg_err = avg_err / rep
            
            print('k={},n={}: avg_err {}'.format(k,n,avg_err))
            img[n,k] = avg_err
    
    return img

def MonteCarlo_noisy(K,N,L,end_val,step_size,rep):
    # input
    # -- K: subspace dimension of w
    # -- N: subspace dimension of x
    # -- L: number of observation
    # -- end_val: range of SNR (dB)
    # -- step_size: step size of SNR (dB)
    # -- rep: number of trials per (K,N)
    # output
    # -- img: plot of experiment result
    
    assert step_size > 0
    assert end_val > step_size
    
    SNR = step_size
    err_list = []
    SNR_list = []
    
    while(SNR <= end_val):
        avg_err = 0
        for i in range(rep):
            B = generate_orthonormal_mat(L,K)
            C = generate_rand_mat(L,N)
            h = generate_rand_vec(K)
            m = generate_rand_vec(N)
            w = B @ h
            x = C @ m
            y = circ_conv_1d(w,x)
            X = np.kron(m,h).reshape(K,N)
            sig,z = generate_noise(X,SNR,L)
            X_opt = solve_cvx(B,C,(y+z),sig)
            
            err = relative_error(X,X_opt)
            avg_err += err
        
        avg_err = avg_err / rep
        err_list.append(avg_err)
        SNR_list.append(SNR)
        print('SNR={}dB: avg_err {}'.format(SNR,avg_err))
        SNR += step_size
    
    err_array = np.array(err_list)
    SNR_array = np.array(SNR_list)  # plot(SNR_array,err_array)
    
    return err_array,SNR_array

# violating diffuse condition of B
def MonteCarlo_violate(K,N,L,rep):
    # input
    # -- K: subspace dimension of w
    # -- N: subspace dimension of x
    # -- L: number of observation
    # -- rep: number of trials per (K,N)
    # output
    # -- img: plot of experiment result
    
    img = np.zeros((N,K))
    
    for k in range(2,K):
        for n in range(2,N):
            # violating diffuse condition of B
            avg_err = 0
            
            for i in range(rep):
                B = generate_orthonormal_mat(L,K,diffuse=False)
                C = generate_rand_mat(L,N)
                h = generate_rand_vec(K)
                m = generate_rand_vec(N)
                
                w = B @ h
                x = C @ m
                y = circ_conv_1d(w,x)
                X = np.kron(m,h).reshape(K,N)
                X_opt = solve_cvx(B,C,y)
                
                err = relative_error(X,X_opt)
                avg_err += err
                
            avg_err = avg_err / rep
            
            print('k={},n={}: avg_err {}'.format(k,n,avg_err))
            img[n,k] = avg_err
    
    return img

# use pseudo-inverse of circular convolution matrix
# to recover x directly
# assume w = Bh is known, recover x = Cm
def MonteCarlo_nonblind(K,N,L,rep):
    # input
    # -- K: subspace dimension of w
    # -- N: subspace dimension of x
    # -- L: number of observation
    # -- rep: number of trials per (K,N)
    # output
    # -- img: plot of experiment result
    
    img = np.zeros((N,K))
    
    for k in range(2,K):
        for n in range(2,N):
            # violating diffuse condition of B
            avg_err = 0
            
            for i in range(rep):
                B = generate_orthonormal_mat(L,K)
                C = generate_rand_mat(L,N)
                h = generate_rand_vec(K)
                m = generate_rand_vec(N)
                
                w = B @ h
                x = C @ m
                y = circ_conv_1d(w,x)
                
                circ_w = generate_circ_matrix(w)
                x_opt = np.linalg.pinv(circ_w) @ y
                
                err = np.linalg.norm(x_opt-x)/np.linalg.norm(x)
                avg_err += err
            
            avg_err = avg_err / rep
            print('k={},n={}: avg_err {}'.format(k,n,avg_err))
            img[n,k] = avg_err
    
    return img


        
            
            
            
            
            
    
    
    
    
    
    
    