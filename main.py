
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:23:26 2021

@author: jianz
"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time
import pickle



from MonteCarlo import (MonteCarlo_noiseless,
MonteCarlo_noisy,
MonteCarlo_violate,
MonteCarlo_nonblind,
)  
    

    

if __name__ == '__main__':
    # start_time = time.time()
    K = 12
    N = 12
    L = 64
    
    ##### Noiseless Blind Deconvolution #####
    # img = MonteCarlo_noiseless(K,N,L,5)
    # plt.imshow(img[2:,2:])
    # fname = 'MonteCarlo_noiseless'
    # outfile = open(fname,'wb')
    # pickle.dump(img,outfile)
    # outfile.close()
    
    # infile = open('MonteCarlo_noiseless','rb')
    # img = pickle.load(infile)
    
    # fig, ax = plt.subplots()
    # im = ax.imshow(img)
    
    # plt.xlabel('K')
    # plt.ylabel('N')
    # plt.title("Noiseless Blind Deconvolution")
    # plt.colorbar(im,label="Relative Error",orientation="vertical")
    
    # fig.tight_layout()
    # plt.savefig('noiseless_blind_decon')
    # plt.show()
    
    
    ##### Noisy Blind Deconvolution #####
    # err_array,SNR_array = MonteCarlo_noisy(K,N,L,end_val=80,step_size=10,rep=10)
    # plt.plot(SNR_array,err_array)
    # fname = 'MonteCarlo_noisy'
    # outfile = open(fname,'wb')
    # pickle.dump(((err_array,SNR_array)),outfile)
    # outfile.close()
    
    # infile = open('MonteCarlo_noisy','rb')
    # err_array,SNR_array = pickle.load(infile)
    # plt.plot(SNR_array,err_array)
    # plt.xlabel('SNR (dB)')
    # plt.ylabel('Relative Error')
    # plt.title('Noisy Blind Deconvolution')
    # plt.savefig('noisy_blind_decon')
    
    
    ##### Non-Diffuse Blind Deconvolution #####
    # img = MonteCarlo_violate(K,N,L,5)
    # plt.imshow(img[2:,2:])
    # fname = 'MonteCarlo_violate'
    # outfile = open(fname,'wb')
    # pickle.dump(img,outfile)
    # outfile.close()
    
    # infile = open('MonteCarlo_violate','rb')
    # img = pickle.load(infile)
    
    # fig, ax = plt.subplots()
    # im = ax.imshow(img)
    
    # plt.xlabel('K')
    # plt.ylabel('N')
    # plt.title("Non-Diffuse Blind Deconvolution")
    # plt.colorbar(im,label="Relative Error",orientation="vertical")
    
    # fig.tight_layout()
    # plt.savefig('violated_blind_decon')
    # plt.show()
    
    
    ##### Non-Blind Deconvolution #####
    # img = MonteCarlo_nonblind(K,N,L,20)
    # plt.imshow(img[2:,2:])
    # fname = 'MonteCarlo_nonblind'
    # outfile = open(fname,'wb')
    # pickle.dump(img,outfile)
    # outfile.close()
    
    infile = open('MonteCarlo_nonblind','rb')
    img = pickle.load(infile)
    
    fig, ax = plt.subplots()
    im = ax.imshow(img)
    
    plt.xlabel('K')
    plt.ylabel('N')
    plt.title("Non-Blind Deconvolution")
    plt.colorbar(im,label="Relative Error",orientation="vertical")
    
    fig.tight_layout()
    plt.savefig('non_blind_decon')
    plt.show()
    
    
    
    
    # print('used {:.3f} s'.format(time.time()-start_time))
    
    
    
    
    
    
    
    
    
    
    
    
    
    