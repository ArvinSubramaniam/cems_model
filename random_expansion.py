#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capacity of CDP with random projections
"""

#import sys
#sys.path.append('/Users/arvingopal5794/Documents/cognitive_control/context_rank_spectra')
from fusi_barak_rank import *
from context_dep_capacity import *
from perceptron_capacity_conic import make_patterns, perceptron_storage
import numpy as np
from numpy import linalg as LA
import random
import scipy as sp
from scipy.optimize import linprog
from scipy.special import comb
from scipy.special import binom
import seaborn as sns
import itertools
import pulp
from matplotlib.lines import Line2D


def generate_pattern_context2(stim,cont):
    """
    Generates an 2N x PK matrix of (input,context) pairs GIVEN stim,cont
    
    """
    N = stim.shape[0]
    P = stim.shape[1]
    M = cont.shape[0]
    K = cont.shape[1]
    mat = np.zeros((N+M,P*K))
    for p in range(P):
        for l in range(K):
            #print("index of col",p*K + l)
            mat[:N,p*K + l] = stim[:,p]
    for c in range(K):
        for k in range(P):
            mat[N:N+M,c + K*k] = cont[:,c]
            #print("index of col2",c + K*k)
            
    return mat 

def generate_pattern_context3(stim,cont1,cont2):
    """
    Generates an 3N x PK^2 matrix of (input,context) pairs GIVEN stim,cont
    
    """
    N = stim.shape[0]
    P = stim.shape[1]
    M1 = cont1.shape[0]
    K1 = cont1.shape[1]
    M2 = cont2.shape[0]
    K2 = cont2.shape[1]
    
    mat = np.zeros((N+M1+M2,P*K1*K2))
    
    parray = np.tile(stim,(1,K1*K2))
    phi1_array = np.zeros((M1,P*K1))
    for i in range(K1):
        print("shape of cont1",cont1[:,i].reshape(M1,1).shape)
        phi1_rep = np.tile(cont1[:,i].reshape(M1,1),(1,P))
        phi1_array[:,P*i:P*i + P] = phi1_rep
        
    phi1_array_in = np.tile(phi1_array,K2)
    
    phi2_array = np.zeros((M2,P*K1*K2))
    for i in range(K2):
        phi2_rep = np.tile(cont2[:,i].reshape(M2,1),(1,P*K1))
        phi2_array[:,P*K1*i:P*K1*i + P*K1] = phi2_rep
    
    mat[:N,:] = parray
    mat[N:N+M1,:] = phi1_array_in
    mat[N+M1:N+M1+M2,:] = phi2_array
        
            
    return mat 


def generate_random_sparse_matrix(H,N,Kd,Nm=1):
    """
    Args:
        Kd: The number of input neurons of which there exists pre-synaptic arborizations
        Nm: Number of modalities
    """
    mat_out = np.zeros((H,N))
    for i in range(H):
        row = np.random.normal(0,np.sqrt(Nm)/np.sqrt(Kd),Kd)
        row0 = np.zeros(N)
        ints =  np.random.choice(N,Kd,replace=True)
        row0[ints] = row
        mat_out[i] = row0
        
    return mat_out
    

def random_project_hidden_layer(stim,cont,H,sc=1,sp=1,noise=False,noise_amp=0.1,sparse=False):
    """
    Takes in contextual inputs and randomly projects them onto hidden layer.
    Draw matrices RANDOMLY FOR EACH P AND K
    Gets dimensionality via the rank of h
    
    Params:
        H: Number of units in mixed layer
        sp: Variance in pattern projection
        sc: Variance in context projection
        noise: False (no noise) by default
        sparse: If sprase feed-forward distribution instead
    """
    P = stim.shape[1]
    K = cont.shape[1]
    N = stim.shape[0]
    M = cont.shape[0]
    h = np.zeros((H,P*K))
#    stim = make_patterns(N,P)
#    cont = make_patterns(M,K)
    
    matc = np.random.normal(0,sc/np.sqrt(M),(int(H),M))
    matp = np.random.normal(0,sp/np.sqrt(N),(int(H),N))
    
    matbig = np.random.normal(0,1/np.sqrt(N+M),(H,N+M))
    
    for i in range(P):
        #h_stim = np.matmul(matbig[:,:N],stim[:,i])
        h_stim = np.matmul(matp,stim[:,i])
        for j in range(K):
            #h_cont = np.matmul(matbig[:,N:],cont[:,j])
            h_cont = np.matmul(matc,cont[:,j])
            h_in = h_stim + h_cont
            if noise == False:
                h[:,K*i + j] = h_in
            else:
                vec_noise = np.random.normal(0,1/np.sqrt(H),H)
                h[:,K*i + j] = h_in + noise_amp*vec_noise
            
    dim = LA.matrix_rank(h)
            
    return h

def random_project_hidden_layer_3modes(stim,cont1,cont2,H,sc=1,sp=1,noise=False,noise_amp=0.1):
    """
    Random projection for Nm=3
    """
    P = stim.shape[1]
    K1 = cont1.shape[1]
    K2 = cont2.shape[1]
    
    N = stim.shape[0]
    M1 = cont1.shape[0]
    M2 = cont2.shape[0]
    h = np.zeros((H,P*K1*K2))
#    stim = make_patterns(N,P)
#    cont = make_patterns(M,K)
    
    matc = np.random.normal(0,sc/np.sqrt(M1),(int(H),M1))
    matp = np.random.normal(0,sp/np.sqrt(N),(int(H),N))
    matc2 = np.random.normal(0,sc/np.sqrt(M2),(int(H),M2))
    
    matbig = np.random.normal(0,1/np.sqrt(N+M1+M2),(H,N+M1+M2))
    
    for i in range(P):
        #h_stim = np.matmul(matbig[:,:N],stim[:,i])
        h_stim = np.matmul(matp,stim[:,i])
        for j in range(K1):
            #h_cont = np.matmul(matbig[:,N:N+M1],cont1[:,j])
            h_cont = np.matmul(matc,cont1[:,j])
            for k in range(K2):
                #h_cont2 =  np.matmul(matbig[:,N+M1:],cont2[:,k])
                h_cont2 = np.matmul(matc2,cont2[:,k])
                h_in = h_stim + h_cont + h_cont2
                #print("index",K1*K2*i + K2*j + k)
                if noise == False:
                    #print("shape h_in",h_in.shape)
                    h[:,K1*K2*i + K2*j + k] = h_in
                else:
                    vec_noise = np.random.normal(0,1/np.sqrt(H),H)
                    h[:,K1*K2*i + K2*j + k] = h_in + noise_amp*vec_noise
#            
    dim = LA.matrix_rank(h)
         
    return h


def output_mixed_layer(h,thres=0.,act="sign"):
    """
    Args:
        h: H \times PK matrix of mixed layer activations
    
    Applies heaviside non-linearity to activations
    Output used for linear check
    
    Returns output and coding level
    """
    g = np.zeros((h.shape[0],h.shape[1]))
    
    t_arr = thres*np.ones(h.shape[0])
    cods = []
    for i in range(g.shape[1]):
        if act=="sign":
            g[:,i] = np.sign(h[:,i] - t_arr)
        else: #Choose ReLU
            g[:,i] = h[:,i]*(h[:,i]>t_arr)
            print("eg of relu",g[:,i])
        num = len(np.where(g[:,i]>0.)[0])
        cod = num/(len(g[:,i]))
        cods.append(cod)
        
    cod = np.round(np.mean(cods),2)
    cod_std = np.round(np.std(cods),2)

    #print("coding mean",cod,"coding std",cod_std)
    
    return g, cod, cod_std


def run_pipeline_mixedlayer(stim,cont,H,theta=0.,annealed=False,structured=False,
                            frac_exc=0.5,w_noise=False,amp_noise=0.1):
    """
    Runs pipeline from stim-context untill non-linearities
    
    Returns:
        h: Linear activations
        out: Non-linear activations
        cod: Coding level
    """
    
    if structured:
        h, dim = struc_project_hidden_layer(stim,cont,H,sc=1,sp=1,fexc=frac_exc)
    else:
        h, dim = random_project_hidden_layer(stim,cont,H,sc=1,sp=1,noise=w_noise,noise_amp=amp_noise)
    out, cod ,codpm = output_mixed_layer(h,thres=theta)
    
    return h, out, cod


def run_pipeline_mixedlayer_multim(stim,cont,cont2,H,theta=0.,annealed=False,structured=False,
                            frac_exc=0.5,w_noise=False,amp_noise=0.1):
    """
    Runs pipeline from multimodal stimuli
    """
    
    if structured:
        h, dim = struc_project_hidden_layer_3modes(stim,cont,H,sc=1,sp=1,fexc=frac_exc)
    else:
        h, dim = random_project_hidden_layer_3modes(stim,cont,cont2,H,sc=1,sp=1,noise=w_noise,noise_amp=amp_noise)
    out, cod ,codpm = output_mixed_layer(h,thres=theta)
    
    return h, out, cod


def random_proj_generic(H,patt,thres=0.,sparse=False):
    """
    Perform generic random projection with a H x N matrix
    """
    N = patt.shape[0]
    Kd = 7
    h = np.zeros((H,patt.shape[1]))
    if sparse:
        wrand = generate_random_sparse_matrix(H,N,Kd)
    else:
        wrand = np.random.normal(0,1/np.sqrt(N),(H,N))
    patt_in = patt
    
    for p in range(patt.shape[1]):
        h[:,p] = np.matmul(wrand,patt_in[:,p]) - thres
        
    return h


###CHECK THAT SPARSE WEIGHT HAS THE SAME NORM
#N=100
#P=50
#H=2000
#stim = make_patterns(N,P)
#th=0.0
#h = random_proj_generic(H,stim,th,sparse=True)
#norm = (1/H)*np.dot(h[:,0],h[:,0])
#print("norm",norm)




