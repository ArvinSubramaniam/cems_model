#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rank of context dependent pattern matrix
"""

from wishart_correlated_spectrum import *
from perceptron_capacity_conic import*


N = 500
M = 600

def generate_pattern_context(N,M,P,K,f=0.5):
    """
    Generates an 2N x PK matrix of (input,context) pairs
    
    N: Dimension of input space
    P: Number of stimuli
    M: Number of contexts
    f: Coding level
    
    """
    mat = np.zeros((N+M,P*K))
    for p in range(P):
        stim = generate_pm_with_coding(N,f)
        for l in range(K):
            #print("index of col",p*K + l)
            mat[:N,p*K + l] = stim
    for c in range(K):
#        print("c",c)
        cont = generate_pm_with_coding(M,f)
        for k in range(P):
            mat[N:N+M,c + K*k] = cont
            #print("index of col2",c + K*k)
            
    return mat 

def generate_pattern_context2(N,M,P,K,fp=0.5,fc=0.5,C1=0.5,C2=0.5):
    """
    Generates an 2N x PK matrix, with correlated stimuli/context
    
    N: Dimension of input space
    P: Number of stimuli
    M: Number of contexts
    f: Coding level
    
    """
    mat = np.zeros((N+M,P*K))
    patt = make_patterns(N,P,cod=fp)
    print("shape patt",patt.shape)
    patt_l = low_rank_reconst(patt,C=C1)
    print("shape patt_l",patt_l.shape)
    patt_con = make_patterns(M,K,cod=fc)
    count = 0
    for i in range(patt_l.shape[1]):
        for j in range(patt_con.shape[1]):
            print("shapes",patt_l[:,i].shape,patt_con[:,j].shape)
            conc = np.hstack((patt_l[:,i],patt_con[:,j]))
            print("conc shape",conc.shape)
            mat[:,count] = conc
            count += 1
            
    return mat 

        

