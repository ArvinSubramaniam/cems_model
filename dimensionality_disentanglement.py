#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dimensionality and disentanglement
"""
from perceptron_capacity_conic import *
from scipy import integrate
from random_expansion import *
from perceptron_cap_fix_rank import low_rank_reconst


def flip(stim,pm=True):
    """
    Flip individual bit from {0,1} to {1,0}
    """
    if pm:
        if stim == -1:
            stim_o = 1
        elif stim == 1:
            stim_o = -1
        
    else:
        if stim == 0:
            stim_o = 1
        elif stim == 1:
            stim_o =0
    
    return stim_o


def flip_patterns_cluster(stim,var,typ=True):
    """
    Flips other members of the cluster with prob var/2
    typ: "True' if patterns are {+1,-1}
    """
    N=stim.shape[0]
    stim_out = np.zeros(N)
    for i in range(stim.shape[0]):
        #print("i={}",i)
        if np.random.rand() > 1 - var/2:
            #print("flipped!,i={}".format(i))
            #print("stim[i]}",stim[i])
            stim_out[i] = flip(stim[i],pm=typ)
        else:
            stim_out[i] = stim[i]
            
    return stim_out


def compute_sparsity(stim):
    """
    Computes sparsity of patterns given
    """
    sparsity = np.sum(stim)/(len(stim))
    
    return sparsity


def compute_diff(patt_ref,patt_other):
    """
    Computes difference between reference and other pattern. Sums over neurons
    """
    return np.sum(np.abs(patt_ref - patt_other))


def compute_delta_out(out,patt_ref):
    """
    Here, out should be one vector of test pattern
    """
    
    diff = compute_diff(out,patt_ref)
                
    return (1/out.shape[0])*diff


def compute_pr_eigvals(mat):
    """
    Computes PR of a matrix using eigenvalues
    """
    eigs = LA.eigvals(mat)
    numer = np.sum(eigs)**(2)
    eigs_denom = []
    for i in range(mat.shape[0]):
        eigs_denom.append(eigs[i]**(2))
    denom = np.sum(eigs_denom)
    
    return np.real(numer/denom)


def compute_pr_theory_sim(o,th,N,pm=True):
    """
    Args:
        f: Sparsity
    
    Returns:
        pr_emp: Empirical participation ratio
        pr_theory: Theoretical participation ratio based on 4 point corr function
        fp_corr: Four point correlation function
        
    """
    H = o.shape[0]
    P = o.shape[1]
    cov_o = np.matmul(o,o.T)
    numer = (np.matrix.trace(cov_o))**(2)
    rand_int = np.random.randint(o.shape[1])
    numer_theory = 1

    
    cov2 = np.matmul(cov_o,cov_o)
    denom = np.matrix.trace(cov2)

    
    erf = erf1(th)
    if pm:
        q1_theory = 1
        q2_theory = 1
        q3_theory = 1 - 8*(erf**(3) * (1-erf) + (1-erf)**(3) * erf)
        print("q3 is",q3_theory)
        excess_over = (16/N)*np.exp(-2*th**(2))/((2*np.pi)**(2))
        print("excess_over is",excess_over)
        q4_theory = (1-2*erf)**(2)
        
        q3_theory_in = q3_theory + excess_over
        fp_corr_theory = q3_theory_in
        
    else: 
        q1_theory = erf*(1-erf)
        q2_theory = (erf*(1-erf))**(2)
        q3_theory = (1/N) * np.exp(-2*th**(2))/((2*np.pi)**(2))
        q3_theory_in = q3_theory
        
    fp_corr_theory = q3_theory_in
    
    denom_theory1 = (1/(H*P))*(q2_theory/(q1_theory)**(2)) + 1/P + 1/H + (q3_theory_in/((q1_theory)**(2)))
    
    pr_theory=numer_theory/denom_theory1
    
    pr_emp = compute_pr_eigvals(cov_o)
    
    return pr_emp, pr_theory, fp_corr_theory


def compute_delta_out(out,patt_ref):
    """
    Here, out should be one vector of test pattern
    """
    
    diff = compute_diff(out,patt_ref)
                
    return (1/out.shape[0])*diff


def random_proj_generic_test(H,patt,test,thres,bool_=True):
    """
    Same as random_proj_generic but for both stim and test
    """
    N = patt.shape[0]
    h = np.zeros((H,patt.shape[1]))
    h_test = np.zeros((H,test.shape[1]))
    
    if bool:
        wrand = np.random.normal(0,1/np.sqrt(N),(H,N))
        patt_in = patt
        test_in = test
    else:
        wrand = np.random.normal(0,2/np.sqrt(N),(H,N))
        patt_in = patt - 0.5
        test_in = test - 0.5
        
    for p in range(patt.shape[1]):
        h[:,p] = np.matmul(wrand,patt_in[:,p]) - thres
        
    for q in range(test_in.shape[1]):
        h_test[:,q] = np.matmul(wrand,test_in[:,q]) - thres
        
    return h, h_test






