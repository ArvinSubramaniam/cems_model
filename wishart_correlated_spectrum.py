#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rank and spectrum of correlated Wishart matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='white')
from numpy import linalg as LA


def compute_spectrum(N,M):
    mat = np.random.normal(0,1/np.sqrt(M),(N,M))
    wish = np.matmul(mat,mat.T)
    eigs = LA.eigvals(wish)
    return eigs

def generate_pm(N):
    v = np.zeros(N)
    for i in range(N):
        if np.random.rand() > 0.5:
            v[i] = 1
        else:
            v[i] = -1
    return v

def generate_pm_with_coding(N,f):
    """f: Fraction of plus ones"""
    v = np.zeros(N)
    for i in range(N):
        if np.random.rand() > 1-f:
            v[i] = 1
        else:
            v[i] = -1
            #v[i] = 0
    return v

def compute_spectrum_hopfield(N,M,f,coding=False):
    mat = np.zeros((N,M))
    for i in range(M):
        if coding:
            vec = generate_pm_with_coding(N,f)
        else:
            vec = generate_pm(N)
        mat[:,i] = vec
        
    wish = (1/M)*np.matmul(mat,mat.T)
    eigs = LA.eigvals(wish)
    return eigs


"""Spectrum"""


#eigss = []
#for i,a in enumerate(alphas):
#    M = a*N
#    eigss.append(compute_spectrum_hopfield(N,int(M),0.5,coding=True))
#    
#plt.figure()
##plt.title(r'Wishart matrix with varying coding level, $\alpha = {}$'.format(1.2))
#plt.title(r'Regular Hopfield matrix with varying $\alpha$')
#for i,a in enumerate(alphas):
#    sns.distplot(eigss[i],hist=False, kde=True,label=r'$\alpha={}$'.format(a))
#plt.legend()
#plt.xlim(-1,8)
#plt.show()


"""Rank"""

#M = 600
#ranks = []
#
#codings = [0.001,0.005,0.02,0.95,0.98,0.998]
#codings.extend(list(np.linspace(0.1,0.8,8))[:])
#for c in codings:
#    mat = np.zeros((N,M))
#    for i in range(M):
#        mat[:,i] = generate_pm_with_coding(N,c)
#    ranks.append(LA.matrix_rank(mat))
#    
#plt.figure()
##plt.title(r'Wishart matrix with varying coding level, $\alpha = {}$'.format(1.2))
#plt.title(r'Rank of Hopfield matrix ($\alpha=1.2$) for varying $f$')
#plt.plot(codings,np.asarray(ranks)/500,'bo')
#plt.legend()
#plt.show()
