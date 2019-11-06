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
from numpy import linalg as LA
import random
import scipy as sp
from scipy.optimize import linprog
from scipy.special import comb
from scipy.special import binom
import seaborn as sns
import itertools
from matplotlib.lines import Line2D


def generate_pattern_context2(stim,cont):
    """
    Generates an 2N x PK matrix of (input,context) pairs GIVEN stim,cont
    
    """
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


def random_project_hidden_layer(N,M,P,K,H,sc=1,sp=1):
    """
    Creates contextual inputs and randomly projects them onto hidden layer.
    Draw matrices RANDOMLY FOR EACH P AND K
    Gets dimensionality via the rank of h
    
    Params:
        H: Number of units in mixed layer
        sp: Variance in pattern projection
        sc: Variance in context projection
    """
    h = np.zeros((H,P*K))
    stim = make_patterns(N,P)
    cont = make_patterns(M,K)
    
    #matc = np.random.normal(0,sc/M,(int(H/2),M))
    #matp = np.random.normal(0,sp/N,(int(H/2),N))
    
    for i in range(P):
        matp = np.random.normal(0,sp/N,(int(H/2),N))
        h_stim = np.matmul(matp,stim[:,i])
        #print("h_stim",h_stim)
        #print("dimension h_stim",h_stim.shape)
        for j in range(K):
            matc = np.random.normal(0,sc/M,(int(H/2),M))
            h_cont = np.matmul(matc,cont[:,j])
            #print("h_cont",h_cont)
            #print("dimension h_cont",h_cont.shape)
            h_in = np.vstack((h_stim.reshape(int(H/2),1),h_cont.reshape(int(H/2),1))).reshape(H)
            #print("index",K*i + j)
            h[:,K*i + j] = h_in
            
    dim = LA.matrix_rank(h)
    #print("dimensioality of h is",dim)
            
    return h, dim



def output_mixed_layer(h,thres=0.):
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
        g[:,i] = np.sign(h[:,i] - t_arr)
        num = len(np.where(g[:,i]>0.)[0])
        cod = num/(len(g[:,i]))
        cods.append(cod)
        
    cod = np.round(np.mean(cods),2)
    cod_std = np.round(np.std(cods),2)

    #print("coding mean",cod,"coding std",cod_std)
    
    return g, cod, cod_std

    

def compute_participation_ratio(h_in):
    """
    Computes participation ratio of mixed layer rep.
    """
    h = h_in.reshape(h_in.shape[0],1)
    #print("h shape",h.shape)
    cov = np.matmul(h,h.T)
    #print("cov shape",cov.shape)
    cov2 = np.matmul(cov,cov.T)
    numer = (np.matrix.trace(cov))**(2)
    denom = np.matrix.trace(cov2)
    
    pr = numer/denom
    
    return pr
    

####Get dimensionality as a function of number of neurons####
from scipy.spatial import distance as spd

def func_pairwise_distances(patt,type='e'):
    """
    Pairwise distances between P columns
    Type: 'e' for Euclidean, 'h' for Hamming
    """
    matrix = np.zeros((patt.shape[1],patt.shape[1]))
    for i in range(patt.shape[1]):
        p1 = patt[:,i]
        for j in range(patt.shape[1]):
            p2 = patt[:,j]
            if type == 'e':
                matrix[i,j] = (1/(patt.shape[0]))*spd.euclidean(p1,p2)
            else:
                matrix[i,j] = (1/(patt.shape[0]))*spd.hamming(p1,p2)
                
    return matrix

def similarity_measure(x,y):
    """
    Takes the MDS distance by computing Frobenius norm between xTx and yTy
    """
    gram1 = np.matmul(x.T,x)
    gram2 = np.matmul(y.T,y)
    diff = gram1 - gram2
    dist = 0.5*LA.norm(diff)
    
    return dist

N=500
M=500
P=10
K=10
H=100
stim = make_patterns(N,P)
cont = make_patterns(M,K)
patt_c = generate_pattern_context2(stim,cont)
print("dim stim",LA.matrix_rank(patt_c))
h, dim = random_project_hidden_layer(N,M,P,K,H)
print("dim activations",LA.matrix_rank(h))
out, cod, cod_std = output_mixed_layer(h)
print("dim outputs",LA.matrix_rank(out))


#N=500
#M=500
#P=10
#K=10
#ratios = [0.1,0.2,0.4,0.8,1.0]
#dist1 = []
#dist1_err = []
#dist2 = []
#dist2_err = []
#dist3 = []
#dist3_err = []
#n_real = 50
#
#for i,r in enumerate(ratios):
#    H = int(r*N)
#    dists1 = []
#    dists2 = []
#    dists3 = []
#    for n in range(n_real):
#        stim = make_patterns(N,P)
#        cont = make_patterns(M,K)
#        patt_c = generate_pattern_context2(stim,cont)
#        h, dim = random_project_hidden_layer(N,M,P,K,H)
#        out, cod, cod_std = output_mixed_layer(h)
#        dists1.append(similarity_measure(patt_c,h))
#        dists2.append(similarity_measure(h,out))
#        dists3.append(similarity_measure(out,patt_c))
#        
#    dist1.append(np.mean(dists1))
#    dist1_err.append(np.std(dists1))
#    dist2.append(np.mean(dists2))
#    dist2_err.append(np.std(dists2))
#    dist3.append(np.mean(dists3))
#    dist3_err.append(np.std(dists3))
#
#plt.figure()
#plt.title(r'Similarity measures vs expansion ratio',fontsize=18)
#plt.errorbar(ratios,dist1,yerr=dist1_err,marker='s',linestyle='-',label=r'$\frac{1}{2}||h^T h - \bar{\xi}^T \bar{\xi}||_F ^{2}$',
#             capsize=5, markeredgewidth=2)
#plt.errorbar(ratios,dist2,yerr=dist2_err,marker='o',linestyle='--',label=r'$\frac{1}{2}||h^T h - o^T o||_F ^{2}$',
#             capsize=5, markeredgewidth=2)
#plt.errorbar(ratios,dist3,yerr=dist3_err,marker='^',linestyle='dashdot',label=r'$\frac{1}{2}||o^T o - \bar{\xi}^T \bar{\xi}||_F ^{2}$',
#             capsize=5, markeredgewidth=2)
#plt.xlabel(r'$\mathcal{R}$',fontsize=14)
#plt.ylabel(r'Distance measure',fontsize=14)
#plt.legend(fontsize=12)
#plt.show()



def mutual_info(x,y):
    """
    Computes mutual information between to vectors
    """
    px = np.histogram(x,density=True)[0]
    py = np.histogram(y,density=True)[0]
    px_pos = px[np.where(px!=0)[0]]
    py_pos = py[np.where(py!=0)[0]]
    lpx = np.log(px_pos)
    lpy = np.log(py_pos)
    dot1 = np.dot(px_pos,lpx)
    dot2 = np.dot(py_pos,lpy)
    
    minfo = dot1 - dot2
    
    return minfo


###REPRODUCE FUSI,BARAK PLOT
#ratios = [0.1,0.2,0.4,0.8] + list(np.linspace(1,5,5))
#thress = [0,0.08,0.1]
#cods = np.zeros(len(thress))
#cods_std = np.zeros(len(thress))
#dims = np.zeros((len(ratios),len(thress)))
#part_ratio = np.zeros((len(ratios),len(thress)))
#for i,r in enumerate(ratios):
#    for j,t in enumerate(thress):
#        H = int(r*N)
#        h, dim = random_project_hidden_layer(N,M,P,K,H)
#        print("input dimensionality",dim)
#        out, cod, cod_std = output_mixed_layer(h,thres=t)
#        print("output dimensionality",LA.matrix_rank(out))
#        part_ratio[i,j] = compute_participation_ratio(out)
#        print("participation ratio",part_ratio[i,j])
#        dims[i,j] = LA.matrix_rank(out)
#        cods[j] = cod
#        cods_std[j] = cod_std
#        
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.title(r'Dimensionality of mixed layer, $K={}$,$P={}$'.format(K,P),fontsize=16)
#for j,t in enumerate(thress):
#    
#    plt.plot(ratios,dims[:,j],linestyle='-',label=r'$f={}, \theta={}$'.format(cods[j],t))
#plt.axhline(dim_stim,linestyle='--',label=r'Rank of $\bar{\xi} = 29$')
#plt.axhline(P*K,linestyle='dashdot',label=r'Maximal rank = {}'.format(P*K))
#plt.xlabel(r'Expansion ratio, $\mathcal{R}$',fontsize=16)
#plt.ylabel(r'Dimensionality',fontsize=16)
#plt.legend(fontsize=14)
#plt.show() 


###Histogram of activations in mixed layer for different theta###
#N=100
#M=100
#P=10
#K=10
#thetas = [0,0.08,0.1]
#codings = []
#outs = {}
#H=2*N
#h, dim = random_project_hidden_layer(N,M,P,K,H)
#for i,t in enumerate(thetas):
#    out, cod, cod_std = output_mixed_layer(h,thres=t)
#    outs[i] = out
#    codings.append(cod)
#    
#
#plt.figure(i)
##ax = fig.add_subplot(111)
#plt.title(r'Threshold and coding level, $N={}$'.format(N),fontsize=16)
##plt.hist(h,label=r'$\theta = {}$'.format(thetas[0]))
#ax = sns.distplot(h[:,0])
#colors = itertools.cycle(('r', 'g', 'black','y'))
#for i,t in enumerate(thetas):
#    ax.axvline(t,color=next(colors),label=r'$\theta={}$,$f={}$'.format(t,codings[i]))
#ax.legend(fontsize=14)
#plt.show()



def func_evaluate_capacity_mixed(ratio,K_list):
    """
    Capacity as a funciton of expansion ratio
    """
    N = 100
    M=N
    #K = 1
    #len_P = 10
    len_P = 15 #For K > 1
    n_real = 5
    sucss_matrix = np.zeros((len(K_list),len_P))
    sucss_dev = np.zeros((len(K_list),len_P))
    P_list = np.linspace(0.2*N,3.5*N,len_P)
    #P_list = np.linspace(1,15,15) #For K > 1
    rank_mixed_layer = np.zeros((len(K_list),len_P))
    print("ratio",ratio)
    H = int(ratio*(N+M))
    for i,K in enumerate(K_list):
        for j,P in enumerate(P_list):
            sucs = []
            for n in range(n_real):
                patt = make_patterns(N,int(P))
                h, dim = random_project_hidden_layer(N,M,int(P),K,H,sc=1,sp=1)
                out, cod ,codpm = output_mixed_layer(h)
                rank_mixed_layer[i,j] = (1/(N+M))*LA.matrix_rank(h) ###USE LINEAR PROEJCTION
                print("scaled rank of mixed layer",rank_mixed_layer[i,j])
                #rank_mixed_layer[i] = LA.matrix_rank(out)
                w, status = perceptron_storage(out)
                if status == 0:
                    sucs.append(1)
                else:
                    sucs.append(0)
            print("number in matrix",np.mean(sucs))
            sucss_matrix[i,j] = np.mean(sucs)
            sucss_dev[i,j] = np.std(sucs)

    fig = plt.figure()
    plt.suptitle(r'$\mathcal{R}=1$')
    ax = fig.add_subplot(121)
    ax.set_title(r'Capacity,N={},K={}'.format(N,K),fontsize=16)
    for i,K in enumerate(K_list):
        ind_up = np.where(sucss_matrix[i,:] + sucss_dev[i,:] >= 1.)[0] #Check if it's too high
        sucss_dev[i,ind_up] = np.ones(len(ind_up)) - sucss_matrix[i,ind_up]
        
        ind_down = np.where(sucss_matrix[i,:] - sucss_dev[i,:] <= 0.)[0]#Check if it's too low
        sucss_dev[i,ind_down] = sucss_matrix[i,ind_down]
        
        ax.errorbar((1/N)*P_list,sucss_matrix[i,:],yerr=sucss_dev[i,:],marker='s',linestyle='-', capsize=5, markeredgewidth=2,
                     label=r'$K={}$'.format(K))
        ax.axvline(x=2*rank_mixed_layer[i]/(K),linestyle='dashdot',color='r',label=r'$P_c = 2/K \times c$')
    ax.axhline(y=0.5,linestyle='--',label='Prob. = 0.5')
    #plt.xlabel(r'$P$',fontsize=14)
    ax.set_xlabel(r'$\beta$',fontsize=14)
    ax.set_ylabel('Prob of success',fontsize=14)
    ax.legend(fontsize=12)
    #plt.savefig(r'{}/capacity_curve_errorbars_K={}.png'.format(path,K))
    
    ax2 = fig.add_subplot(122)
    ax2.set_title(r'$raank(h^T h)$ vs. $P$. $K={}$'.format(K))
    for i,K in enumerate(K_list):
        ax2.plot((1/N)*P_list,rank_mixed_layer[i,:],marker='o',capsize=8)
    ax2.set_xlabel(r'$\beta$',fontsize=14)
    ax2.set_ylabel('$rank(h^T h)',fontsize=14)
    plt.show()



###RUN WITH R=1 FOR DIFFERENT K
ratio=1
K_list = [2]
func_evaluate_capacity_mixed(ratio,K_list)







