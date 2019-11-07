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
#from cvxopt import matrix, solvers
import itertools
#import pulp
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


def random_project_hidden_layer(stim,cont,H,sc=1,sp=1):
    """
    Takes in contextual inputs and randomly projects them onto hidden layer.
    Draw matrices RANDOMLY FOR EACH P AND K
    Gets dimensionality via the rank of h
    
    Params:
        H: Number of units in mixed layer
        sp: Variance in pattern projection
        sc: Variance in context projection
    """
    P = stim.shape[1]
    K = cont.shape[1]
    N = stim.shape[0]
    M = cont.shape[0]
    h = np.zeros((H,P*K))
#    stim = make_patterns(N,P)
#    cont = make_patterns(M,K)
    
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

def random_proj_generic(H,patt):
    """
    Perform generic random projection with a H x N matrix
    """
    N = patt.shape[0]
    h = np.zeros((H,patt.shape[1]))
    for p in range(patt.shape[1]):
        wrand = np.random.normal(0,1/N,(H,N))
        h[:,p] = np.matmul(wrand,patt[:,p])
        
    return h
    


###CHECK CLASSIFICATION FOR GENERIC CORRELATED STIMULI
def make_patterns_corr(N,P,fcorr=0.7,cod=0.5):
    """
    fcorr: Underlying rank
    """
    ind_rem = random.sample(range(P), int((1-fcorr)*P))
    #print("indices",ind_rem)
    print("rank should be",int(fcorr*P))
    
    patterns_red = np.zeros((N,P))
    
    #Make a repeated pattern vector
    vec = generate_pm_with_coding(N,cod)
    #Loop over distinct pairs of (remove,replace)
    s1 = set(np.linspace(0,P-1,P))
    #print("set1",s1)
    s2 = set(ind_rem)
    #print("set2",s2)
    sdiff = s1.difference(s2)
    #print("set diff",sdiff)
    ind_others0 = list(sdiff)
    ind_others = [int(l) for l in ind_others0]
    
    for i in ind_others:
        patterns_red[:,i] = generate_pm_with_coding(N,f=cod)
        
    for i in ind_rem:
        patterns_red[:,i] = vec
     
    print("rank reduced",LA.matrix_rank(patterns_red))
    
    return patterns_red


###RANK OF MIXED LAYER FOR CONT-STIM VS. NORMAL STIM
#N=100
#M=100
#P=80
#K=2
##H_list = np.linspace(50,210,9) #NEED TO KEEP H even
#H_list = np.linspace(20,180,17) #NEED TO KEEP H even
#c = 0.5
#patt = make_patterns(N,P)
#patt_corr = make_patterns_corr(N,P,fcorr=c)
#stim = make_patterns(N,P)
#cont = make_patterns(M,K)
#patt_c = generate_pattern_context2(stim,cont)
#
#dimh_patt = []
#dimo_patt = []
#dimh_patt_c = []
#dimo_patt_c = []
#for i,H_in in enumerate(H_list):
#    H = int(H_in)
#    h = np.zeros((H,P))
#    out = np.zeros((H,P))
#    for i in range(P):
#        wrand = np.random.normal(0,1/N,(H,N))
#        h[:,i] = np.matmul(wrand,patt_corr[:,i])
#        out[:,i] = np.sign(h[:,i])
#    
#    
#    h2, dim = random_project_hidden_layer(stim,cont,H)
#    out2,cod1,cod2 = output_mixed_layer(h2)
#    
#    dimh_patt.append(LA.matrix_rank(h))
#    dimo_patt.append(LA.matrix_rank(out))
#    dimh_patt_c.append(LA.matrix_rank(h2))
#    dimo_patt_c.append(LA.matrix_rank(out2))
#
#
#plt.figure()
#plt.title(r'Dimensionality vs. $N_m$,$N={}$,$M={}$,$P={}$,$K={}$'.format(N,M,P,K))
#plt.plot(H_list,dimh_patt,'s-',label=r'$h_{\xi}, (generic, rank = 40)$ ')
#plt.plot(H_list,dimo_patt,'o-',label=r'$o_{\xi} (generic, rank = 40)$ ')
#plt.plot(H_list,dimh_patt_c,'^--',label=r'$h_{\bar{\xi}} (context-stim, rank = 81)$')
#plt.plot(H_list,dimo_patt_c,'*--',label=r'$o_{\bar{\xi}} (context-stim, rank = 81)$')
#plt.xlabel(r'$N_m$',fontsize=18)
#plt.ylabel(r'Dimensionality')
#plt.legend()
#plt.show()



###CHECK CLASSIFICATION FOR EACH LAYER
##RANDOM CORR
#N=100
#P=80
#c = 0.5
#patt = make_patterns(N,P)
#patt_corr = make_patterns_corr(N,P,fcorr=c)
#print("rank of correlated stim",LA.matrix_rank(patt_corr))
#H = 40
#h = np.zeros((H,P))
#out = np.zeros((H,P))
#for i in range(P):
#    wrand = np.random.normal(0,1/N,(H,N))
#    h[:,i] = np.matmul(wrand,patt_corr[:,i])
#    out[:,i] = np.sign(h[:,i])  
#
#print("rank of mixed layer",LA.matrix_rank(h))    
#w1,s1 = perceptron_storage(h)
#print("success",s1)

##CONTEXT-STIM
#N=100
#M=100
#P=10
#K=10
#H=60
#stim = make_patterns(N,P)
#cont = make_patterns(M,K)
#patt_c = generate_pattern_context2(stim,cont)
#print("dim stim",LA.matrix_rank(patt_c))
#print("alpha",(P*K/(N+M)))
##print("beta",P/N)
#h, dim = random_project_hidden_layer(stim,cont,H)
#w1,s1 = perceptron_storage(h)
#print("storage of linear projection",s1)
#print("dim activations",LA.matrix_rank(h))
#out, cod, cod_std = output_mixed_layer(h)
#w2,s2 = perceptron_storage(out)
#print("storage of non-linear projection",s2)


    

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

###MEASURE SIMILARITY BETWEEN SUCCESSIVE LAYERS
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
#        h, dim = random_project_hidden_layer(stim,cont,H)
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
#        h, dim = random_project_hidden_layer(stim,cont,H)
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
#h, dim = random_project_hidden_layer(stim,cont,H)
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
    Capacity as a funciton of expansion ratio.
    At the moment for linear vs. non-linear
    """
    N = 100
    M=N
    #K = 1
    #len_P = 10
    len_P = 15 #For K > 1
    n_real = 5
    sucss_matrix = np.zeros((len(K_list),len_P))
    sucss_dev = np.zeros((len(K_list),len_P))
    sucss_matrix2 = np.zeros((len(K_list),len_P))  #Of non-linear outputs
    sucss_dev2 = np.zeros((len(K_list),len_P))
#    P_list = np.linspace(0.2*N,2.0*N,len_P)
    #P_list = np.linspace(1,15,15) #For K > 1
    rank_mixed_layer = np.zeros((len(K_list),len_P))
    print("ratio",ratio)
    H = int(ratio*N)
    for i,K in enumerate(K_list):
        beta_max = 4*ratio/K
        P_list = np.linspace(0.2*N,beta_max*N,len_P)
        for j,P in enumerate(P_list):
            sucs = []
            sucs2 = []
            for n in range(n_real):
                stim = make_patterns(N,int(P))
                cont = make_patterns(M,K)
                h, dim = random_project_hidden_layer(stim,cont,H,sc=1,sp=1)
                out, cod ,codpm = output_mixed_layer(h)
                rank_mixed_layer[i,j] = (1/(N+M))*LA.matrix_rank(out) 
                print("scaled rank of mixed layer",rank_mixed_layer[i,j])
                #rank_mixed_layer[i] = LA.matrix_rank(out)
                w, status = perceptron_storage(h)
                w2, status2 = perceptron_storage(out)
                if status == 0:
                    sucs.append(1)
                else:
                    sucs.append(0)
                    
                if status2 == 0:
                    sucs2.append(1)
                else:
                    sucs2.append(0)
                    
            sucss_matrix[i,j] = np.mean(sucs)
            sucss_dev[i,j] = np.std(sucs)
            sucss_matrix2[i,j] = np.mean(sucs2)
            sucss_dev2[i,j] = np.std(sucs2)

    fig = plt.figure()
    plt.suptitle(r'$N=100,M=100,\mathcal{R}=1$',fontsize=16)
    ax = fig.add_subplot(121)
    ax.set_title(r'Capacity of $h$ - linear')
    colors = itertools.cycle(('blue','red','black'))
    colors_ver = itertools.cycle(('lightskyblue','lightcoral','grey'))
    for i,K in enumerate(K_list):
        ind_up = np.where(sucss_matrix[i,:] + sucss_dev[i,:] >= 1.)[0] #Check if it's too high
        sucss_dev[i,ind_up] = np.ones(len(ind_up)) - sucss_matrix[i,ind_up]
        
        ind_down = np.where(sucss_matrix[i,:] - sucss_dev[i,:] <= 0.)[0]#Check if it's too low
        sucss_dev[i,ind_down] = sucss_matrix[i,ind_down]
        
        ax.errorbar((1/N)*P_list,sucss_matrix[i,:],yerr=sucss_dev[i,:],color=next(colors),marker='s',linestyle='-', capsize=5, markeredgewidth=2,
                     label=r'$K={}$'.format(K))
        ax.axvline(x=2*rank_mixed_layer[i,-int(len_P/2)]*(N+M)/(N*K),linestyle='dashdot',color=next(colors_ver))
    ax.axhline(y=0.5,linestyle='--',label='Prob. = 0.5')
    #plt.xlabel(r'$P$',fontsize=14)
    ax.set_xlabel(r'$\beta$',fontsize=14)
    ax.set_ylabel('Prob of success',fontsize=14)
    ax.legend(fontsize=12)
    #plt.savefig(r'{}/capacity_curve_errorbars_K={}.png'.format(path,K))
    
    ax2 = fig.add_subplot(122)
    ax2.set_title(r'Capacity of $o$ - non-linear')
    for i,K in enumerate(K_list):
        ind_up = np.where(sucss_matrix2[i,:] + sucss_dev2[i,:] >= 1.)[0] #Check if it's too high
        sucss_dev2[i,ind_up] = np.ones(len(ind_up)) - sucss_matrix2[i,ind_up]
        
        ind_down = np.where(sucss_matrix2[i,:] - sucss_dev2[i,:] <= 0.)[0]#Check if it's too low
        sucss_dev2[i,ind_down] = sucss_matrix2[i,ind_down]
        
        ax2.errorbar((1/N)*P_list,sucss_matrix2[i,:],yerr=sucss_dev2[i,:],color=next(colors),marker='s',linestyle='-', capsize=5, markeredgewidth=2,
                     label=r'$K={}$'.format(K))
        ax2.axvline(x=2*rank_mixed_layer[i,-int(len_P/2)]*(N+M)/(N*K),linestyle='dashdot',color=next(colors_ver))
    ax2.axhline(y=0.5,linestyle='--',label='Prob. = 0.5')
    #plt.xlabel(r'$P$',fontsize=14)
    ax2.set_xlabel(r'$\beta$',fontsize=14)
    ax2.set_ylabel('Prob of success',fontsize=14)
    ax2.legend(fontsize=12)
    plt.show()



###COMARE EIGENVALUE SPECTRA OF LINEAR VS NON-LINER ACTIVATION LAYER
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#N=100
#M=100
#P=80
#K=2
#H=100
#stim = make_patterns(N,int(P))
#cont = make_patterns(M,K)
#patt_c = generate_pattern_context2(stim,cont)
#cov1 = (1/patt_c.shape[1])*np.matmul(patt_c,patt_c.T)
#u1,d1,v1 = LA.svd(patt_c)
#h, dim = random_project_hidden_layer(stim,cont,H,sc=1,sp=1)
#cov2 = (1/h.shape[1])*np.matmul(h,h.T)
#u2,d2,v2 = LA.svd(h)
#out, cod ,codpm = output_mixed_layer(h)
#cov3 = (1/out.shape[1])*np.matmul(out,out.T)
#u3,d3,v3 = LA.svd(out)
#
#
#fig = plt.figure(1)
#plt.suptitle(r'Eigenvalue spectra',fontsize=14)
#ax = fig.add_subplot(131)
#ax.set_title(r'$\bar{\xi}$')
#ax.hist(LA.eigvals(cov1))
#axins = inset_axes(ax,width="50%", height="60%", loc=1)
#img1 = axins.imshow(patt_c)
#
#ax2 = fig.add_subplot(132)
#ax2.set_title(r'$h$')
#ax2.hist(LA.eigvals(cov2))
#axins2 = inset_axes(ax2,width="50%", height="60%", loc=1)
#img2 = axins2.imshow(h)
#
#ax3 = fig.add_subplot(133)
#ax3.set_title(r'$o$')
#ax3.hist(LA.eigvals(cov3))
#axins3 = inset_axes(ax3,width="50%", height="60%", loc=1)
#img3 = axins3.imshow(out)
#
#fig.colorbar(img1,ax=ax3)
#
#plt.show()
    
    
def func_capacity_plus_theory(ratio,K_list,linear=True):
    """
    Same as above, but calculates beta_c over a few realizations and compares with theory
    """
    N = 100
    M=N
    #K = 1
    #len_P = 10
    len_P = 20 #For K > 1
    n_real = 5
    sucss_matrix = np.zeros((len(K_list),len_P))
    sucss_dev = np.zeros((len(K_list),len_P))
#    P_list = np.linspace(0.2*N,2.0*N,len_P)
    #P_list = np.linspace(1,15,15) #For K > 1
    rank_mixed_layer = np.zeros((len(K_list),len_P))
    print("ratio",ratio)
    H = int(ratio*N)
    cap_list = np.zeros(len(K_list)) #An empirical estmate of Pc for each K
    for i,K in enumerate(K_list):
        beta_max = 4*ratio/K
        P_list = np.linspace(0.2*N,beta_max*N,len_P)
        for j,P in enumerate(P_list):
            sucs = []
            sucs2 = []
            for n in range(n_real):
                stim = make_patterns(N,int(P))
                cont = make_patterns(M,K)
                h, dim = random_project_hidden_layer(stim,cont,H,sc=1,sp=1)
                out, cod ,codpm = output_mixed_layer(h)
                rank_mixed_layer[i,j] = (1/(N+M))*LA.matrix_rank(out) 
                print("scaled rank of mixed layer",rank_mixed_layer[i,j])
                #rank_mixed_layer[i] = LA.matrix_rank(out)
                if linear:
                    w, status = perceptron_storage(h)
                else:
                    w,status = perceptron_storage(out)
                if status == 0:
                    sucs.append(1)
                else:
                    sucs.append(0)
                    
            sucss_matrix[i,j] = np.mean(sucs)
            sucss_dev[i,j] = np.std(sucs)
    
        s1 = set(np.where(sucss_matrix[i,:]<=0.8)[0])
        s2 = set(np.where(sucss_matrix[i,:]>=0.2)[0])
        ind_int = list(s1.intersection(s2))
        if len(ind_int) == 0:
            print("Can't have reasonable estimate! - Choose finer points!")
        else:
            s1 = set(np.where(sucss_matrix[i,:]<=0.9)[0])
            s2 = set(np.where(sucss_matrix[i,:]>=0.1)[0])
            ind_int = list(s1.intersection(s2))
            if len(ind_int) == 0:
                raise ValueError("STILL can't have reasonable estimate! - Choose finer points!")
        ps_bet  = P_list[ind_int]
        pcrit = np.median(ps_bet)
        cap_list[i] = pcrit/N
        
    return cap_list

def cap_list_theory(ratio,K_list):
    """
    Returns theoretical estimate of capactity for each ratio (REMEMBER ratio = 0.5 X R as defined in paper)
    """
    cap_estimate = []
    for i, K in enumerate(K_list):
        cap = 2*ratio/K
        cap_estimate.append(cap)
    
    return cap_estimate
    
    

ratios=[0.5]
ratio = ratios[0]
K_list = [2,3,4,5]
cap_lists = {}
cap_lists[0] = func_capacity_plus_theory(ratio,K_list,linear=False)
cap_lists[1] = func_capacity_plus_theory(ratio,K_list,linear=True)
cap_estimates = cap_list_theory(ratio,K_list)
plt.figure()
colors = itertools.cycle(('b', 'black', 'r','y'))
plt.title(r'Capacity for different K,$\mathcal{R}=0.5$')
plt.plot(K_list,cap_lists[0],'s',color = 'b',markersize=12,label=r'Simulation, non-linear')
plt.plot(K_list,cap_lists[1],'o',color = 'black',markersize=12,label=r'Simulation, linear')    
plt.plot(K_list,cap_estimates,'--',color= 'black',markersize=12,label=r'Theory')
plt.xlabel(r'$K$',fontsize=14)
plt.ylabel(r'$\beta_{c}$',fonsize=14)
plt.legend()
plt.show()




    
    
    
    
    
    
