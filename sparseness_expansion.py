#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sparseness and expansion - Babadi, Sompolinksy 2014
"""

from perceptron_capacity_conic import *
from random_expansion import *
from scipy import integrate
import itertools
#import seaborn as sns


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


def generate_clustered_stim(N,P,delta_s,pm=True):
    """
    Generates clustered stimuli given delta_s
    Args:
        P: Number of patterns in cluster
        
    Returns:
        N x P cluster
    """
    patt = np.zeros((N,P))
    
    if pm:
        patt[:,0] = generate_pm_with_coding(N,f=0.5) #Central pattern
    else:
        patt[:,0] = 0.5*(generate_pm_with_coding(N,f=0.5) + 1) 
    for i in range(P-1):
        #print("patterns to be flipped",i+1)
        patt[:,i+1] = flip_patterns_cluster(patt[:,0],delta_s,typ=pm) #Flips central pattern with prob delta_s/2
    
    return patt, make_labels(0.5)


def generate_clustered_stim_context(N,M,P,delta_s,delta_c,pm=True):
    """
    Similar to above. Here, N is effectively N+M
    Args:
        delta_s/c: Spread for each of stimuli and context
    """
    patt = np.zeros((N+M,P))
    delta = M/N
    stim_c = generate_pm_with_coding(N,f=0.5)
    cont_c = generate_pm_with_coding(M,f=0.5)
    
    if pm==True:
        patt[:,0] = np.concatenate((stim_c,cont_c))
    else:
        v1 = 0.5*(stim_c + 1)
        v2 = 0.5*(cont_c + 1)
        patt[:,0] = np.concatenate((v1,v2))
    
    
    for i in range(P-1):
        patt[:N,i+1] = flip_patterns_cluster(patt[:N,0],delta_s,typ=pm) #Flips central pattern with prob delta_s/2
        patt[N:,i+1] = flip_patterns_cluster(patt[N:,0],delta_c,typ=pm)

    
    return patt, make_labels(0.5)


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


def compute_delta_s(stim_clus,pm=True):
    """
    Computes delta_s for clustered sitmuli
    """
    diffs = np.zeros((stim_clus.shape[1],stim_clus.shape[1]))
    diffs2 = []
    for i in range(stim_clus.shape[1]):
        patt_ref = stim_clus[:,0]
        for j in range(stim_clus.shape[1]):
            if j!=i:
                patt_other = stim_clus[:,j]
                diff = compute_diff(patt_ref,patt_other) #computes sum over neurons
                diffs[i,j]=diff
                diffs2.append(diff)
    
    number_diffs_patterns = stim_clus.shape[1]*(stim_clus.shape[1]-1)/2
    diffs_av = np.sum(diffs)/(stim_clus.shape[0]*number_diffs_patterns) #Factor of two absorbed in 
    
#    print("length of diffs",len(diffs2))
#    diffs_av = np.mean(diffs2)/(stim_clus.shape[0]) #Divide by N
    
    if pm:
        diffs_av_out = 0.5*diffs_av
    else:
        diffs_av_out = diffs_av
        
    return diffs_av_out


###CHECK FORMULA FOR DELTA_F
#N=200
#M=100
#P=20
#K=5
#delta=M/N
#delta_s_list = np.linspace(0.1,0.8,8)
#delta_c_list = [0.1,0.5,0.9]
#
#delta_emps = np.zeros((len(delta_s_list),len(delta_c_list)))
#delta_theorys = np.zeros((len(delta_s_list),len(delta_c_list)))
#for i,d1 in enumerate(delta_s_list):
#    for j,d2 in enumerate(delta_c_list):
#        delta_f = (1/(1+delta))*d1 + (delta/(1+delta))*d2
#        patt_clus = generate_clustered_stim_context(N,M,P*K,d1,d2)
#        delta_emp = compute_delta_s(patt_clus)
#        delta_emps[i,j] = delta_emp
#        delta_theorys[i,j] = delta_f
#        
#colors = itertools.cycle(('blue','red','black'))
#colors_ver = itertools.cycle(('lightskyblue','lightcoral','grey'))        
#plt.figure()
#plt.title(r'Test of $\Delta S = \frac{1}{1 + \delta} \Delta \xi + \frac{\delta}{1 + \delta} \Delta \phi$',fontsize=14)
#for i,d2 in enumerate(delta_c_list):
#    clr = next(colors)
#    clr_theory = next(colors_ver)
#    plt.plot(delta_s_list,delta_emps[:,i],'s',markersize=8,color=clr,label=r'$\Delta \phi = {}$'.format(d2))
#    plt.plot(delta_s_list,delta_theorys[:,i],'--',color=clr_theory,label='Theory')
#plt.xlabel(r'$\Delta \xi$')
#plt.ylabel(r'$\Delta S_{emp}$')
#plt.legend()
#plt.show()


#def compute_delta_c_prob(stim_clus):
#    """
#    Computes the probability that the centroid neuron is 1, and that the others are 0
#    Averages this over all patterns in cluster
#    """
#    N = stim_clus.shape[0]
#    C = stim_clus.shape[1]
#    prob_ones_ref = (1/N) * (N - np.sum(stim_clus[:,0]))
#    num_zeros_others = []
#    for i in range(C-1):
#        num = (1/N) * (N - np.sum(stim_clus[:,i+1]))
#        num_zeros_others.append(num)
#    prob_num_zeros_others = np.mean(num_zeros_others)
#    
#    return prob_num_zeros_others


def random_expansion_clustered(stim_clus,H,theta=0.00,pm=True):
    """
    Args:
        stim: NxP matrices, of centroids(P==0) + noisy realizations       
    """
    N = stim_clus.shape[0]
    P = stim_clus.shape[1]
    sparsity = np.zeros(P) #Sparsity for each P patterns presented
    cort_acts = np.zeros((H,P))
    outs = np.zeros((H,P))
    
    if pm==True:
        mat_rand = np.random.normal(0,1/np.sqrt(N),(H,N))
    else:
        mat_rand = np.random.normal(0,2/np.sqrt(N),(H,N))
        
    for i in range(stim_clus.shape[1]):
        if pm==True:
            stim_in = stim_clus[:,i]
        else:
            stim_in = stim_clus[:,i] - 0.5
            
        out = np.matmul(mat_rand,stim_in)
        outs[:,i] = out
        
        if pm==True:
            cort_acts[:,i] = np.sign(out - theta)
        else:
            cort_acts[:,i] = 0.5*(np.sign(out - theta)+ 1)
        cort_acts_spars = 0.5*(np.sign(out - theta)+ 1)
        if i==10 and False:
            print("cortical activations are",cort_acts[:,i],"where non-zero",np.where(cort_acts[:,i]!=0))
        sparsity[i] = compute_sparsity(cort_acts_spars)
     
    return cort_acts, outs, sparsity


def random_expansion_two_clusters(stim,cont,H,theta=0.00,pm=True):
    """
    Same as above but for two clusters (stim and context)    
    """
    N = stim.shape[0]
    M = cont.shape[0]
    P = stim.shape[1]
    K = cont.shape[1]
    sparsity = np.zeros(P*K) #Sparsity for each P patterns presented
    cort_acts = np.zeros((H,P*K))
    outs = np.zeros((H,P*K))
    
    if pm==True:
        matbig = np.random.normal(0,1/(N+M),(H,N+M))
    else:
        matbig = np.random.normal(0,2/(N+M),(H,N+M))
    
    for i in range(P):
        if pm==True:
            stim_in = stim[:,i]
        else:
            stim_in = stim[:,i] - 0.5
            
        h_stim = np.matmul(matbig[:,:N],stim_in)
        
        for j in range(K):
            cont_in = cont[:,j] - 0.5
            out = h_stim + np.matmul(matbig[:,N:],cont_in)
            outs[:,K*i + j] = out
            if i==10 and False:
                print("cortical activations are",outs[:,K*i + j],"where non-zero",np.where(outs[:,K*i + j]!=0))
            if pm==True:
                cort_acts[:,K*i + j] = np.sign(out - theta)
            else:
                cort_acts[:,K*i + j] = 0.5*(np.sign(out - theta)+ 1)
            cort_acts_spars = 0.5*(np.sign(out - theta)+ 1)
            sparsity[K*i + j] = compute_sparsity(cort_acts_spars)
        
    return cort_acts, outs, sparsity

###CHECK PROJECTION OF TWO CLUSTERS
#delta_ss = np.linspace(0.1,0.8,8)
#out_ds = []
#out_ds_rep = []
#N = 200
#M = 100
#P = 20 #Number of clusters
#K = 20
#C = 20 #Number of patterns in a cluster
#H = 500
#ds = 0.5
#clusts = {}
#bool_ = False
#stim_clus = generate_clustered_stim(N,C,ds,pm=bool_)
#cont_clus = generate_clustered_stim(M,K,ds)
#
#delta_s_check1 = compute_delta_s(stim_clus)
#print("delta_s_check1",delta_s_check1)
#
#delta_s_check2 = compute_delta_s(cont_clus)
#print("delta_s_check2",delta_s_check2)
#
##
##Pick one representative pattern from each cluster to project
##cort_acts, hs, sparsity = random_expansion_two_clusters(stim_clus,cont_clus,H,theta=0.35)
#cort_acts, hs, sparsity = random_expansion_clustered(stim_clus,H,theta=1.5,pm=bool_)
#
#
#print("all sparsities",sparsity)
#s_av = np.mean(sparsity)
#print("sparsity out",s_av)
#print("f(1-f)",s_av*(1-s_av))
#delta_c = compute_delta_s(cort_acts,pm=bool_)
#print("delta_c",delta_c)
#delta_cf = (1/(4*s_av*(1-s_av))) * delta_c #Factor of 4 is important
#print("delta_cf",delta_cf)


"""Theoretical plot of Eq. 12"""
###ORDER OF X,Y IMPORTANT!
gaussian_func_2dim = lambda y,x: (1/np.sqrt(2*np.pi))*np.exp(-0.5*x**(2)) * (1/np.sqrt(2*np.pi))*np.exp(-0.5*y**(2)) 

def lower_bound(T,ds,x):
    b = ((1-ds)*x - T)/(np.sqrt(ds*(2-ds)))
    return b

def erf_full(T,ds,f):
    res = integrate.dblquad(gaussian_func_2dim, T, np.inf, lambda x: lower_bound(T,ds,x), lambda x: np.inf)
    #return res[0]
    return 1/(f*(1-f)) * res[0]

def erf_full_pmone(T,ds,f):
    res = integrate.dblquad(gaussian_func_2dim, T, np.inf, lambda x: lower_bound(T,ds,x), lambda x: np.inf)
    #return res[0]
    return 4 * res[0]



run_diff_theta = False
n_trials = 50
if run_diff_theta:
    bool_=True
    #thetas = [1.5,1.8,2.0,2.5,3.0,3.1,3.2]
    thetas = [1.0,1.75,2.5]
    delta_s = np.linspace(0.1,0.9,10)
    ###Plot of delta C vs delta S
    delta_cs = np.zeros((len(thetas),len(delta_s)))
    delta_cs_err = np.zeros((len(thetas),len(delta_s)))
    dc_theory = np.zeros((len(thetas),len(delta_s)))
    codings = np.zeros((len(thetas),len(delta_s)))
    for i,th in enumerate(thetas):
        for j,ds in enumerate(delta_s):
            delta_cs_trials = []
            dc_theory_trials = []
            for n in range(n_trials):
                patt_c = generate_clustered_stim(N,C,ds,pm=bool_)
                cort_acts, hs, sparsity = random_expansion_clustered(patt_c,H,theta=th,pm=bool_)
                s_av = np.mean(sparsity)
                print("sparsity out",s_av) #Average over all patterns in cluster
                codings[i,j] = s_av
                delta_cs_trials.append((1/(4*s_av*(1-s_av))) * compute_delta_s(cort_acts,pm=bool_))
                #delta_cs[i,j] = (1/(4*s_av*(1-s_av))) * compute_delta_s(cort_acts)
                #delta_cs[i,j] = compute_delta_s(cort_acts)
                #dc_theory[i,j] = erf_full(th,ds,s_av)
                dc_theory_trials.append(erf_full(th,ds,s_av))
                #dc_theory[i,j] = erf_full_pmone(th,ds,s_av)
            delta_cs[i,j] = np.mean(delta_cs_trials)
            delta_cs_err[i,j] = np.std(delta_cs_trials)
            dc_theory[i,j] = np.mean(dc_theory_trials)
        
    
    cods_out = np.mean(codings,1)
    print("codings for different theta",cods_out)
    colors = itertools.cycle(('r', 'g', 'black','y'))
    plt.figure()
    plt.title(r'$\Delta C = \sum_{i} \frac{\langle |C_{i} - \bar{C}_{i}| \rangle}{2N_c f(1-f)} $, (Fig 2A)',fontsize=12)        
    for i, th in enumerate(thetas):
        clr = next(colors)
        plt.errorbar(delta_s,delta_cs[i,:],yerr = delta_cs_err[i,:],fmt='o',color=clr,markersize=8,
                     capsize=5,markeredgewidth=2,label=r'$f={}$'.format(np.round(cods_out[i],3)))
        plt.plot(delta_s,dc_theory[i,:],'--',color=clr,markersize=10,label=r'$f={}$'.format(np.round(cods_out[i],3)))
    plt.xlabel(r'$\Delta S$',fontsize=14)
    plt.ylabel(r'$\Delta C$',fontsize=14)
    plt.legend()
    plt.show()
    
#    plt.figure()
#    plt.title(r'$\Delta C$ vs $f$, (Fig 2B)',fontsize=12)        
#    for i, ds in enumerate(delta_s):
#        plt.plot(codings[:,j],delta_cs[:,i],'o-',markersize=10,label=r'$\Delta S={}$'.format(np.round(ds,3)))
#    plt.xlabel(r'$f$',fontsize=14)
#    plt.ylabel(r'$\Delta C$',fontsize=14)
#    plt.legend()
#    plt.show()


run_diff_theta_stim_cont = False
n_trials = 50
if run_diff_theta_stim_cont:
    N=100
    M=100
    H=300
    C=20
    K=5
    P=20
    bool_=True
    #thetas = [1.5,1.8,2.0,2.5,3.0,3.1,3.2]
    #thetas = [1.0,1.75,2.5]
    th = 1.2
    delta = M/N
    delta_s = np.linspace(0.1,0.9,10)
    delta_c = [0.1,0.3,0.5]
    ###Plot of delta C vs delta S
    delta_cs = np.zeros((len(delta_c),len(delta_s)))
    delta_cs_err = np.zeros((len(delta_c),len(delta_s)))
    dc_theory = np.zeros((len(delta_c),len(delta_s)))
    codings = np.zeros((len(delta_c),len(delta_s)))
    for i,dc in enumerate(delta_c):
        for j,ds in enumerate(delta_s):
            delta_cs_trials = []
            dc_theory_trials = []
            for n in range(n_trials):
                patt_c = generate_clustered_stim_context(N,M,P*K,ds,dc,pm=bool_)
                cort_acts, hs, sparsity = random_expansion_clustered(patt_c,H,theta=th,pm=bool_)
                df = (1/(1+delta))*ds + (delta/(1+delta))*dc
                s_av = np.mean(sparsity)
                print("sparsity out",s_av) #Average over all patterns in cluster
                codings[i,j] = s_av
                delta_cs_trials.append((1/(4*s_av*(1-s_av))) * compute_delta_s(cort_acts,pm=bool_))
                #delta_cs[i,j] = (1/(4*s_av*(1-s_av))) * compute_delta_s(cort_acts)
                #delta_cs[i,j] = compute_delta_s(cort_acts)
                #dc_theory[i,j] = erf_full(th,ds,s_av)
                dc_theory_trials.append(erf_full(th,df,s_av))
                #dc_theory[i,j] = erf_full_pmone(th,ds,s_av)
            delta_cs[i,j] = np.mean(delta_cs_trials)
            delta_cs_err[i,j] = np.std(delta_cs_trials)
            dc_theory[i,j] = np.mean(dc_theory_trials)
        
    
    cods_out = np.mean(codings,1)
    print("codings for delta_c",cods_out)
    colors = itertools.cycle(('r', 'g', 'black','y'))
    plt.figure()
    plt.title(r'$\Delta C$ for stim-context pair,$f={}$'.format(np.round(cods_out[0],3)),fontsize=12)        
    for i, dc in enumerate(delta_c):
        clr = next(colors)
        plt.errorbar(delta_s,delta_cs[i,:],yerr = delta_cs_err[i,:],fmt='o',color=clr,markersize=8,
                     capsize=5,markeredgewidth=2,label=r'$\Delta \phi={}$'.format(dc))
        plt.plot(delta_s,dc_theory[i,:],'--',color=clr,markersize=10,label=r'Theory')
    plt.xlabel(r'$\Delta \xi$',fontsize=14)
    plt.ylabel(r'$\Delta C$',fontsize=14)
    plt.legend()
    plt.show()
    
#    plt.figure()
#    plt.title(r'$\Delta C$ vs $f$, (Fig 2B)',fontsize=12)        
#    for i, ds in enumerate(delta_s):
#        plt.plot(codings[:,j],delta_cs[:,i],'o-',markersize=10,label=r'$\Delta S={}$'.format(np.round(ds,3)))
#    plt.xlabel(r'$f$',fontsize=14)
#    plt.ylabel(r'$\Delta C$',fontsize=14)
#    plt.legend()
#    plt.show()


"""Train with Hebbian weights, test and compute readout error"""
def learn_w_hebb(stim,labels):
    """
    Learns Hebbian weights with random stimuli and labels
    """
    N = stim.shape[0]
    print("stim shape",stim.shape)
    print("labels shape",labels.shape)
    w_hebb = (1/np.sqrt(N))*np.matmul(stim,labels)
    
    return w_hebb


    
"""Test the readout error from training with centroids"""
#N_list = [500,1000,1500,2000]
#N=200
#delta_ins = [0.01,0.1,0.2]
#codings = []
#thress = [1.2,1.5,1.8,1.9,2.0,2.1,2.3,2.5]
#errors = np.zeros((len(delta_ins),len(thress)))
#for j,th in enumerate(thress):
#    stabs = []
#    n_trials=100
#    for k,d_in in enumerate(delta_ins):
#        for i in range(n_trials):
#            P=50
#            H=int(3*N)
#            C=100
#            ds=d_in
#            bool_=True
#            stims = {}
#            outs = {}
#            centroids = np.zeros((N,C))
#            centroids_out = np.zeros((H,C))
#            labels = np.zeros(C)
#            sparsities = np.zeros(C)
#            for c in range(C):
#                stim,lbl = generate_clustered_stim(N,P,ds,pm=bool_)
#                delta_s_check = compute_delta_s(stim)
#                #print("check delta_s",delta_s_check)
#                labels[c] = lbl
#                #print("label is",lbl,"for cluster",c)
#                #print("theta is",th)
#                cort_acts, hs, sparsity = random_expansion_clustered(stim,H,theta=th,pm=bool_)
#                #print("sparsity is",np.mean(sparsity))
#        #        delta_c = compute_delta_s(cort_acts)
#        #        print("check delta_c",delta_c)
#                stims[c] = stim
#                outs[c] = cort_acts
#                centroids[:,c] = stim[:,0]
#                centroids_out[:,c] = cort_acts[:,0]
#                sparsities[c] = np.mean(sparsity)
#                
#            
#            w_hebb = learn_w_hebb(centroids_out,labels)
#            print("shape w_hebb",w_hebb.shape)
#            w_rand = np.random.normal(0,1/np.sqrt(N),(H,N))
#            c_test = np.random.randint(C) #Pick random cluster
#            lbl_test = labels[c_test]
#            #for p in range(P):
#            out_test = outs[c_test][:,np.random.randint(P)] #Pick a random output of cluster
#            stab = lbl_test*np.matmul(w_hebb,out_test)
#            stabs.append(stab)
#            print("stability is",stab)
#            print("sparsity is",sparsities[0])
#        
#        err = (len(np.where(np.asarray(stabs)<0)[0]))/(len(stabs))
#        errors[k,j] = err
#        print("EMPIRICAL ERROR is",err,"N=",N)
#        
#    codings.append(sparsities[0])
#
#plt.figure()
#for k,d_in in enumerate(delta_ins):
#    plt.plot(codings,errors[k,:],'s',markersize=8,label='$\Delta S={}$'.format(d_in))
#plt.xlabel(r'$f$',fontsize=12)
#plt.ylabel(r'$\epsilon$',fontsize=12)
#plt.legend()
#plt.show()
        