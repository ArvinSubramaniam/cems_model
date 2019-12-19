#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Litwin-Kumar et.al simulations
"""

from sparseness_expansion import *
from dimensionality_disentanglement import *


##CHECK PR MEASURE FOR VERY SPARSE ACTIVATIONS - SPARSE CODING NEEDS VERY HIGH P!
#N=100
#P=100
#H=200
#th=2.5
#stim = make_patterns(N,P)
#h = random_proj_generic(H,stim,thres=th)
#o = 0.5*(np.sign(h) + 1)
#rand_int = np.random.randint(P)
#print("o",o)
#f = compute_sparsity(o[:,rand_int])
#print("sparsity",f)
#
#pr_emp, pr_t, fp1, fp2 = compute_pr_theory_sim(o,th)
#print("pr_emp",pr_emp)
#print("pr_theory",pr_t)


def compute_distance_squared(o,o_test,pm=True):
    d = o - o_test
    sq = np.dot(d,d)
    if pm:
        denom = 2*o.shape[0]
    else:
        deom = o.shape[0]
        
    return sq/denom


"""TRAIN HEBBIAN WEIGHTS AND TEST ON NON-LINEAR LAYER"""
def predict_class_error(N,P,H,th,delta_in,pm=True):
    stim = make_patterns(N,P)
    labels = np.zeros(P)
    for i in range(P):
        labels[i] = make_labels(0.5)
    
    ##CREATE TEST PATTERN
    rand_int = np.random.randint(P)
    patt_typ = stim[:,rand_int]
    patt_test = flip_patterns_cluster(patt_typ,delta_in)
    h, h_test = random_proj_generic_test(H,stim,patt_test,thres=th)
    if pm:
        o = np.sign(h)
    else:
        o = 0.5*(np.sign(h) + 1)
    o_spars = 0.5*(np.sign(h) + 1)
    o2 = o_spars
    f = compute_sparsity(o_spars[:,rand_int])
    print("sparsity of training",f)
    
    #w_hebb = learn_w_hebb(o,labels)
    w_hebb = learn_w_hebb(h,labels)
    
    
    o_test = np.sign(h_test)
    o_test_spars = 0.5*(np.sign(h_test) + 1)
    o_test2 = o_test_spars
    f_test = compute_sparsity(o_test_spars)
    print("sparsity of test",f_test)
    
    patt_label = labels[rand_int]
    #stab = patt_label*np.matmul(w_hebb,o_test)
    stab = patt_label*np.matmul(w_hebb,h_test)
    dist = compute_distance_squared(o[:,rand_int],o_test)
        
    return stab, o, o_test, dist, rand_int


#N=100
#P=100
#H=200
#th=1.8
#d_in = 0.0
#bool_=True
#stabs = []
#n_trials = 50
#for i in range(n_trials):
#    stab,o,o_test,dist, rand_int = predict_class_error(N,P,H,th,d_in,pm=bool_)
#    print("stability is",stab)
#    print("distance is",dist)
#    stabs.append(stab)
#err = (len(np.where(np.asarray(stabs)<0)[0]))/(n_trials)
#print("EMPIRICAL ERROR is",err)
#err_theory = compute_readout_error_numerics(o,o_test,rand_int)
#print("theoretical error is",err_theory)


"""Reproduce readout error sims - but for random weights"""
#thres_list = [0.5,1.0,1.5,2.0,2.5]
#N=100
#P=100
#H=200
#prob_negs = []
#err_theory = []
#codings = []
#for i,th in enumerate(thres_list):
#    stabs = []
#    for i in range(100):
#        stab,o_last, dist= predict_class_error(N,P,H,th)
#        stabs.append(stab)
#
#    len_neg = len(np.where(np.asarray(stabs)<0)[0])
#    len_full = len(stabs)
#    prob_neg = len_neg/len_full
#    print("misclass prob",prob_neg)
#    prob_negs.append(prob_neg)
#    
#    rand_int = np.random.randint(P)
#    cod = compute_sparsity(o_last[:,rand_int])
#    print("sparsity is",cod)
#    codings.append(cod)
#
#    pr_emp, pr_t, fp1, fp2 = compute_pr_theory_sim(o_last,th)
#    snr = (pr_emp*(1-dist)**(2))/P
#    err = erf1(np.sqrt(snr))
#    print("theoretical error is",err)
#    err_theory.append(err)
#    
#plt.figure()
#plt.title(r'Litwin-Kumar, et. al setup, $P={}$'.format(P),fontsize=14)
#plt.plot(codings,prob_negs,'bo',label=r'Empirical')
#plt.plot(codings,err_theory,'--',color='blue',label=r'Theory from Eq. 6')
#plt.xlabel(r'$f$',fontsize=12)
#plt.ylabel(r'$\epsilon$',fontsize=12)
#plt.legend()
#plt.show()


"""SIMPLEST POSSIBLE HEBBIAN LEARNING"""
def simplest_possible_hebbian(N,P):
    stabs = []
    for i in range(100):
        stim = make_patterns(N,P)
        labels = np.zeros(P)
        for i in range(P):
            labels[i] = make_labels(0.5)
            #labels[i] = +1
        
        w_hebb = np.matmul(stim,labels) 
        ##CREATE TEST PATTERN
        rand_int = np.random.randint(P)
        patt_typ = stim[:,rand_int]
        lbl_test = labels[rand_int]
        #patt_test = flip_patterns_cluster(patt_typ,0.9)
        patt_test = patt_typ
        print("patt_typ",patt_typ)
        dist = compute_distance_squared(patt_typ,patt_test) 
        print("distance is",dist) 
        
        print("shape",w_hebb.shape)
        #lbl_test = 1
        
        stab = lbl_test*np.dot(w_hebb,patt_test)
        print("stab is",stab)
        
        stabs.append(stab)
        
    
    err = (len(np.where(np.asarray(stabs)<0)[0]))/(len(stabs))
    print("error is",err)
    
    four_point_corr = compute_order_param3(stim)
    print("four point corr is",four_point_corr)
    snr = 1/(P*(four_point_corr + 1/N))
    err_theory = erf1(np.sqrt(snr))
    print("theoretical error is",err_theory)
    
    return err, err_theory


#P_list = [60,80,100,125,150]
#err_emps = []
#err_theorys = []
#for i,P in enumerate(P_list):
#    err, err_theory = simplest_possible_hebbian(N,P)
#    err_emps.append(err)
#    err_theorys.append(err_theory)
#
#plt.figure()
#plt.title(r'Simplest possible Hebbian learning, $\Delta_{test}=0$',fontsize=12)
#plt.plot(P_list,err_emps,'s',markersize=8,label=r'Emprirical')
#plt.plot(P_list,err_theorys,'--',label=r'Theory')
#plt.xlabel(r'$P$',fontsize=10)
#plt.ylabel(r'$\epsilon$',fontsize=10)
#plt.show()


