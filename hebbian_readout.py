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



"""SIMPLEST POSSIBLE HEBBIAN LEARNING"""
def simplest_possible_hebbian(N,P,d_in=0.0):
    n_real = 50
    errs = np.zeros(n_real)
    len_test = int(0.2*P)
    for j in range(n_real):
        stim = make_patterns(N,P)
        labels = np.zeros(P)
        for i in range(P):
            labels[i] = make_labels(0.5)
            #labels[i] = +1
        
        w_hebb = np.matmul(stim,labels) 
        
        four_point_list = []
        ##CREATE TEST PATTERN
        stabs = []
        for n in range(len_test):#Pick 20 test points
            rand_int = np.random.randint(P)
            patt_typ = stim[:,rand_int]
            lbl_test = labels[rand_int]
            patt_test = flip_patterns_cluster(patt_typ,d_in)
            
            print("shape",w_hebb.shape)
            #lbl_test = 1
            
            stab = lbl_test*np.dot(w_hebb,patt_test)
            print("stab is",stab)
            
            stabs.append(stab)
            
            #four_point_list.append(compute_order_param_mixed(stim,patt_test))
        
        err = (len(np.where(np.asarray(stabs)<0)[0]))/(len(stabs))
        errs[j] = err

    err_mean = np.mean(errs)
    err_std = np.std(errs)
    print("error is",err_mean)
    
    #four_point_corr = compute_order_param3(stim)
    #four_point_corr = np.mean(four_point_list)
    #print("four point corr is",four_point_corr)
    
    four_point_corr = 0
    
    snr = ((1-d_in)**(2))/(P*(four_point_corr + 1/N))
    err_theory = erf1(np.sqrt(snr))
    print("theoretical error is",err_theory)
    
    return err_mean, err_std, err_theory


#N=1000
#P_list = [500,625,750,875,1000,1125,1250,1375,1500]
##P_list = [500,1000,1500]
#deltas = [0.1,0.5,0.9]
##deltas = [0.1,0.3]
#err_emps_mean = np.zeros((len(deltas),len(P_list)))
#err_emps_std = np.zeros((len(deltas),len(P_list)))
#err_theorys = np.zeros((len(deltas),len(P_list)))
#for j,d in enumerate(deltas):
#    delta = d
#    for i,P in enumerate(P_list):
#        err_mean, err_std, err_theory = simplest_possible_hebbian(N,P,d_in=delta)
#        err_emps_mean[j,i] = err_mean
#        err_emps_std[j,i] = err_std
#        err_theorys[j,i] = err_theory
#    
#plt.figure()
#plt.title(r'Simplest Hebbian learning',fontsize=12)
#colors = itertools.cycle(('green','blue','red','black'))
#colors_ver = itertools.cycle(('lightgreen','lightskyblue','lightcoral','grey'))
#for j,d in enumerate(deltas):
#    clr = next(colors)
#    clr_theory = next(colors_ver)
#    print("color is",clr)
#    print("color theory is",clr_theory)
#    plt.errorbar(P_list,err_emps_mean[j,:],yerr=err_emps_std[j,:],color=clr,marker='o',
#                 capsize=5, markeredgewidth=2,label=r'Empirical,$\Delta={}$'.format(d))
#    plt.plot(P_list,err_theorys[j,:],'--',color=clr_theory)
#plt.xlabel(r'$P$',fontsize=14)
#plt.ylabel(r'$\epsilon$',fontsize=14)
#plt.legend()
#plt.show()


def erf_full(T,ds):
    res = integrate.dblquad(gaussian_func_2dim, T, np.inf, lambda x: lower_bound(T,ds,x), lambda x: np.inf)
    return res[0]

def compute_delta_out2(patt,patt_test):
    N=patt.shape[0]
    d_list = []
    for i in range(N):
        d = np.abs(patt[i] - patt_test[i])
        d_list.append(d)
    dist = np.mean(d_list)
    return dist
    

"""Hebbian learning at the mixed layer"""
def hebbian_mixed_layer(N,P,H,th,d_in=0.0):
    n_real = 50
    errors = []
    len_test = int(0.2*P)
    for j in range(n_real):
        stim = make_patterns(N,P)
        labels = np.zeros(P)
        for i in range(P):
            labels[i] = make_labels(0.5)
        
        four_point_list = []
        labels_test = []
        patts_test = np.zeros((N,len_test))
        
        ##CREATE TEST PATTERN
        stabs = []
        ints = []
        for n in range(len_test):#Pick test points
            rand_int = np.random.randint(P)
            patt_typ = stim[:,rand_int]
            lbl_test = labels[rand_int]
            labels_test.append(lbl_test)
            patt_test = flip_patterns_cluster(patt_typ,d_in)
            d_in_check = compute_delta_out(patt_test,patt_typ)
            #print("check d_in",d_in_check)
            patts_test[:,n] = patt_test
            ints.append(rand_int)
        
        h,h_test = random_proj_generic_test(H,stim,patts_test,th)
            
        o = np.sign(h)
        o_spars = 0.5*(np.sign(h)+1)
        f = compute_sparsity(o_spars[:,np.random.randint(P)])
        print("sparsity is",f)
        o_test = np.sign(h_test)
        
        w_hebb = np.matmul(o,labels) 
        #print("shape hebbian weights",w_hebb.shape)
        
        for m in range(len_test):
            stab = labels_test[m]*np.dot(w_hebb,o_test[:,m])
            #print("stab is",stab)
            stabs.append(stab)
    
            
        err = (len(np.where(np.asarray(stabs)<0)[0]))/(len(stabs))
        errors.append(err)
    
        d_outs = []
        for n in range(len_test):
            d_out = compute_delta_out(o_test[:,n],o[:,ints[n]])
            #print("d_out is",d_out)
            d_outs.append(d_out)
    
    err_mean = np.mean(errors)
    err_std = np.std(errors)
    
    d_out = np.mean(d_outs)
    print("d_out is",d_out)
    d_theory = 4*erf_full(th,d_in)
    print("d_theory is",d_theory)
    
    #four_point_corr = (1-2*erf1(th))**(2)
    four_point_corr = (1-0.5*d_theory)**(2)
    print("four point corr is",four_point_corr)
    
    
    snr = ((1-0.5*d_theory)**(2))/(P*(four_point_corr + 1/H))
    err_theory = erf1(np.sqrt(snr))
    print("theoretical error is",err_theory)
    
    return err_mean, err_std, err_theory

N=1000
P=2500
H=2000
th=1.5
delta = 0.1
err_mean, err_std, err_theory = hebbian_mixed_layer(N,P,H,th,d_in=delta)
print("error mean",err_mean,"err_std",err_std,"err_theory",err_theory)


###CHECK DISTANCE
#N=100
#P=100
#H=200
#thetas = [0.1,1.0,1.9]
#sparsities = np.zeros(len(thetas))
#d_list = np.linspace(0.1,0.9,9)
#d_theorys = np.zeros((len(thetas),len(d_list)))
#d_emps = np.zeros((len(thetas),len(d_list)))
#for k,t in enumerate(thetas):
#    th = t
#    for j,d in enumerate(d_list):
#        d_in = d
#        stim = make_patterns(N,P)
#        patts_test = np.zeros((N,10))
#        int_test = []
#        for i in range(10):
#            rand_int = np.random.randint(P)
#            int_test.append(rand_int)
#            stim_typ = stim[:,rand_int]
#            patt_test = flip_patterns_cluster(stim_typ,d_in)
#            d_check = compute_delta_out2(stim_typ,patt_test)
#            patts_test[:,i] = patt_test
#        
#        h,h_test = random_proj_generic_test(H,stim,patts_test,th)      
#        o = np.sign(h)
#        o_spars = 0.5*(np.sign(h)+1)
#        o_test = np.sign(h_test)
#        o_test_spars = 0.5*(np.sign(h)+1)
#        
#        spars = compute_sparsity(o_spars[:,np.random.randint(P)])
#        sparsities[k] = spars
#        
#        o1 = o[:,int_test[0]]
#        o1_test = o_test[:,0]
#        d_out = compute_delta_out2(o1,o1_test)
#        print("d_out is",d_out)
#        
#        d_theory = 4*erf_full(th,d_in)
#        print("d_theory is",d_theory)
#        
#        d_theorys[k,j] = d_theory
#        d_emps[k,j] = d_out
#        
#plt.figure()
#plt.title(r'Distance at mixed layer, $N=P=100$,$\mathcal{R}=2$',fontsize=12)
#colors = itertools.cycle(('green','blue','red','black'))
#colors_ver = itertools.cycle(('lightgreen','lightskyblue','lightcoral','grey'))
#for k,t in enumerate(thetas):
#    clr = next(colors)
#    clr_theory = next(colors_ver)
#    cod = np.round(sparsities[k],3)
#    plt.plot(d_list,d_emps[k,:],'o',color=clr,label=r'$f = {}$'.format(cod))
#    plt.plot(d_list,d_theorys[k,:],'--',color=clr_theory)
#plt.xlabel(r'$\Delta \xi$',fontsize=12)
#plt.ylabel(r'$\Delta C$',fontsize=12)
#plt.legend(fontsize=12)
#plt.show()


###FULL SWEEP
#N=1000
#H=2000
#P_list = [500,625,750,875,1000,1125,1250,1375,1500]
##P_list = [500,1000,1500]
#deltas = [0.1,0.5,0.9]
#th=1.0
##deltas = [0.1,0.3]
#err_emps_mean = np.zeros((len(deltas),len(P_list)))
#err_emps_std = np.zeros((len(deltas),len(P_list)))
#err_theorys = np.zeros((len(deltas),len(P_list)))
#for j,d in enumerate(deltas):
#    delta = d
#    for i,P in enumerate(P_list):
#        err_mean, err_std, err_theory = hebbian_mixed_layer(N,P,H,th,d_in=delta)
#        err_emps_mean[j,i] = err_mean
#        err_emps_std[j,i] = err_std
#        err_theorys[j,i] = err_theory
#    
#plt.figure()
#plt.title(r'Hebbian at mixed layer, $\mathcal{R}=2$, $\theta=1.0$',fontsize=12)
#colors = itertools.cycle(('green','blue','red','black'))
#colors_ver = itertools.cycle(('lightgreen','lightskyblue','lightcoral','grey'))
#for j,d in enumerate(deltas):
#    clr = next(colors)
#    clr_theory = next(colors_ver)
#    plt.errorbar(P_list,err_emps_mean[j,:],yerr=err_emps_std[j,:],color=clr,marker='o',
#                 capsize=5, markeredgewidth=2,label=r'Empirical,$\Delta={}$'.format(d))
#    plt.plot(P_list,err_theorys[j,:],'--',color=clr_theory)
#plt.xlabel(r'$P$',fontsize=14)
#plt.ylabel(r'$\epsilon$',fontsize=14)
#plt.legend()
#plt.show()




