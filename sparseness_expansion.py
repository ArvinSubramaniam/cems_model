#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sparseness and expansion - result of unimodal model
"""

from perceptron_capacity_conic import *
from random_expansion import *
from hebbian_readout import *
from dimensionality_disentanglement import *
from scipy import integrate
import itertools

gaussian_func = lambda x: (1/np.sqrt(2*np.pi))*np.exp(-0.5*x**(2))

def erf1(T):
    res = integrate.quad(gaussian_func, T, np.inf)
    return res[0]

###ORDER OF X,Y IMPORTANT!
gaussian_func_2dim = lambda y,x: (1/np.sqrt(2*np.pi))*np.exp(-0.5*x**(2)) * (1/np.sqrt(2*np.pi))*np.exp(-0.5*y**(2)) 

def lower_bound(T,ds,x):
    b = ((1-ds)*x - T)/(np.sqrt(ds*(2-ds)))
    return b

def erf_full(T,ds,f):
    res = integrate.dblquad(gaussian_func_2dim, T, np.inf, lambda x: lower_bound(T,ds,x), lambda x: np.inf)
    return 1/(f*(1-f)) * res[0]

def erf_full1(T,ds):
    res = integrate.dblquad(gaussian_func_2dim, T, np.inf, lambda x: lower_bound(T,ds,x), lambda x: np.inf)
    return res[0]


"""Generate \Delta m plots"""
def generate_delta_m(N,P,H,d_in,th,pm=True):
    n_real = 1
    len_test = int(0.2*P)
    for j in range(n_real):
        stim = make_patterns(N,P)
        labels = np.zeros(P)
        patts_test = np.zeros((N,len_test))
        labels_test = []
        ints = []
        ##CREATE TEST PATTERNS
        for n in range(len_test):#Pick test points - perturb ONE pattern randomly
            rand_int = np.random.randint(P)
            patt_typ = stim[:,rand_int]
            lbl_test = labels[rand_int]
            labels_test.append(lbl_test)
            patt_test = flip_patterns_cluster(patt_typ,d_in)
            d_in_check = compute_delta_out(patt_test,patt_typ)
            print("check d_in",d_in_check)
            patts_test[:,n] = patt_test
            ints.append(rand_int)
        
        print("before random projection")
        h,h_test = random_proj_generic_test(H,stim,patts_test,th,bool_=pm)
        print("after random projection")
        
        o_spars = 0.5*(np.sign(h)+1)
        o = np.sign(h)
        o_test = np.sign(h_test)
        f = compute_sparsity(o_spars[:,np.random.randint(P)])
        print("coding",f)
        
        o_test_spars = 0.5*(np.sign(h_test)+1)
        
        if pm==True:
            o_in = o
            o_test_in = o_test
            
        else:
            o_in = o_spars
            o_test_in = o_test_spars
        
        w_hebb = np.matmul(o_in,labels) 
        
        erf = erf1(th)
        
        #print("shape hebbian weights",w_hebb.shape)
        stabs = []
        d_outs = []
        acts_typ = np.zeros((H,len_test))
        for m in range(len_test):
            stab = labels_test[m]*np.dot(w_hebb,o_test_in[:,m])
            stabs.append(stab)
            if pm:
                d_out = compute_delta_out(o_test_in[:,m],o_in[:,ints[m]])
            else:
                d_out = (1/(2*erf*(1-erf))) * compute_delta_out(o_test_in[:,m],o_in[:,ints[m]])
            d_outs.append(d_out)
            acts_typ[:,m] = o_in[:,ints[m]]
        
        d_out_mean = np.mean(d_outs)
        d_std = np.std(d_outs)
        print("d_out_mean",d_out_mean)
        
        if pm:
            d_out_theory = 4*erf_full1(th,d_in)
        else:
            d_out_theory = erf_full(th,d_in,erf)
        
    return d_out_mean, d_std, d_out_theory, erf

def excess_over_theory(th,f):
    numer = np.exp(-th**(2))
    denom = 2*np.pi*f*(1-f)
    return numer/denom

def excess_over_no_f(th):
    numer = np.exp(-th**(2))
    denom = 2*np.pi
    return numer/denom

    
###CHECK DIMENSIONALITY VS. EO
plot_dim = False
if plot_dim:
    bool_=False
    H_list = [50,100,200]
    ths = np.linspace(0.02,2.1,10)
    #ths = [0.8]
    n_trials=1
    pr_theorys = np.zeros(len(ths))
    pr_emps = np.zeros(len(ths))
    pr_emps_dev = np.zeros(len(ths))
    fp_corr_means = np.zeros(len(ths))
    eo_means = np.zeros(len(ths))
    cods = np.zeros(len(ths))
    for i,th in enumerate(ths):
        fp_corrs = []
        pr_emps_trials = []
        pr_theorys_trials = []
        eo_trials = []
        N=100
        M=100
        P=1000
        H=1200
        for n in range(n_trials):
            stim = make_patterns(N,P)
            cont = make_patterns(M,K)
            #cont = np.zeros((M,K))
            #h = random_project_hidden_layer(stim,cont,H) - th #NOT NORMALIZED!!
            h = random_proj_generic(H,stim,thres=th)
            if bool_:
                o = np.sign(h)
            else:
                o = 0.5*(np.sign(h)+1)
            o_spars = 0.5*(np.sign(h)+1)
            f = compute_sparsity(o_spars[:,np.random.randint(P)])
            o_in = o - f
            f_in = erf1(th)
            cods[i] = erf1(th)
            pr_emp, pr_theory, fp_corr = compute_pr_theory_sim(o_in,th,N,pm=bool_)
            eo = excess_over_theory(th,f_in)
            eo_in = (eo**(2))/(N)
            pr_theory_eo = 1/(1/(H*P) + (1/P) + (1/H) + eo_in)
            pr_theory_eo2 = 1/(eo_in)
            print("pr_emp",pr_emp)
            print("pr_theory",pr_theory)
            print("pr_theory_eo",pr_theory_eo)
            pr_emps_trials.append(pr_emp)
            pr_theorys_trials.append(pr_theory_eo)
            fp_corrs.append(fp_corr)
            eo_trials.append(eo_in)
    
        pr_emp_mean = np.mean(pr_emps_trials) 
        pr_emp_std = np.std(pr_emps_trials) 
        pr_theory_mean = np.mean(pr_theorys_trials)
        fp_corr_mean = np.mean(fp_corrs)
        eo_mean = np.mean(eo)
        print("averaged correlation",fp_corr_mean)
        print("eo_mean",eo_mean)
        
        pr_theorys[i] = pr_theory_mean
        pr_emps[i] = pr_emp_mean
        pr_emps_dev[i] = pr_emp_std
        fp_corr_means[i] = fp_corr_mean
        eo_means[i] = eo_mean
        
        
    plt.figure()
    plt.title(r'$\mathcal{D}$ vs. $Q^{2} - $unimodal$ ($P=1000$, $H=1200$)$',fontsize=12)
    colors = itertools.cycle(('blue','red','black'))
    colors_ver = itertools.cycle(('lightskyblue','lightcoral','grey'))
    clr = next(colors)
    clr_theory = next(colors_ver)
    plt.errorbar(eo_means,pr_emps,yerr = pr_emps_dev,color=clr,fmt='s', 
                 capsize=5, markeredgewidth=2)
    plt.plot(eo_means,pr_theorys,'--',color=clr_theory)
    #plt.plot(cods,fp_corr_means,'s',color=clr)
    #plt.plot(cods,eo_means,'s-',color=clr)
    plt.ylabel(r'$\mathcal{D}$',fontsize=14)
    plt.xlabel(r'$Q^{2}$',fontsize=14)
    plt.legend()
    plt.show()
    #
    #
    plt.figure()
    plt.title(r'$Q^{2}$ vs.$T$ - $unimodal$ ($P=1000$, $H=1200$)',fontsize=12)
    colors = itertools.cycle(('blue','red','black'))
    colors_ver = itertools.cycle(('lightskyblue','lightcoral','grey'))
    clr = next(colors)
    clr_theory = next(colors_ver)
    #plt.errorbar(eo_means,pr_emps,yerr = pr_emps_dev,color=clr,fmt='s', 
                 #capsize=5, markeredgewidth=2)
    #plt.plot(cods,fp_corr_means,'s',color=clr)
    plt.plot(cods,eo_means,'s-',color=clr)
    plt.ylabel(r'$Q^{2}$',fontsize=14)
    plt.xlabel(r'$f$',fontsize=14)
    plt.legend()
    plt.show()
    
gaussian_func_2dim_easy = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (0)**(2))))  \
                                        *np.exp(-(1./(2*(1 - (0)**(2))))*(x**(2) + y**(2) - 2*(0)*x*y))
                                        
gaussian_func_2dim_extra = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (0)**(2))))*x*y  \
                                        *np.exp(-(1./(2*(1 - (0)**(2))))*(x**(2) + y**(2) - 2*(0)*x*y))                                        

def two_pt_easy(th):
    """Should give f^(2)"""
    res = integrate.dblquad(gaussian_func_2dim_easy, th, np.inf, lambda x: th, lambda x: np.inf)
    return res[0]

def eo_numerical(th):
    """Should give same as theory"""
    res = integrate.dblquad(gaussian_func_2dim_extra, th, np.inf, lambda x: th, lambda x: np.inf)
    return res[0]



"""Compute readout error & SNR """
def compute_err_and_snr(N,P,H,d_in,th):
    """Always {0,1}"""
    n_real = 50
    errors = np.zeros(n_real)
    len_test = P
    for j in range(n_real):
        stim = make_patterns(N,P)
        labels = np.zeros(P)
        for i in range(P):
            labels[i] = make_labels(0.5)
        
        patts_test = np.zeros((N,len_test))
        labels_test = labels
        ints = np.arange(P)
        ##CREATE TEST PATTERNS
        for n in range(len_test):#Pick test points - perturb ONE pattern randomly
            patt_typ = stim[:,n]
            patt_test = flip_patterns_cluster(patt_typ,d_in)
            d_in_check = compute_delta_out(patt_test,patt_typ)
            patts_test[:,n] = patt_test

        h,h_test = random_proj_generic_test(H,stim,patts_test,th)
        
        o_spars = 0.5*(np.sign(h)+1)
        o = np.sign(h)
        o_test = np.sign(h_test)
        f = compute_sparsity(o_spars[:,np.random.randint(P)])
        print("coding",f)
        
        o_test_spars = 0.5*(np.sign(h_test)+1)

        o_in = o_spars - f
        o_test_in = o_test_spars - f
        
        w_hebb = np.matmul(o_in,labels) 
        
        stabs = []
        d_outs = []
        acts_typ = np.zeros((H,len_test))
        f_in = erf1(th)
        for m in range(len_test):
            stab = labels_test[m]*np.dot(w_hebb,o_test_in[:,m])
            stabs.append(stab)
            d_out = (1/(2*f_in*(1-f_in))) * compute_delta_out(o_test_in[:,m],o_in[:,ints[m]])
            d_outs.append(d_out)
            acts_typ[:,m] = o_in[:,ints[m]]
        
        d_out_mean = np.mean(d_outs)
        d_std = np.std(d_outs)
        print("d_out_mean",d_out_mean)
        
        d_out_theory = erf_full(th,d_in,f_in)
        print("d_out theory",d_out_theory)
        
        err = (len(np.where(np.asarray(stabs)<0)[0]))/(len(stabs))
        errors[j] = err
        
    err_mean = np.mean(errors)
    err_std = np.std(errors)
  
    numer_theory = (1 - d_out_theory)
    denom_theory = P/H + (P/N) * excess_over_theory(th,f_in)**(2)
    print("excess over before divided by f",(1/N)*excess_over_theory(th,f_in)**(2) * (f_in*(1-f_in))**(2))
  
    snr_theory = (numer_theory**(2))/denom_theory
    err_theory = erf1(np.sqrt(snr_theory))

    return err_mean, err_std, err_theory, f


