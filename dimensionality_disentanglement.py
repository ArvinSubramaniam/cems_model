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


def flip_patterns_cluster(stim,var):
    """
    Flips other members of the cluster with prob var/2 to FORM TEST DATUM
    typ: "True' if patterns are {+1,-1}
    """
    N=stim.shape[0]
    stim_out = np.zeros(N)
    for i in range(stim.shape[0]):
        #print("i={}",i)
        if np.random.rand() > 1 - var/2:
            #print("flipped!,i={}".format(i))
            #print("stim[i]}",stim[i])
            stim_out[i] = flip(stim[i])
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
    
    wrand = np.random.normal(0,1/np.sqrt(N),(H,N))
    patt_in = patt
    test_in = test
        
    for p in range(patt.shape[1]):
        h[:,p] = np.matmul(wrand,patt_in[:,p]) - thres
        
    for q in range(test_in.shape[1]):
        h_test[:,q] = np.matmul(wrand,test_in[:,q]) - thres
        
    return h, h_test


def compute_pr_theory_sim(o,th,N,pm=False):
    """
    Args:
        f: Sparsity
        pm: If {+,-} at mixed layer instead
    
    Returns:
        pr_emp: Empirical participation ratio
        pr_theory: Theoretical participation ratio based on 4 point corr function
        fp_corr: Four point correlation function
        
    """
    H = o.shape[0]
    P = o.shape[1]
    cov_o = np.matmul(o,o.T)
    rand_int = np.random.randint(o.shape[1])
    numer_theory = 1
    
    erf = erf1(th)
    print("erf is",erf)
    if pm:
        q1_theory = 1
        q2_theory = 1
        q3_theory = 1 - 8*(erf**(3) * (1-erf) + (1-erf)**(3) * erf)
        print("q3 is",q3_theory)
        excess_over = (16/N)*np.exp(-2*th**(2))/((2*np.pi)**(2))
        print("excess_over is",excess_over)
        q3_theory_in = q3_theory + excess_over
        
    else: 
        q1_theory = erf*(1-erf)
        q2_theory = (erf*(1-erf))**(2)
        q3_theory = (1/N) * np.exp(-2*th**(2))/((2*np.pi)**(2))
        q3_theory_in = q3_theory
        
    fp_corr_theory = q3_theory_in
    
    ratio1 = q2_theory/(q1_theory)**(2)
    denom_theory1 = (1/(H*P))*(q2_theory/(q1_theory)**(2)) + 1/P + 1/H + (q3_theory_in/((q1_theory)**(2)))
    
    pr_theory = numer_theory/denom_theory1
    
    pr_emp = compute_pr_eigvals(cov_o)
    
    return pr_emp, pr_theory, fp_corr_theory


run_dimensionality = True
if run_dimensionality:
    N=100
    P=200
    H=2000
    stim = make_patterns(N,P)
    stim_test = np.zeros((N,P))
    for p in range(stim.shape[1]):
        stim_test[:,p] = flip_patterns_cluster(stim[:,p],0.1)
    thress = np.linspace(0.1,2.8,20)
    pr_emps = np.zeros(len(thress))
    pr_theorys = np.zeros(len(thress))
    fp_corrs = np.zeros(len(thress))
    cods = np.zeros(len(thress))
    for i,th in enumerate(thress):
        print("th",th)
        #h,h_test = random_proj_generic_test(H,stim,stim_test,th)
        h = random_proj_generic(H,stim)
        o = np.sign(h-th)
        o_spars = 0.5*(o + 1)
        f = compute_sparsity(o_spars[:,np.random.randint(P)])
        o_spars_in = o_spars - f
        print("f is",f)
        cods[i] = f
        #o_test = 0.5*(np.sign(h_test) + 1)
        pr_emp,pr_th,fp_corr = compute_pr_theory_sim(o_spars_in,th,N,pm=False)
        print("pr_theory",pr_th)
        print("pr_emp",pr_emp)
        pr_emps[i] = pr_emp
        pr_theorys[i] = pr_th
        fp_corrs[i] = fp_corr
        
    plt.figure()
    import matplotlib.ticker as ticker
    ax = plt.subplot(121)
    ax.set_title(r'Dimensionality',fontweight="bold",fontsize=16)
    ax.plot(fp_corrs,pr_emps,'s',markersize=8,color='blue')
    ax.plot(fp_corrs,pr_theorys,'--',color='lightblue')
    start1, end1 = ax.get_xlim()
    ax.set_ylabel(r'$\mathcal{D}$',fontsize=16)
    ax.set_xlabel(r'$\frac{\langle \mathcal{I}_{4} \rangle}{\langle q_{2} \rangle^{2}}$',fontsize=16)
    diff = fp_corrs[3] - fp_corrs[4]
    print("start,end,diff",start1,end1,diff)
    ax.set_xticks(np.arange(start1, end1, 3*diff))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2E'))
    
    ax2 = plt.subplot(122)
    ax2.set_title(r'Re-scaled interference',fontweight="bold",fontsize=16)
    ax2.plot(cods,fp_corrs,'s-',markersize=8,color='blue')
    start2, end2 = ax2.get_xlim()
    diff2 = cods[1] - cods[2]
    print("start,end,diff",start2,end2,diff2)
    #ax.plot(fp_corrs,pr_theorys,'--',color='lightblue')
    ax2.set_xlabel(r'$f$',fontsize=16)
    ax2.set_ylabel(r'$\frac{\langle \mathcal{I}_{4} \rangle}{\langle q_{2} \rangle^{2}}$',fontsize=16)
    ax2.set_xticks(np.arange(start2, end2, 3*diff2))
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2E'))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2E'))
    
    plt.tight_layout()
 
    plt.show()






