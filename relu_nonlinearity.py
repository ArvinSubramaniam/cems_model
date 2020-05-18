#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threshold linear unimodal model
"""

from dimensionality_disentanglement import *
from interpolate import *

def compute_sparsity_relu(out):
    numer  = len(np.where(out>0)[0])
    denom  = len(out)
    
    return numer/denom


def relu_nonlinearity(h,th=0):
    outs = np.zeros((h.shape[0],h.shape[1]))
    for i in range(h.shape[1]):
        bool_arr = (h[:,i] > th)*1
        arr_in = bool_arr * h[:,i]
        outs[:,i] = arr_in
    
    return outs

def compute_q2(th):
    t1 = th * (1/(np.sqrt(2*np.pi))) * np.exp(-0.5*th**(2))
    t2 = erf1(th)
    
    return t1 + t2

def compute_q4(th):
    t1 = 3*erf1(th)
    t2 = (th**(2) + 3*th) * (1/(np.sqrt(2*np.pi))) * np.exp(-0.5*th**(2))
    
    return t1 + t2

def compute_i4(th,N):
    t1 = (1/(2*np.pi)**(2)) * np.exp(-2*th**(2))
    t2 = (1/N)*compute_q2(th)**(4)
    return t1 + t2


def compute_pr_relu(o,th,N):
    H  = o.shape[0]
    P = o.shape[1]
    denom1 = 1/(H*P)
    
    q2 = compute_q2(th)
    q4 = compute_q4(th)
    
    ratio1 = (q4/(q2**(2)))
    
    eo = compute_i4(th,N)
    
    denom2 = eo/(q2**(2))
    
    
    pr = 1/(denom1*ratio1 + 1/H + 1/P + denom2)
    
    return pr



gaussian_func_2dim_ds1 = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (0.9)**(2))))*x*y  \
                                        *np.exp(-(1./(2*(1 - (0.9)**(2))))*(x**(2) + y**(2) - 2*(0.9)*x*y))
  
gaussian_func_2dim_ds1_eo = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (0.9)**(2))))* x**(2) * y**(2)  \
                                        *np.exp(-(1./(2*(1 - (0.9)**(2))))*(x**(2) + y**(2) - 2*(0.9)*x*y))    

def two_pt_relu(th):
    res = integrate.dblquad(gaussian_func_2dim_ds1, th, np.inf, lambda x: th, lambda x: np.inf)[0]
    return res

def eo_relu(th):
    res = integrate.dblquad(gaussian_func_2dim_ds1_eo, th, np.inf, lambda x: th, lambda x: np.inf)[0]
    return res

def c1_theory(pk):
    frac = pk**(2)/((1-(pk)**(2))**(2))
    return frac

def c2_theory(pk):
    numer = 1
    denom = (1 - pk**(2))**(2)
    return numer/denom
        
def compute_numer_relu(N,th,pk):
    "pk = 1-ds"
    c1 = c1_theory(pk)
    print("c1",c1)
    c2 = c2_theory(pk)
    print("c2",c2)
    
    i2 = two_pt_relu(th)
    eo = eo_relu(th)
    
    ratio1 = 1 + (1/N)*c1
    ratio2 = (1/N)*c2
    ratio3 = pk/((1-pk**(2))**(2)) * (1/N)
    
    return ratio1*(i2**(2)) + ratio2*(eo**(2)) + ratio3*i2*eo

def compute_numer_relu2(N,th):
    lead = two_pt_relu(th)
    eo2 = eo_relu(th)
    sublead = (1/N)*eo2

    return lead + sublead

def hebbian_mixed_layer_relu(H,N,P,th,ds=0.1):
    """
    comp_num: True if want to compare "numerical theory" to theory
    """
    n_real = 20
    errors = np.zeros(n_real)
    Peff = P
    len_test = int(Peff)
    erf = erf1(th)
    stim = make_patterns(N,P)
    for j in range(n_real):
        labels = np.zeros(int(Peff))
        for i in range(int(Peff)):
            labels[i] = make_labels(0.5)

        patts_test = np.zeros((N,len_test))
        labels_test = labels
        ints = np.arange(P)
        ##CREATE TEST PATTERNS
        for n in range(len_test):#Pick test points - perturb ONE pattern randomly
            patt_typ = stim[:,n]
            patt_test = flip_patterns_cluster(patt_typ,ds)
            d_in_check = compute_delta_out(patt_test,patt_typ)
            #print("d_in_check",d_in_check)
            patts_test[:,n] = patt_test
        
        h,h_test = random_proj_generic_test(H,stim,patts_test,0)
        
        o = relu_nonlinearity(h,th)
        o_test = relu_nonlinearity(h,th)
        
        o_in = o
        o_test_in = o
        
        f = compute_sparsity_relu(o_in[:,np.random.randint(Peff)])
        print("f is",f)
        
        w_hebb = np.matmul(o_in,labels) 
        
        #print("shape hebbian weights",w_hebb.shape)
        stabs = []
        d_outs = []
        acts_typ = np.zeros((H,len_test))
        labels_test = labels
        for m in range(len_test):
            #print("label2 is",labels_test[m])
            stab = labels_test[m]*np.dot(w_hebb,o_test_in[:,m])
            #print("stab is",stab)
            stabs.append(stab)
            #print("computing d_out")
            d_out = (1/(2*erf*(1-erf))) * compute_delta_out(o_test_in[:,m],o_in[:,m])
            d_outs.append(d_out)
            #print("d_out is",d_out)
            acts_typ[:,m] = o_in[:,m]
          
        #print("finished one test cycle,alpha=",P/N)
            
        err = (len(np.where(np.asarray(stabs)<0)[0]))/(len(stabs))
        errors[j] = err
            
    err_mean = np.mean(errors)
    err_std = np.std(errors)
    
    d_out_mean = np.mean(d_outs)
    print("d_out is",d_out_mean)

    d_in = ds

#    delta = compute_numer_relu(N,th,1-d_in)
#    print("delta is",delta)
    
    delta = compute_numer_relu2(N,th)
    print("delta is",delta)
    
    q_theory_in = compute_i4(th,N) 
    print("q_theory in",q_theory_in)
    
    ratio = compute_q2(th)**(2)

    numer = delta**(2)

    print("first term in denom",(Peff/H)*ratio)
    print("second term in denom",Peff * q_theory_in)
    denom2 = (Peff/H)*ratio + Peff * q_theory_in 
    
    snr = (numer)/(denom2)
    snr_in = snr
    err_theory = erf1(snr_in**(0.5))

    
    return snr, err_mean, err_std, err_theory, erf1(th)

run_dimensionality_relu = False
if run_dimensionality_relu:
    N=100
    P=200
    H=2000
    stim = make_patterns(N,P)
    stim_test = np.zeros((N,P))
    for p in range(stim.shape[1]):
        stim_test[:,p] = flip_patterns_cluster(stim[:,p],0.1)
    thress = np.linspace(0.1,3.2,10)
    pr_emps = np.zeros(len(thress))
    pr_theorys = np.zeros(len(thress))
    fp_corrs = np.zeros(len(thress))
    cods = np.zeros(len(thress))
    for i,th in enumerate(thress):
        print("th",th)
        #h,h_test = random_proj_generic_test(H,stim,stim_test,th)
        h = random_proj_generic(H,stim)
        o = relu_nonlinearity(h,th)
        #o_spars = 0.5*(o + 1)
        f = compute_sparsity(o[:,np.random.randint(P)])
        o_spars_in = o
        cov = np.matmul(o_spars_in,o_spars_in.T)
        print("f is",f)
        cods[i] = f
        #o_test = 0.5*(np.sign(h_test) + 1)
        pr_emp = compute_pr_relu(o_spars_in,th,N)
        pr_theory = compute_pr_eigvals(cov)
        fp_corr = compute_i4(th,N)
        print("pr_theory",pr_theory)
        print("pr_emp",pr_emp)
        pr_emps[i] = pr_emp
        pr_theorys[i] = pr_theory
        fp_corrs[i] = fp_corr
        
    plt.figure()
    import matplotlib.ticker as ticker
    plt.title(r'Dimensionality',fontweight="bold",fontsize=16)
    plt.plot(fp_corrs,pr_emps,'s',markersize=8,color='blue')
    plt.plot(fp_corrs,pr_theorys,'--',color='lightblue')
    plt.ylabel(r'$\mathcal{D}$',fontsize=16)
    plt.xlabel(r'$f$',fontsize=16)
    plt.legend()
    plt.show()

run_sweep_ratio = False
"""
3 sweeps:
    1. For different f
    2. For different Delta_in
    3. For different alpha
"""
if run_sweep_ratio:
    N=100 #N larger for {+,-}
    #H=20000
    bool_ = False #True for {+,-}
    bool2_ = False
    #P=100
    thress = np.linspace(0,2.1,10)
    #alphas = [0.5,1.0,2.5]
    alphas = [0.01,0.1,0.3]
    delta = 0.1
    #thress = [1,500] #HERE THRESS "=" R
    P_arr = np.linspace(0.2*N,4*N,10)
    P_list = [int(p) for p in P_arr]
    H_arr = np.linspace(5.0*N,30*N,20)
    H_list = [int(p) for p in H_arr]
    #thress = [20*N]
    #H_arr = np.asarray(thress)
    err_means = np.zeros((len(thress),len(alphas)))
    err_stds = np.zeros((len(thress),len(alphas)))
    err_theorys = np.zeros((len(thress),len(alphas)))
    snrs = np.zeros((len(thress),len(alphas)))
    cods = np.zeros((len(thress)))
    for j,th in enumerate(thress):
        for i,a in enumerate(alphas):
            H = 4000
            #H = int(H_in)
            #th = a
            #P=int(a*N)
            P=50
            snr, err_mean, err_std, err_theory, f= hebbian_mixed_layer_relu(H,N,P,th,ds=a)
            err_means[j,i] = err_mean
            print("empirical error",err_mean)
            err_theorys[j,i] = err_theory
            print("theoretical error",err_theory)
            err_stds[j,i] = err_std
            
            snrs[j,i] = snr
            cods[j] = f
            #print("coding is",f)

    colors = itertools.cycle(('green','blue','red','black'))
    colors_ver = itertools.cycle(('lightgreen','lightskyblue','lightcoral','grey'))
    plt.figure()
    plt.title(r'ReLU model,$\alpha=0.5$',fontsize=12)
    for i,a in enumerate(alphas):
        clr = next(colors)
        clr_theory = next(colors_ver)
        print("lenghts",len(H_arr),len(err_means[:,i]),len(err_stds[:,i]))
        plt.errorbar(cods,err_means[:,i],yerr=0.1*err_stds[:,i],color=clr,fmt='s',
                     capsize=5, markeredgewidth=2,label=r'$\Delta \xi={}$'.format(a))
        plt.plot(cods,err_theorys[:,i],'--',color=clr_theory)
    plt.xlabel(r'$f$',fontsize=14)
    plt.ylabel(r'Readout error',fontsize=14)
    plt.legend(fontsize=10)
    plt.show()


