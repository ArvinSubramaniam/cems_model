#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heterogeneous mixing model 
"""

from random_expansion import *
from dimensionality_disentanglement import *
from sparseness_expansion import *
import random as random


def generate_hetergeneous(H,N,P,K,p1,p2,p3,d_stim,d_cont):
    """
    Generates mixed layer activations of a heterogeneous mixing 
    """
    N1 = int(p1*H)
    N2 = int(p2*H)
    N3 = int(p3*H)
    
    
    ### Produce stimuli and contextual inputs
    stim = make_patterns(N,P)
    cont = make_patterns(N,K)
    
    stim_test = make_patterns(N,P)
    cont_test = make_patterns(N,K)
    stim_comp_test = np.zeros((2*N,int(P*K)))
    
    ### Generate test patterns - flip all
    for p in range(P):
        stim_test[:,p] = flip_patterns_cluster(stim[:,p],d_stim)
    for k in range(K):
        cont_test[:,k] = flip_patterns_cluster(cont[:,k],d_cont)
    
    # plt.figure()
    # plt.imshow(stim)
    # plt.show()
    
    
    # Flip mixed one
    count0 = 0
    for p in range(P):
        stim_t_flip = flip_patterns_cluster(stim[:,p],d_stim)

        
        for k in range(K):
            cont_t_flip = flip_patterns_cluster(cont[:,k],d_cont)
            
            stim_comp_test[:,count0] = np.concatenate((stim_t_flip,cont_t_flip),axis=0)
            count0 += 1
            
        
    
    ### Generate three separate weight matrices
    mat1 = np.random.normal(0,1/np.sqrt(N),(N1,N))
    mat2 = np.random.normal(0,1/np.sqrt(N),(N2,N))
    
    mat3a = np.random.normal(0,1/np.sqrt(2*N),(N3,N))
    mat3b = np.random.normal(0,1/np.sqrt(2*N),(N3,N)) #Same prob, but different matrix. Normalized accordingly

    ### Plot composite pattern
    # plt.figure()
    # plt.imshow(stim_comp_test[:N])
    # plt.show()
    
    ### Initialize heterogeneous mixed layer activities - three separate matrices
    h1 = np.zeros((N1,P*K))
    h2 = np.zeros((N2,P*K))
    h3 = np.zeros((N3,P*K))
    
    # Same for test
    h1_test = np.zeros((N1,P*K))
    h2_test = np.zeros((N2,P*K))
    h3_test = np.zeros((N3,P*K))
    
    
    ### Fill in all entries - number of columns should be P*K
    
    ## for each pattern p
    count=0
    hcont_in = np.zeros((N2,K))
    hcont_in_t = np.zeros((N2,K))
    for p in range(P):
    # generate K times the pattern --> h_aug1
        h_stim = np.matmul(mat1,stim[:,p])
        #print("shape",h_stim.shape)
        h_aug1 = np.tile(h_stim,(K,1)).T
    
        h_stim_t = np.matmul(mat1,stim_test[:,p])
        h_aug1_t = np.tile(h_stim_t,(K,1)).T
        
        #Get h_stim for below
        h_stim_comb = np.matmul(mat3a,stim[:,p])
        h_stim_comb_t = np.matmul(mat3a,stim_test[:,p])
        
        h1[:,p*K : (p+1)*K] = h_aug1
        h1_test[:,p*K : (p+1)*K] = h_aug1_t
    
        ##for each context k
        for k in range(K):
            #get h_cont
            h_cont = np.matmul(mat2,cont[:,k])
            hcont_in[:,k] = h_cont
            
            h_cont_t = np.matmul(mat2,cont_test[:,k])
            hcont_in_t[:,k] = h_cont_t
            
            
            #Get h_cont for mixed
            h_cont_comb = np.matmul(mat3b,cont[:,k])
            h_cont_comb_t = np.matmul(mat3b,cont_test[:,k])
            
            #get h_comb = h_stim + h_cont
            h_comb = h_stim_comb + h_cont_comb
            h3[:,count] = h_comb
            
            h_comb_t = h_stim_comb_t + h_cont_comb_t
            h3_test[:,count] = h_comb_t
            count += 1
    
    #Repeat hcont_in P times
    h_cont_rep = np.tile(hcont_in,(1,P))
    h2 = h_cont_rep
    
    h_cont_rep_t = np.tile(hcont_in_t,(1,P))
    h2_test = h_cont_rep_t
    
    ### Join all h's row wise
    conc1 = np.vstack((h1,h2))
    h_out = np.vstack((conc1,h3))
    
    conc1_t = np.vstack((h1_test,h2_test))
    h_out_test = np.vstack((conc1_t,h3_test))
    
    
    return h_out, h_out_test

def genOutApplyNonlinearity(H,N,P,K,p1,p2,p3,th,ds,dc):
    
        
    h, h_test = generate_hetergeneous(H,N,P,K,p1,p2,p3,d_stim=ds,d_cont=dc)
    
    f = erf1(th)
    
    m = 0.5*(np.sign(h - th) + 1) - f
    m_test = 0.5*(np.sign(h_test - th) + 1) - f
    
    return m, m_test

def genOutApplyNonlinearityUnimod(H,N,P,th,ds):
    
    stim = make_patterns(N,P)
    len_test = P
    patts_test = np.zeros((N,len_test))
    ##CREATE TEST PATTERNS
    for n in range(len_test):#Pick test points - perturb ONE pattern randomly
        patt_typ = stim[:,n]
        patt_test = flip_patterns_cluster(patt_typ,ds)
        d_in_check = compute_delta_out(patt_test,patt_typ)
        patts_test[:,n] = patt_test

    h,h_test = random_proj_generic_test(H,stim,patts_test,th)
    
    f = erf1(th)
    
    m = 0.5*(np.sign(h) + 1) - f
    m_test = 0.5*(np.sign(h_test) + 1) - f
    
    return m, m_test


def compute_delta_m_theory_het(ds,dc,p1,p2,p3,th):
    d_eff = 0.5*ds + 0.5*dc
    cod = erf1(th)
    size1 = p1 * erf_full(th,ds,cod)
    size2 = p2 * erf_full(th,dc,cod)
    size3 = p3 * erf_full(th,d_eff,cod)
    return size1 + size2 + size3
    
    
## Check for value of delta_s effective, and coding
# N=100
# P=10
# K=10
# ds=0.4
# dc=0.4
# H=2100
# p1 = 1/3
# p2 = 1/3
# p3 = 1/3
# th = 0.8   

# h,h_test = generate_hetergeneous(H,N,P,K,p1,p2,p3,ds,dc)

# Pin = np.random.randint(P)
# Nin = int(p1*H)
# N2 = Nin + int(p2*H)
# inner = (1/(int(p3*H)))*np.dot(h[N2:,Pin],h_test[N2:,Pin])
    
# m, m_test = genOutApplyNonlinearity(H,N,P,K,p1,p2,p3,th,ds,dc)
# print("coding is",compute_sparsity(m[:,Pin]))
# print("theory",erf1(th))

# dot = (1/H)*np.dot(m[:,Pin],m_test[:,Pin])
# print("dot",dot)

### Compute cluster size
run_delta_m2 = False
if run_delta_m2:
    N=100
    M=100
    P=10
    K=10
    H=2100
    
    dc = 0.1
    
    thress = np.linspace(0,3.1,20)
    ds_list = [0.1,0.3,0.5] #Actually a list of p1=p2, but p3 is determined
    p1 = 0.1
    p2 = p1
    p3 = 1 - p1 - p2
    cods = np.zeros((len(thress)))
    
    delta_emps = np.zeros((len(ds_list),len(thress)))
    delta_theorys = np.zeros((len(ds_list),len(thress)))

    for j,th in enumerate(thress):
        cod = erf1(th)
        cods[j] = cod
        print("coding",cod)
        
        for i, ds in enumerate(ds_list):
            o, o_test = genOutApplyNonlinearity(H,N,P,K,p1,p2,p3,th,ds,dc)
            
            d1_list = []
            
            for p in range(o.shape[1]):
                d1 = compute_delta_out(o[:,p],o_test[:,p])
                d1_list.append(d1)
            
            delta_emps[i,j] = np.mean(d1_list)/(2*cod*(1-cod))
            print("delta emps",np.mean(d1_list))
            
            ### Cluster size is p1*d1(ds) + p2*d2(dc) + p3*(deff)
            delta_theorys[i,j] = compute_delta_m_theory_het(ds,dc,p1,p2,p3,th)
            print("delta theorys",compute_delta_m_theory_het(ds,dc,p1,p2,p3,th))
            
    
    plt.figure()
    x_list = [1,2,3]
    colors = itertools.cycle(('blue','red','black'))
    colors_ver = itertools.cycle(('lightskyblue','lightcoral','grey'))
    plt.title(r'Heterogeneous model, $\Delta \phi = 0.1$, $p3={}$,$p2=p1$'.format(np.round(p3,2)))
    for i,ds in enumerate(ds_list):
        clr = next(colors)
        clr_theory = next(colors_ver)
        plt.plot(cods,delta_emps[i,:],'s',label=r'$\Delta \xi={}$'.format(np.round(ds,2)),color=clr)
        plt.plot(cods,delta_theorys[i,:],'--',color=clr_theory)
    plt.xlabel(r'$f$',fontsize=14)
    plt.ylabel(r'$\Delta m$',fontsize=14)
    plt.legend()
    plt.show()


#### Compute readout error
    
gaussian_func_onehalf_= lambda y,x: (1/(2*np.pi*np.sqrt(1 - (1/2)**(2)))) * x * y \
                                        *np.exp(-(1./(2*(1 - (1/2)**(2))))*(x**(2) + y**(2) - 2*(1/2)*x*y))  
    
def two_pt(th,pk):
    res = integrate.dblquad(gaussian_func_onehalf_ ,th, np.inf, lambda x: th, lambda x: np.inf)
    return (pk/(1 - (pk)**(2))) * res[0]
    
def compute_excess_over_hetero(N,P,K,th):
    """
    Excess overlap for heterogeneous mixing

    """
    pk=1/2
    erf = erf1(th)
    feff2 = two_pt(th,pk)
    q2 = erf*(1-erf)
    i4 = feff2
    
    prefact = 1/(np.sqrt(N))
    
    excess_over_structured = (prefact*i4 - erf**(2))**(2) / (q2**(2))
    
    #prob_prefact = 1 - (1/K) - (1/P)
    prob_prefact = 1
    
    return prob_prefact * excess_over_structured

gaussian_func_2dim_onehalf = lambda y,x: (1/(2*np.pi))*np.exp(-(1./2)*(x**(2) + y**(2)))

def two_dim_integral_thres(t_in):
    """
    Parameters
    ----------
    t_phi : Either t_phi or t_xi 

    """
    res = integrate.dblquad(gaussian_func_2dim_onehalf, t_in, np.inf, lambda x: th, lambda x: np.inf)
    return res[0]

def func_3dim(x,y,p):
    exp_ = np.exp(-(1./2)*(x**(2) + y**(2)) + p*x*y)
    denom = (1/(2*np.pi))
    return denom * exp_

def func_3dim_g(x,y,p):
    exp_ = x*y*np.exp(-(1./2)*(x**(2) + y**(2)) + p*x*y)
    denom = (1/(2*np.pi))
    return denom * exp_

def two_dim_i_integral(t_in,p):
    """
    Parameters
    ----------
    t_in : Either t_phi or t_xi 
    p: Peak 

    """
    res = integrate.dblquad(func_3dim, t_in, np.inf, lambda x: t_in, lambda x: np.inf,args=(p,))
    return res[0]

def two_dim_g_integral(t_in,p):
    """
    Parameters
    ----------
    t_in : Either t_phi or t_xi 
    p: Peak 

    """
    res = integrate.dblquad(func_3dim_g, t_in, np.inf, lambda x: t_in, lambda x: np.inf,args=(p,))
    return res[0]
    
    
def hebbian_mixed_layer_heterogeneous(H,N,P,K,th,p1,p2,p3,ds=0.1,dc=0.):
    """
    comp_num: True if want to compare "numerical theory" to theory
    """
    n_real = 20
    errors = np.zeros(n_real)
    Peff = P*K
    #Peff=P
    len_test = int(Peff)
    erf = erf1(th)
    for j in range(n_real):
        labels = np.zeros(int(Peff))
        for i in range(int(Peff)):
            labels[i] = make_labels(0.5)
            
        o_in, o_test_in = genOutApplyNonlinearity(H,N,P,K,p1,p2,p3,th,ds,dc)
        
        #o_in, o_test_in = genOutApplyNonlinearityUnimod(H,N,P,th,ds)
        
        w_hebb = np.matmul(o_in,labels) 
        stabs = []
        d_outs = []
        acts_typ = np.zeros((H,len_test))
        labels_test = labels
        
        for m in range(len_test):
            stab = labels_test[m]*np.dot(w_hebb,o_test_in[:,m])
            stabs.append(stab)
            d_out = (1/(2*erf*(1-erf))) * compute_delta_out(o_test_in[:,m],o_in[:,m])
            d_outs.append(d_out)
            #print("d_out is",d_out)
            acts_typ[:,m] = o_in[:,m]
        
        #print("stabilities",stab)
        err = (len(np.where(np.asarray(stabs)<0)[0]))/(len(stabs))
        errors[j] = err
            
    err_mean = np.mean(errors)
    err_std = np.std(errors)
    
    d_out_mean = np.mean(d_outs)
    print("d_out is",d_out_mean)
    
    d_theory = compute_delta_m_theory_het(ds,dc,p1,p2,p3,th)
    d_theory_out = d_theory
    print("d_out theory",d_theory)
    
    d_theory1 = (1 - erf_full(th,ds,erf))

    diff = 1 - d_theory_out
    numer = diff**(2)
    
    numer1 = (d_theory1)**(2)
    
    alpha = Peff/H
    
    #Get effective qtheory
    f_one_minus = erf*(1-erf) #erf = f 
    
    #Probabilities in theory
    q1 = (K-1)/(Peff-1)
    q2 = (P-1)/(Peff-1)
    
    #Peaks
    p_xi = (1/2)*(1 - ds)
    p_phi = (1/2)*(1 - dc)
    
    #Thresholds
    sqrt_xi = np.sqrt(1 - p_xi**(2))
    sqrt_phi = np.sqrt(1 - p_phi**(2))
    t_xi = th/(sqrt_xi)
    t_phi = th/(sqrt_phi)
    print("thresholds",t_phi,t_xi)
    print("BELOW COD!")
    
    size1 = erf_full(th,ds,erf)
    size2 = erf_full(th,dc,erf)
    ds_eff = 0.5*(ds + dc)
    size3 = erf_full(th,ds_eff,erf)
    
    #Get G functions
    g_func_generic = (1/(2*np.pi)) * np.exp(-th**(2))
    #g_func_xi = (1/(2*np.pi)) * np.exp(-t_xi**(2))
    #g_func_phi = (1/(2*np.pi)) * np.exp(-t_phi**(2))
    
    g_func_xi = two_dim_g_integral(t_xi,p_xi)
    g_func_phi = two_dim_g_integral(t_phi,p_phi)
    
    eo_xi =  (1/N)*excess_over_theory(th,erf)**(2)
    eo_phi = eo_xi
    eo_t_phi = (1/(4*N)) * g_func_phi**(2)
    eo_t_xi = (1/(4*N)) * g_func_xi**(2)
    eo_cross = (1/((2*np.pi)**(2))) * (np.exp(-2*th**(2))) * (1/N)
    eo_sqrt_phi_inter = (1/(2*N)) * np.exp(-th**(2)) * g_func_phi
    eo_sqrt_xi_inter = (1/(2*N)) * np.exp(-th**(2)) * g_func_xi
    eo_cross_inter = (1/(2*N)) * (1/((2*np.pi)**(2))) * (np.exp(-2*th**(2)))
    
    f=erf
    
    #Get the two dimensional integrals
    i_phi = two_dim_i_integral(t_phi,p_phi)
    i_xi = two_dim_i_integral(t_xi,p_xi)
    
    
    #Get the R functions -- > this comes from Eq. 15
    r_xi = f_one_minus*(1-size1)
    r_phi = f_one_minus*(1-size2)
    r_mixed = (q1)*(sqrt_xi)*(i_xi) + (q2)*(sqrt_phi)*(i_phi)  + (1-q1-q2)*f**(2)


    part3_one = (2*f**(3) - 3*f**(4))/(f_one_minus**(2))
    part3_xi = (1/(f_one_minus**(2)))*p1*r_xi*q1
    part3_phi = (1/(f_one_minus**(2)))*p2*q2*r_phi
    part3_mixed = (1/(f_one_minus**(2)))*p3*r_mixed
    part3 = (Peff/H)*(part3_one + (2*f - 1)**(2) * (part3_xi + part3_phi + part3_mixed)) #H = N_{c}
    print("part3",part3)
    
    #Terms for intra part
    part4_xi_1 = ((1/f_one_minus)*r_xi*p1)**(2) * q1
    part4_xi_eo = (p1)**(2) * eo_xi * (1-q1)
    part4_phi_1 = ((1/f_one_minus)*r_phi*p2)**(2) * q2
    part4_phi_eo = (p2)**(2) * eo_phi * (1-q2)
    
    #Part 4 intra for mixed terms
    part4_mixed_four_pt = (q1)*((sqrt_xi*i_xi)**(2) + eo_t_xi) + (q2)*((sqrt_phi*i_phi)**(2) +  eo_t_phi) \
                             + (1 - q1 - q2)*(f**(4) + eo_cross)
    part4_mixed = p3**(2) * (part4_mixed_four_pt - 2*f**(2) * r_mixed + f**(4)) * (1/f_one_minus)**(2)
    
    i4_intra = Peff * (part4_xi_1 + part4_xi_eo + part4_phi_1 + part4_phi_eo + part4_mixed)
    print("i4 intra",i4_intra) 
    
    ##Terms for inter part
    part4_one_three = q1*(r_xi + f**(2))*sqrt_xi*(i_xi)  + \
        (q2)*(f**(2) * sqrt_phi * (i_phi) + sqrt_phi*eo_sqrt_phi_inter) + (1-q1-q2)*(f**(4) + eo_cross_inter)\
        - f**(2) * (q1*r_xi + r_mixed)
    #part4_one_three = 0
    #print("part4_1_3_parts",part4_one_three + f**(2) * (q1*r_xi + r_mixed),f**(2) * (q1*r_xi + r_mixed))
    part4_two_three = q2*(r_phi + f**(2))*sqrt_phi*(i_phi)  + \
        (q1)*(f**(2) * sqrt_xi * (i_xi) + sqrt_xi*eo_sqrt_xi_inter) + (1-q1-q2)*(f**(4) + eo_cross_inter)\
        - f**(2) * (q2*r_phi + r_mixed)
    #part4_two_three = 0
    print("part 4's",part4_one_three,part4_two_three)
    part4_one_two = 0*q1*q2*(r_xi*r_phi)
    
    part4_full = (1/f_one_minus)**(2) * (2*p1*p2*part4_one_two + 2*p1*p3*part4_one_three + 2*p2*p3*part4_two_three)
    i4_inter = Peff * part4_full
    print("i4 inter",i4_inter)

    denom2 = part3 + i4_intra + i4_inter #Peff = PK
    
    denom2_one = P/H + (P/N) * excess_over_theory(th,f)**(2)
    
    snr = (numer)/(denom2)
    
    snr1 = (numer1)/(denom2_one)
    
    err_theory = erf1(snr**(0.5))
    
    err_theory1 = erf1(snr1**(0.5))
    
    return err_theory, err_mean



run_readout_err = True
if run_readout_err:
    N=100
    P=50
    K=4
    H=2100
    
    thress = np.linspace(0,3.1,20)
    #thress = [1.1]
    ds_list = [0.1,0.3,0.5] #Actually a list of p1=p2, but p3 is determined
    #ds_list = [0.1]
    p1 = 0.10
    p2 = p1
    p3 = 1 - p1 - p2
    dc = 0.1
    cods = np.zeros((len(thress)))
    
    err_empirical = np.zeros((len(ds_list),len(thress)))
    err_theorys = np.zeros((len(ds_list),len(thress)))


    for j,th in enumerate(thress):
        cod = erf1(th)
        cods[j] = cod
        print("coding",cod)
        
        for i, ds in enumerate(ds_list):
            
            err_theory, err_mean = hebbian_mixed_layer_heterogeneous(H,N,P,K,th,p1,p2,p3,ds,dc)
            
            # d1_list = []
            
            # for p in range(o.shape[1]) and False:
            #     d1 = compute_delta_out(o[:,p],o_test[:,p])
            #     d1_list.append(d1)
            
            err_empirical[i,j] = err_mean
            print("empirical error",err_mean)
            
            ### Cluster size is p1*d1(ds) + p2*d2(dc) + p3*(deff)
            err_theorys[i,j] = err_theory
            print("theoretical err",err_theory)

    
    plt.figure()
    x_list = [1,2,3]
    colors = itertools.cycle(('blue','red','black'))
    colors_ver = itertools.cycle(('lightskyblue','lightcoral','grey'))
    plt.title(r'Heterogeneous model, $\Delta \phi = 0.1$, $p3={}$,$p2=p1$'.format(np.round(p3,2)))
    for i,ds in enumerate(ds_list):
        clr = next(colors)
        clr_theory = next(colors_ver)
        plt.plot(cods,err_empirical[i,:],'s',label=r'$\Delta \xi={}$'.format(np.round(ds,2)),color=clr)
        plt.plot(cods,err_theorys[i,:],'--',color=clr_theory)
    plt.xlabel(r'$f$',fontsize=14)
    plt.ylabel(r'$\epsilon$',fontsize=14)
    plt.legend()
    plt.show()
    

    

    
    
    