#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dimensionality and disentanglement
"""
from scipy import integrate
from random_expansion import *
from sparseness_expansion import *

def compute_order_param1(out):
    """
    Computes the <o^{2}> order parameter
    
    """
    sum_ = []
    for i in range(out.shape[1]):
        square = out[:,i]**(2)
        sum_.append(np.sum(square))
        #print("into sum_",np.sum(square))
    
    #print("mean",np.mean(sum_))    
    return np.mean(sum_)/out.shape[0]

def compute_order_param1_onepatt(out):
    """
    Same as above but for one (test) pattern
    
    """
    square = out[:]**(2)  
    return np.sum(square)/out.shape[0]


def compute_order_param2(out):
    """
    Computes <o^{4}>
    """
    sum_ = []
    for i in range(out.shape[1]):
        square = out[:,i]**(4)
        sum_.append(np.sum(square))
        #print("into sum_",np.sum(square))
    
    #print("mean",np.mean(sum_))    
    return np.mean(sum_)/out.shape[0]


def compute_order_param3(out):
    """
    Computes <(o_{i}o_{j})^{2}> - four point corr function
    """
    sum_ = []
    for i in range(out.shape[0]):
        for j in range(out.shape[0]):
            if i != j:
                sum_patt = []
                for p in range(out.shape[1]):
                    t1 = out[i,p]*out[j,p]
                    for q in range(out.shape[1]):
                        if p != q:
                            t2 = out[i,q]*out[j,q]
                            sum_.append(t1*t2)
                            #sum_patt.append(t1*t2)
            
                #print("mean for each pattern",np.mean(sum_patt))
                #sum_.append(np.mean(sum_patt))
        
    return np.mean(sum_)


def compute_order_param5(out):
    """
    Computes order parameter from only picking one random pattern
    """
    sum_ = []
    for i in range(out.shape[0]):
        for j in range(out.shape[0]):
            if i != j:
                p = np.random.randint(out.shape[1])
                q = np.random.randint(out.shape[1])
                t1 = out[i,p]*out[j,p]
                t2 = out[i,q]*out[j,q]
                sum_.append(t1*t2)
                
    return np.mean(sum_)



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



gaussian_func = lambda x: (1/np.sqrt(2*np.pi))*np.exp(-0.5*x**(2))

def erf1(T):
    res = integrate.quad(gaussian_func, T, np.inf)
    return res[0]


def compute_order_param4(stim):
    """
    Computes order param that should be (1-2H)^(2) - two point corr function
    """
    sum_ = []
    for i in range(stim.shape[0]):
        for j in range(stim.shape[0]):
            if i!=j:
                #print("shape in loop",stim.shape)
                for p in range(stim.shape[1]):
                    for q in range(stim.shape[1]):
                        if p==q:
                            t1 = stim[i,p]*stim[j,q]
                            sum_.append(t1)
                
    return np.mean(sum_)

def compute_order_param4_onepatt(stim):
    """
    Same as above, but without P loop
    """
    sum_ = []
    for i in range(stim.shape[0]):
        for j in range(stim.shape[0]):
            if i!=j:
                #print("shape in loop",stim.shape)
                t1 = stim[i]*stim[j]
                sum_.append(t1)
                
    return np.mean(sum_)


def compute_order_param_mixed(stim,stim_test):
    """
    Computes order parameter appearing in expression for SNR
    """
    sum_ = []
    for i in range(stim.shape[0]):
        for j in range(stim.shape[0]):
            #print("shape in loop",stim.shape)
            t2 = stim_test[i]*stim_test[j]
            for p in range(stim.shape[1]):
                t1 = stim[i,p]*stim[j,p]
                sum_.append(t1*t2)
                
    return np.mean(sum_)



##TEST TWO AND FOUR POINT CORRELATIONS
#N=100
#P=20
#H=200
#th=0.5
#stim = make_patterns(N,P)
##stim = generate_pm_with_coding(N,0.5)
#q_emps = []
#for i in range(10):
##    wrand = np.random.normal(0,1/np.sqrt(N),(H,N))
##    h = np.matmul(wrand,stim) - th
##    out = np.sign(h)
#
#    h = random_proj_generic(H,stim,thres=th)
#    out = np.sign(h)
#    
#    #q4 = compute_order_param3(out) ###4 POINT CORR
#    #q_emps.append(q4)
#    
#    q4 = compute_order_param4(out)
#    q_emps.append(q4)
#    
##    q4 = compute_order_param4_onepatt(out)
##    q_emps.append(q4)
#    
#
#q_emp = np.mean(q_emps)
#erf = erf1(th)
##q_theory = 1 - 8*(erf**(3) * (1-erf) + (1-erf)**(3) * erf)
#q_theory = (1-2*erf)**(2)
#print("order parameter",q_emp)
#print("q_theory",q_theory)



def compute_pr_theory_sim(o,th):
    """
    Returns:
        pr_emp: Empirical participation ratio
        pr_theory: Theoretical participation ratio based on 4 point corr function
        fp_corr: Four point correlation function
        
    """
    H = o.shape[0]
    P = o.shape[1]
    cov_o = np.matmul(o,o.T)
    numer = (np.matrix.trace(cov_o))**(2)
    q1 = compute_order_param1(o)
    #print("order param 1",q1)
    rand_int = np.random.randint(o.shape[1])
    coding = compute_sparsity(o[:,rand_int])
    #print("sparsity (in compute pr) is",coding)
    numer_theory = q1**(2) * H**(2) * P**(2)
    
    ratio_numer = numer/numer_theory
    print("ratio of numer experiment to theory",ratio_numer)
    
    cov2 = np.matmul(cov_o,cov_o)
    denom = np.matrix.trace(cov2)
    q2 = compute_order_param2(o)
    #print("order param 2",q2)
    q3 = compute_order_param3(o) ###TAKES PROHIBITIVELY LONG OFR LARGE P!
    #q3 = 1
    
    erf = erf1(th)
    #q3_theory = 2*(erf**(4) + (1-erf)**(4)) + 12*((erf)**(2) * (1-erf)**(2)) - 1
    q3_theory = 1 - 8*(erf**(3) * (1-erf) + (1-erf)**(3) * erf)
    
    q4 = compute_order_param4(o)
    q4_theory = (1-2*erf)**(2)
    
    fp_corr_emp = q4
    fp_corr_theory = q4_theory
    #
    
    denom_theory = H*(P*q2 + P*(P-1)*(q1)**(2)) + H*(H-1)*(P*(q1)**(2) + P*(P-1)*q3)
    
    ratio_denom = denom/denom_theory
    print("ratio of denom experiment to theory",ratio_denom)
    
    pr_theory=numer_theory/denom_theory
    
    pr_emp = compute_pr_eigvals(cov_o)
    
    return pr_emp, pr_theory, fp_corr_emp, fp_corr_theory


"""Check PR"""
#N=100
#P=20
#H=200
#th=1.0
#stim = make_patterns(N,P)
#h = random_proj_generic(H,stim,thres=th)
#o = 0.5*(np.sign(h) + 1)
#pr_emp, pr_th, fp1, fp2 = compute_pr_theory_sim(o,th)
#print("empirical pr",pr_emp)
#print("thoeretical pr",pr_th)


"""Plot of theory vs.empirical for <I^{2}>"""
#ths = np.linspace(0.1,2.0,20)
#ntrials=1
#fp_corrs = np.zeros((len(ths),n_trials))
#fp_corr_theorys = np.zeros((len(ths),n_trials))
#for i,th in enumerate(ths):
#    N=100
#    H=200
#    P=20
#    for n in range(n_trials):
#        stim = make_patterns(N,P)
#        h = random_proj_generic(H,stim,thres=th)
#        o = 0.5*(np.sign(h) + 1)
#        pr_emp, pr_theory, fp_corr_emp, fp_corr_theory = compute_pr_theory_sim(o,th)
#        fp_corrs[i,n] = fp_corr_emp
#        print("emprirical corr",fp_corr_emp)
#        fp_corr_theorys[i,n] = fp_corr_theory
#        print("theoretical corr",fp_corr_theory)
#
#fp_corr_theory_mean = np.mean(fp_corr_theorys,1)
#fp_corr_emp_mean = np.mean(fp_corrs,1)
#plt.figure()
#plt.title(r'Verification of $\langle \mathcal{I}^{ij} \rangle$',fontsize=12)
#colors = itertools.cycle(('green','blue','red','black'))
#colors_ver = itertools.cycle(('lightgreen','lightskyblue','lightcoral','grey'))
#clr = next(colors)
#clr_theory = next(colors_ver)
#plt.plot(fp_corr_emp_mean,fp_corr_theory_mean,'o',color=clr,markersize=6)
#plt.plot(fp_corr_emp_mean,fp_corr_emp_mean,'--',color=clr_theory)
#plt.ylabel(r'$\langle \mathcal{I}_{ij} \rangle _{emp}$',fontsize=14)
#plt.xlabel(r'$\langle \mathcal{I}_{ij} \rangle _{theory}$',fontsize=14)
#plt.legend()
#plt.show()




"""Plot of PR vs. <I^{2}> - different degre of mixing"""
#H_list = [50,100,200]
#K_list = [2,3,4]
#ths = np.linspace(0.02,0.1,10)
#n_trials=3
#pr_theorys = np.zeros((len(ths),len(K_list)))
#pr_emps = np.zeros((len(ths),len(K_list)))
#pr_emps_std = np.zeros((len(ths),len(K_list)))
#fp_corrs = np.zeros((len(ths),len(K_list)))
#for j,K in enumerate(K_list):
#    fp_corrs = np.zeros((len(ths),n_trials))
#    pr_emps = np.zeros((len(ths),n_trials))
#    pr_theorys = np.zeros((len(ths),n_trials))
#    for i,th in enumerate(ths):
#        N=100
#        M=100
#        P=20
#        H=200
#        for n in range(n_trials):
#            stim = make_patterns(N,P)
#            cont = make_patterns(M,K)
#            #cont = np.zeros((M,K))
#            h = random_project_hidden_layer(stim,cont,H) - th
#            #h = random_proj_generic(H,stim,thres=th)
#            o = np.sign(h)
#            print("shape of out",o.shape)
#            o_spars = 0.5*(o + 1)
#            f = compute_sparsity(o_spars[:,np.random.randint(P)])
#            print("coding is",f)
#            pr_emp, pr_theory, fp_corr_emp, fp_corr = compute_pr_theory_sim(o,th)
#            pr_emps[i,n] = pr_emp
#            pr_theorys[i,n] = pr_theory
#            fp_corrs[i,n] = fp_corr
#
#    pr_emp_mean = np.mean(pr_emps,1) 
#    pr_emp_std = np.std(pr_emps,1) 
#    pr_theory_mean = np.mean(pr_theorys,1)
#    fp_corr_mean = np.mean(fp_corrs,1)
#    print("averaged correlation",fp_corr_mean)
#    
#    pr_theorys[:,j] = pr_theory_mean
#    pr_emps[:,j] = pr_emp_mean
#    pr_emps_std[:,j] = pr_emp_std
#    fp_corrs[:,j] = fp_corr_mean
#    
#    
#plt.figure()
#plt.title(r'Dimensionality vs. inteference',fontsize=12)
#colors = itertools.cycle(('blue','red','black'))
#colors_ver = itertools.cycle(('lightskyblue','lightcoral','grey'))
#for j,K in enumerate(K_list):
#    clr = next(colors)
#    clr_theory = next(colors_ver)
#    plt.errorbar(fp_corrs[:,j],pr_emps[:,j],yerr = pr_emps_std[:,j],color=clr,fmt='s', 
#                 capsize=5, markeredgewidth=2,label='Empirical, $K={}$'.format(K))
#    plt.plot(fp_corrs[:,j],pr_theorys[:,j],'--',color=clr_theory,label='Theory')
#plt.xlabel(r'$\langle \mathcal{I}^{\mu \nu}_{ij} \rangle$',fontsize=14)
#plt.ylabel(r'$\mathcal{P}$',fontsize=14)
#plt.legend()
#plt.show()


"""GET THEORETICAL INTEFERENCE AND PR VS. THRESHOLD for different Nm"""
#P_list = [20]
#ths = np.linspace(0.1,2.0,10)
#fp_corrs = np.zeros((3,len(ths)))
#fp_corrs_theory = np.zeros((3,len(ths)))
#pr_emps = np.zeros((3,len(ths)))
#pr_theorys = np.zeros((3,len(ths)))
#for i in range(3):
#    N=100
#    M=100
#    H=200 
#    P = 20
#    K=1
#    for j,th in enumerate(ths):
#        if i == 0: #Nm=1
#            print("I=0!")
#            stim = make_patterns(N,P)
#            h = random_proj_generic(H,stim,thres=th)
#        elif i==1: #Nm=2
#            print("I=1!")
#            stim = make_patterns(N,P)
#            cont = make_patterns(M,K)
#            h = random_project_hidden_layer(stim,cont,H) - th
#        elif i==2: #Nm=3
#            print("I=3!")
#            stim = make_patterns(N,P)
#            cont = make_patterns(M,K)
#            cont2 = make_patterns(M,K)
#            h = random_project_hidden_layer_3modes(stim,cont,cont2,H) - th
#        o = np.sign(h)
#        o_spars = 0.5*(np.sign(h)+1)
#        cod = compute_sparsity(o_spars[:,np.random.randint(P)])
#        print("sparsity is",cod,"for i={}".format(i))
#        pr_emp, pr_theory, fp_corr_emp, fp_corr_theory = compute_pr_theory_sim(o,th/np.sqrt(i+1))
#        fp_corrs[i,j] = fp_corr_emp
#        print("empirical four point",fp_corr_emp)
#        fp_corrs_theory[i,j] = fp_corr_theory
#        print("theoretical four point",fp_corr_theory)
#        pr_emps[i,j] = pr_emp
#        pr_theorys[i,j] = pr_theory
#
#
#fig = plt.figure()
#ax = fig.add_subplot(121)
#ax2 = fig.add_subplot(122)
#ax.set_title(r'$\langle \mathcal{I}^{\mu \nu}_{ij} \rangle$ vs. $T$',fontsize=12)
#ax2.set_title(r'$\mathcal{P}$ vs. $\langle \mathcal{I}^{\mu \nu}_{ij} \rangle$',fontsize=12)
#colors = itertools.cycle(('blue','red','black','yellow','green'))
#colors_ver = itertools.cycle(('lightskyblue','lightcoral','grey')) 
#for i in range(3):
#    clr = next(colors)
#    clr_theory = next(colors_ver)
#    ax.plot(ths/np.sqrt(i+1),fp_corrs[i,:],'s',color = clr,label=r'$N_c={}$'.format(i+1))
#    ax.plot(ths/np.sqrt(i+1),fp_corrs_theory[i,:],'--',color = clr_theory,label=r'$Theory$')
#
#    ax2.plot(fp_corrs[i,:],pr_emps[i,:],'s',color = clr,label=r'$N_c={}$'.format(i+1))
#    ax2.plot(fp_corrs[i,:],pr_theorys[i,:],'--',color = clr_theory,label=r'$Theory$')
#    
#ax2.set_ylabel(r'$\mathcal{P}$',fontsize=14)
#ax2.set_xlabel(r'$\langle \mathcal{I}^{\mu \nu}_{ij} \rangle$',fontsize=14)
#
#ax.set_xlabel(r'$\frac{T}{\sqrt{N_{m}}}$',fontsize=14)
#ax.set_ylabel(r'$\langle \mathcal{I}^{\mu \nu}_{ij} \rangle$',fontsize=14)
#ax.legend()
#ax2.legend()
#plt.show()



def compute_delta_out(out,patt_ref,pm=True):
    """
    Here, out should be one vector of test pattern
    """
    
    diff = compute_diff(out,patt_ref)
    if pm:
        denom = 2*out.shape[0]
    else:
        denom = out.shape[0]
                
    return (1/out.shape[0])*diff

def random_proj_generic_test(H,patt,test_in,thres):
    """
    Same as random_proj_generic but for both stim and test
    """
    N = patt.shape[0]
    h = np.zeros((H,patt.shape[1]))
    h_test = np.zeros((H))
    wrand = np.random.normal(0,1/np.sqrt(N),(H,N))
    for p in range(patt.shape[1]):
        h[:,p] = np.matmul(wrand,patt[:,p]) - thres

    h_test = np.matmul(wrand,test_in) - thres
        
    return h, h_test


def compute_readout_error_numerics(out,out_test,rand_int):
    P = out.shape[1]
    H = out.shape[0]
    delta_out = compute_delta_out(out_test,out[:,np.random.randint(P)]) #Compute distance from test outputs
    print("delta_out",delta_out)
    
#    op1_train = compute_order_param1(out)
#    print("order param that should be one",op1_train)
#    op1_test = compute_order_param1_onepatt(out_test)
#    print("order param that should be one",op1_test)
    
    two_point_train = compute_order_param4(out)
    print("intef1",two_point_train)
    two_point_test = compute_order_param4_onepatt(out_test)
    print("intef2",two_point_test)
    print("product of intef",two_point_train*two_point_test)
    
    joint_order_param = compute_order_param_mixed(out,out_test)
    print("joint order param",joint_order_param)
    
    numer = (1-delta_out)*(2)
    denom = P*joint_order_param
    
    snr = numer/denom
    
    err_readout = erf1(np.sqrt(snr))
    print("ERROR NUMERICAL IS",err_readout)
    
    return err_readout


###COMPUTE PROBABILITY OF INCORRECT CLASSIFICATION IN 100 TRIALS
def compute_prob_class(N,P,H,delta_in,th):
    n_trials = 20
    stabs = []
    err_numerics = []
    for i in range(n_trials):
        labels = np.zeros(P)
        stim = make_patterns(N,P)
        rand_int = np.random.randint(P)
        test_in = flip_patterns_cluster(stim[:,rand_int],delta_in)
        delta_check = compute_delta_out(test_in,stim[:,rand_int])
        print("check delta_in",delta_check)
        h, h_test = random_proj_generic_test(H,stim,test_in,thres=th)
        
        o = np.sign(h)
        o_spars = 0.5*(np.sign(h) + 1)
        spars = compute_sparsity(o_spars)
        print("sparsity is",spars)
        o_test = np.sign(h_test)
        
        err = compute_readout_error_numerics(o,o_test,rand_int)
        err_numerics.append(err)
        
        for p in range(P):
            labels[p] = make_labels(0.5)
        weights_hebb = learn_w_hebb(o,labels)
        stab = (1/np.sqrt(stim.shape[0]))*np.matmul(weights_hebb,o_test)*labels[rand_int]
        print("stab is",stab)
        stabs.append(stab)
        
    len_neg = len(np.where(np.asarray(stabs) < 0)[0])
    len_total = len(stabs)
    prob = len_neg/len_total
    print("prob negative is",prob)
    
    return prob, np.mean(err_numerics), np.std(err_numerics)

def compute_prob_class_using_pr(N,P,H,delta_in,th):
    n_trials = 100
    stabs = []
    err_numerics = []
    for i in range(n_trials):
        labels = np.zeros(P)
        stim = make_patterns(N,P)
        rand_int = np.random.randint(P)
        test_in = flip_patterns_cluster(stim[:,rand_int],delta_in)
        #test_in = generate_pm_with_coding(N,0.5)
        delta_check = compute_delta_out(test_in,stim[:,rand_int])
        print("check delta_in",delta_check)
        h, h_test = random_proj_generic_test(H,stim,test_in,thres=th)
        #o = 0.5*(np.sign(h) + 1)
        o = np.sign(h)
#        cod = len(np.where(o[:,rand_int]==1)[0])/(o.shape[0])
#        print("coding level of o",cod)
        #o_test = 0.5*(np.sign(h_test) + 1)
        o_test = np.sign(h_test)
#        cod_test = len(np.where(o_test==1)[0])/(len(o_test))
#        print("coding level of o_test",cod_test)

        delta_out = compute_delta_out(o_test,o[:,rand_int])
        print("delta out is",delta_out)

        for p in range(P):
            labels[p] = make_labels(0.5)
        weights_hebb = learn_w_hebb(o,labels)
        print("weights hebb shape",weights_hebb.shape)
        stab = (1/np.sqrt(stim.shape[0]))*np.matmul(weights_hebb,o_test)*make_labels(0.5)
        print("stab is",stab)
        stabs.append(stab)
        
    len_neg = len(np.where(np.asarray(stabs) < 0)[0])
    len_total = len(stabs)
    prob = len_neg/len_total
    print("prob negative is",prob)
    
    #return prob, np.mean(err_numerics), np.std(err_numerics)
    return prob, o, o_test


"""Test readout error"""
##TEST MIXED ORDER PARAM - SEEM TO BE DIFEFRENT FROM PRODUCT
#N=100
#P=30
#H=200
#th=0.5
#delta_in = 0.1
#stim = make_patterns(N,P)
##stim = generate_pm_with_coding(N,0.5)
#q_seps = []
#q_mixeds = []
#for i in range(3):
#    rand_int = np.random.randint(P)
#    test_in = flip_patterns_cluster(stim[:,rand_int],delta_in,typ=True)
#    delta_test_check = compute_delta_out(test_in,stim[:,rand_int])
#    print("check delta_test",delta_test_check)
#    h, h_test = random_proj_generic_test(H,stim,test_in,thres=th)
#    out = np.sign(h)
#    out_test = np.sign(h_test)
#    
#    q1 = compute_order_param4(out)
#    q2 = compute_order_param4_onepatt(out_test)
#    print("product",q1*q2)
#    q_seps.append(q1*q2)
#    
#    q3 = compute_order_param3(out) #CHECK LK CLAIM
#    print("q3 is",q3)
#    
#    q_mixed = compute_order_param_mixed(out,out_test)
#    print("q_mixed",q_mixed)
#    q_mixeds.append(q_mixed)
#    
#
#q_sep = np.mean(q_seps)
#q_mix = np.mean(q_mixeds)
#erf = erf1(th)
#print("q_sep",q_sep)
#print("q_mixed",q_mix)


###FIRST TEST DEPENDENCE OF DELTA_OUT ON DELTA_IN
#N=100
#P=20
#H=250
#th=1.8
#d_in = 0.5
#stim = make_patterns(N,P)
#
##stim,labels = generate_clustered_stim(N,P,d_in,pm=True)
##delta_test_check = compute_delta_s(stim,pm=True)
##print("check delta_in",delta_test_check)
##out,h,cod = random_expansion_clustered(stim,H,theta=th,pm=True)
##f = np.mean(cod)
##print("sparsity is",f)
#
#rand_int = np.random.randint(P)
#test_in = flip_patterns_cluster(stim[:,rand_int],d_in,typ=True)
#delta_test_check = compute_delta_out(test_in,stim[:,rand_int])
#print("check delta_in",delta_test_check)
#
#h,h_test = random_proj_generic_test(H,stim,test_in,thres=th)
#o = np.sign(h)
#o_test = np.sign(h_test)
#o_spars = 0.5*(np.sign(h)+1)
#f = compute_sparsity(o_spars[:,np.random.randint(P)])
#print("coding is",f)
#delta_out = compute_delta_out(o_test,o[:,np.random.randint(P)])
#
##delta_out = (1/(4*f*(1-f)))*compute_delta_s(out)
#print("delta_out is",delta_out)





##TEST THEORY(NUMERICS) VS. EMP
#N=100
#P=50
#H=250
#d_in = 0.3
#th=2.5
#p,err,err_std = compute_prob_class(N,P,H,d_in,th)
##p, o, o_test = compute_prob_class_using_pr(N,P,H,d_in,th)
#print("prob negative is",p)
#print("err numerical",err)




###DIFFERENT DETLTA_INS
#delta_ins = np.linspace(0.1,0.9,9)
#P_list = [20]
##delta_ins = [0.5]
#err_emps = np.zeros(len(delta_ins))
#err_stds = np.zeros(len(delta_ins))
#err_numerics = np.zeros(len(delta_ins))
#for i,d_in in enumerate(delta_ins):
#    N=100
#    P=20
#    th = 0.9
#    H=200
#    
#    prob,err,err_std = compute_prob_class(N,P,H,d_in,th)
#    err_emps[i] = prob
#    print("empirical prob of having negative stability",prob)
#    err_numerics[i] = err
#    print("nuemrical error",err)
#    err_stds[i] = err_std
#    print("std deviation",err_std)
#    
#plt.figure()
#plt.title(r'Readout error')
#colors = itertools.cycle(('blue','red','black'))
#colors_ver = itertools.cycle(('lightskyblue','lightcoral','grey'))
#clr = next(colors)
#clr_theory = next(colors_ver)
#plt.errorbar(delta_ins,err_emps,yerr = err_std,color=clr,fmt='s', 
#             capsize=5, markeredgewidth=2,label='Empirical')
#plt.plot(delta_ins,err_numerics,'--',color=clr_theory,label='Numerical')
#plt.legend()
#plt.show()

