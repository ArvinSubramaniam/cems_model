#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For contextual decoding
"""

from dimensionality_disentanglement import *
from interpolate import *

def generate_order_full_mixed_test_decoding(H,N,M,P):
    
    """
    Same as in interplate, but K should be 2
    """
    
    stim = make_patterns(N,P)
    cont1 = make_patterns(M,1)
    cont2 = make_patterns(M,1)
    
    K = cont1.shape[1]

    stim_eff = arrange_composite_form(stim,cont1,cont2) #Make sure that all are arranged in a (PK^(2) x Nc) matrix
    print("shape stim_eff",stim_eff.shape)
    
#    plt.figure()
#    plt.title("Stimulus")
#    plt.imshow(stim_eff)
#    plt.colorbar()
#    plt.show()
    
    Peff = P*K*K
    test_len = int(Peff)
    ints_test = np.arange(test_len)
    
    stim_test = stim
    cont1_test = make_patterns(M,1)
    cont2_test = make_patterns(M,1)
    
    stim_test_in = arrange_composite_form(stim_test,cont1_test,cont2_test)
    
#    plt.figure()
#    plt.title("Test stimulus")
#    plt.imshow(stim_test_in)
#    plt.colorbar()
#    plt.show()
    
    for i in range(test_len):
        d_check = compute_delta_out(stim_test_in[:,i],stim_eff[:,i])   
        #print("check delta",d_check)

    mat1 = np.random.normal(0,1/np.sqrt(3*N),(H,N))
    mat2 = np.random.normal(0,1/np.sqrt(3*M),(H,M))
    mat3 = np.random.normal(0,1/np.sqrt(3*M),(H,M))
    
    h = np.zeros((H,Peff))
    h_test = np.zeros((H,test_len))
    
    stim_test, cont1_test, cont2_test = decompose_from_composite(stim_test_in,N,M,P,K)
    
    count=0
    #For loop for training data
    for p in range(P):
        h_stim = np.matmul(mat1,stim[:,p])
        h_test_stim = np.matmul(mat1,stim_test[:,p])
        for k in range(K):
            #count+=1
            h_cont1 = np.matmul(mat2,cont1[:,k])
            h_test_cont1 = np.matmul(mat2,cont1_test[:,k])
            for l in range(K):
                count+=1
                h_cont2 = np.matmul(mat3,cont2[:,l])
                h_test_cont2 = np.matmul(mat3,cont2_test[:,l])
                #print("shape of things multiplied",mat3.shape,stim_test_in[N+M:,p].shape)
                h_in = h_stim + h_cont1 + h_cont2
                h_in_test = h_test_stim + h_test_cont1 + h_test_cont2
                h[:,count-1] = h_in
                h_test[:,count-1] = h_in_test
    
                
    return h, h_test


def generate_order_two_mixed_test_decoding(H,N,M,P):
    """Same as in interplate, but K should be 2"""

    stim = make_patterns(N,P)
    cont1 = make_patterns(M,1)
    cont2 = make_patterns(M,1)
    
    K = cont1.shape[1]

    stim_eff = arrange_composite_form(stim,cont1,cont2) #Make sure that all are arranged in a (PK^(2) x Nc) matrix
    print("shape stim_eff",stim_eff.shape)
    
    Peff = P*K*K
    test_len = int(Peff)
    ints_test = np.arange(test_len)
    
    stim_test = stim
    cont1_test = make_patterns(M,1)
    cont2_test = make_patterns(M,1)
    
    stim_test_in = arrange_composite_form(stim_test,cont1_test,cont2_test)
    
    for i in range(test_len):
        d_check = compute_delta_out(stim_test_in[:,i],stim_eff[:,i])   
        #print("check delta",d_check)


    mat1 = np.random.normal(0,1/np.sqrt(2*N),(int(H/2),N))
    mat2 = np.random.normal(0,1/np.sqrt(2*M),(int(H/2),M))
    mat3 = np.random.normal(0,1/np.sqrt(2*M),(int(H/2),M))
    
    h = np.zeros((H,Peff))
    h_test = np.zeros((H,test_len))
    
    stim_test, cont1_test, cont2_test = decompose_from_composite(stim_test_in,N,M,P,K)
    
    tile_list = np.zeros((int(H/2),P*K))
    tile_list_test = np.zeros((int(H/2),P*K))
    count1=0
    count2=0
    for p in range(P):
        h_stim = np.matmul(mat1,stim[:,p])
        h_stim_test = np.matmul(mat1,stim_test[:,p])
        for k in range(K):
            count1+=1
            #print("CONT1",k)
            h_cont1 = np.matmul(mat2,cont1[:,k])
            h_cont1_test = np.matmul(mat2,cont1_test[:,k])
            h_mix1 = h_stim + h_cont1
            h_mix1_test = h_stim_test + h_cont1_test
            tile_list[:,count1-1] = h_mix1
            tile_list_test[:,count1-1] = h_mix1_test
            
#    plt.figure()
#    plt.title("Image of tile list")
#    plt.imshow(tile_list)
#    plt.show()
    
    h[:int(H/2),:] = np.repeat(tile_list,K,axis=1)
    h_test[:int(H/2),:] = np.repeat(tile_list_test,K,axis=1)
    
#    plt.figure()
#    plt.title("Image of tile_list repeated")
#    plt.imshow(h[:int(H/2),:])
#    plt.show()
  
    for l in range(K):
        count2+=1
        #print("count2",count2)
        h_cont2 = np.matmul(mat3,cont2[:,l])
        h_cont2_test = np.matmul(mat3,cont2_test[:,l])
        tile_list2 = np.zeros((int(H/2),P*K))
        tile_list2_test = np.zeros((int(H/2),P*K))
        for p in range(P):
            h_stim = np.matmul(mat1,stim[:,p])
            h_stim_test = np.matmul(mat1,stim_test[:,p])
            h_mix2 = h_stim + h_cont2
            h_mix2_test = h_stim_test + h_cont2_test
            #h2t = np.repeat(np.reshape(h_mix2,(int(H/2),1)),K,axis=1) #REPEAT (xi,eta) mixing K times for each P
            h2t = np.tile(np.reshape(h_mix2,(int(H/2),1)),K)
            h2test = np.tile(np.reshape(h_mix2_test,(int(H/2),1)),K)
            tile_list2[:,p*K:(p+1)*K] = h2t #K spacing P times = PK
            tile_list2_test[:,p*K:(p+1)*K] = h2test #K spacing P times = PK
        h[int(H/2):,l*P*K:(l+1)*P*K] = tile_list2
        h_test[int(H/2):,l*P*K:(l+1)*P*K] = tile_list2_test

#    plt.figure()
#    plt.title("Image of tile_list2")
#    plt.imshow(tile_list2)
#    plt.show()

    
    return h, h_test


def generate_order_one_mixed_test_decoding(H,N,M,P):
    
    """
    Same as in interplate, but K should be 2
    """
    
    stim = make_patterns(N,P)
    cont1 = make_patterns(M,1)
    cont2 = make_patterns(M,1)
    
    K = cont1.shape[1]
    
    stim_eff = arrange_composite_form(stim,cont1,cont2) #Make sure that all are arranged in a (PK^(2) x Nc) matrix
    print("shape stim_eff",stim_eff.shape)
    
#    plt.figure()
#    plt.title("Stimulus")
#    plt.imshow(stim_eff)
#    plt.colorbar()
#    plt.show()
    
    Peff = P*K*K
    test_len = int(Peff)
    ints_test = np.arange(test_len)
    
    stim_test = stim
    cont1_test = make_patterns(M,1)
    cont2_test = make_patterns(M,1)
    
    stim_test_in = arrange_composite_form(stim_test,cont1_test,cont2_test)
    
    for i in range(test_len):
        d_check = compute_delta_out(stim_test_in[:,i],stim_eff[:,i])   
        #print("check delta",d_check)
        
#    plt.figure()
#    plt.title("Test pattern")
#    plt.imshow(stim_test_in)
#    plt.colorbar()
#    plt.show()
        
    stim_test, cont1_test, cont2_test = decompose_from_composite(stim_test_in,N,M,P,K)
    
    h_big = np.zeros((H,P*K*K))
    h_big_test = np.zeros((H,P*K*K))
    
    mat1 = np.random.normal(0,1/np.sqrt(N),(int(H/3),N))
    mat2 = np.random.normal(0,1/np.sqrt(M),(int(H/3),M))
    mat3 = np.random.normal(0,1/np.sqrt(M),(int(H/3),M))
    
    arr_stim = np.zeros((int(H/3),P)) #To be repeated K^(2) times
    arr_stim_test = np.zeros((int(H/3),P))
    arr_cont1 = np.zeros((int(H/3),P*K)) #To be repeated K times
    arr_cont1_test = np.zeros((int(H/3),P*K))
    arr_cont2 = np.zeros((int(H/3),P*K*K))
    arr_cont2_test = np.zeros((int(H/3),P*K*K))

    for l in range(K):
        h_cont1 = np.matmul(mat2,cont1[:,l])
        h_cont1_test = np.matmul(mat2,cont1_test[:,l])
        tile_cont1 = np.tile(np.reshape(h_cont1,(int(H/3),1)),P)
        tile_cont1_test = np.tile(np.reshape(h_cont1_test,(int(H/3),1)),P)
        arr_cont1[:,l*P:(l+1)*P] = tile_cont1
        arr_cont1_test[:,l*P:(l+1)*P] = tile_cont1_test
    
    for l1 in range(K):    
        h_cont2 = np.matmul(mat3,cont2[:,l1])
        h_cont2_test =  np.matmul(mat3,cont2_test[:,l1])
        #tile_cont2 = np.repeat(np.reshape(h_cont2,(int(H/3),1)),P*K,axis=1)
        tile_cont2 = np.tile(np.reshape(h_cont2,(int(H/3),1)),P*K)
        tile_cont2_test = np.tile(np.reshape(h_cont2_test,(int(H/3),1)),P*K)
        arr_cont2[:,l1*(P*K):(l1+1)*(P*K)] = tile_cont2
        arr_cont2_test[:,l1*(P*K):(l1+1)*(P*K)] = tile_cont2_test
    
    for p in range(P):
        h_stim = np.matmul(mat1,stim[:,p])
        h_stim_test = np.matmul(mat1,stim_test[:,p])
        arr_stim[:,p] = h_stim
        arr_stim_test[:,p] = h_stim_test
    
    arr_stim_in = np.tile(arr_stim,K**(2))
    arr_stim_in_test = np.tile(arr_stim_test,K**(2))
    arr_cont1_in = np.tile(arr_cont1,K)
    arr_cont1_in_test = np.tile(arr_cont1_test,K)
    h_big[:int(H/3),:] = arr_stim_in
    h_big[int(H/3):int(2*H/3),:] = arr_cont1_in
    h_big[int(2*H/3):,:] = arr_cont2
    h_big_test[:int(H/3),:] = arr_stim_in_test
    h_big_test[int(H/3):int(2*H/3),:] = arr_cont1_in_test
    h_big_test[int(2*H/3):,:] = arr_cont2_test
    
#    h_stim, h_cont, h_cont2 = decompose_from_composite(h_big,int(H/3),int(H/3),P,K)
#    print("shapes",h_stim.shape,h_cont.shape,h_cont2.shape)
#    
#    fig = plt.figure()
#    #plt.title(r'Decomposed h')
#    ax = fig.add_subplot(131)
#    ax.set_title(r'H stim')
#    ax2 = fig.add_subplot(132)
#    ax2.set_title(r'H cont')
#    ax3 = fig.add_subplot(133)
#    ax3.set_title(r'H cont2')
#    im = ax.imshow(h_stim)
#    im2 = ax2.imshow(h_cont)
#    im3 = ax3.imshow(h_cont2)
#    plt.colorbar(im)
#    plt.show()
    
    return h_big, h_big_test

###CHECK COVARIANCE STRUCTURE
#N=100
#M=100
#H=900
#P=50
#th = 0.8
#erf = erf1(th)
#h,h_test = generate_order_one_mixed_test_decoding(H,N,M,P)
#corrs = []
#for i in range(h.shape[1]):
#    over = (1/H)*np.dot(h[:,i],h_test[:,i])
#    corrs.append(over)
#delta_s_eff = 1 - np.mean(corrs)
#print("delta_s_eff is",delta_s_eff)
#
#o = 0.5*(np.sign(h - th) + 1)
#o_test = 0.5*(np.sign(h_test - th) + 1)
#
#d_ms = []
#for i in range(h.shape[1]):
#    dm = compute_delta_out(o_test[:,i],o[:,i])
#    coeff = 1/(2*erf*(1-erf))
#    d_ms.append(coeff*dm)
#
#dm_emp = np.mean(d_ms)
#print("dm_emp",dm_emp)
#
#dm_theory = (2/3)*erf_full(th,1,erf)
#print("dm_theory",dm_theory)



#f1 = compute_sparsity(o[:,np.random.randint(P)])
#f2 = compute_sparsity(o_test[:,np.random.randint(P)])
#print("f1 and f2",f1,f2)


#cov = (1/H)*np.matmul(h.T,h)
#cov_flatt = np.matrix.flatten(cov)
#
#plt.figure()
#plt.title(r'$\mathcal{M}=1$')
#plt.hist(cov_flatt,bins=50)
#plt.show()

#plt.figure()
#plt.title(r'$\mathcal{M}=2$')
#plt.imshow(cov)
#plt.colorbar()
#plt.show()
    

gaussian_func_2dim_onethird = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (1/3)**(2))))  \
                                        *np.exp(-(1./(2*(1 - (1/3)**(2))))*(x**(2) + y**(2) - 2*(1/3)*x*y))
                                       
gaussian_func_2dim_twothird = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (2/3)**(2)))) \
                                        *np.exp(-(1./(2*(1 - (2/3)**(2))))*(x**(2) + y**(2) - 2*(2/3)*x*y))                                        
                                        
gaussian_func_2dim_onehalf = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (1/2)**(2)))) \
                                        *np.exp(-(1./(2*(1 - (1/2)**(2))))*(x**(2) + y**(2) - 2*(1/2)*x*y))  
                                        
gaussian_func_2dim_onequarter = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (1/4)**(2)))) \
                                        *np.exp(-(1./(2*(1 - (1/4)**(2))))*(x**(2) + y**(2) - 2*(1/4)*x*y))                                         

gaussian_func_onethird_eo = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (1/3)**(2)))) * x * y  \
                                        *np.exp(-(1./(2*(1 - (1/3)**(2))))*(x**(2) + y**(2) - 2*(1/3)*x*y))

gaussian_func_twothird_eo = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (2/3)**(2)))) * x * y \
                                        *np.exp(-(1./(2*(1 - (2/3)**(2))))*(x**(2) + y**(2) - 2*(2/3)*x*y))

gaussian_func_onehalf_eo = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (1/2)**(2)))) * x * y \
                                        *np.exp(-(1./(2*(1 - (1/2)**(2))))*(x**(2) + y**(2) - 2*(1/2)*x*y)) 
                                        
gaussian_func_onequarter_eo = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (1/4)**(2)))) * x * y \
                                        *np.exp(-(1./(2*(1 - (1/4)**(2))))*(x**(2) + y**(2) - 2*(1/4)*x*y))                                                                              


def two_pt(th,pk=1/3):
    if pk == 1/3:
        res = integrate.dblquad(gaussian_func_2dim_onethird, th, np.inf, lambda x: th, lambda x: np.inf)
    elif pk == 2/3:
        res = integrate.dblquad(gaussian_func_2dim_twothird, th, np.inf, lambda x: th, lambda x: np.inf)
    elif pk == 1/2:
        res = integrate.dblquad(gaussian_func_2dim_onehalf, th, np.inf, lambda x: th, lambda x: np.inf)
    elif pk == 1/4:
        res = integrate.dblquad(gaussian_func_2dim_onequarter, th, np.inf, lambda x: th, lambda x: np.inf)    
    elif pk == 0: #To check if unimodal results recovered
        res = integrate.dblquad(gaussian_func_2dim_easy, th, np.inf, lambda x: th, lambda x: np.inf)
    return res[0]

def eo_multimod(th,pk=1/3):
    if pk == 1/3:
        res = integrate.dblquad(gaussian_func_onethird_eo, th, np.inf, lambda x: th, lambda x: np.inf)
    elif pk == 2/3:
        res = integrate.dblquad(gaussian_func_twothird_eo, th, np.inf, lambda x: th, lambda x: np.inf)
    elif pk == 1/2:
        res = integrate.dblquad(gaussian_func_onehalf_eo, th, np.inf, lambda x: th, lambda x: np.inf)
    elif pk == 1/4:
        res = integrate.dblquad(gaussian_func_onequarter_eo, th, np.inf, lambda x: th, lambda x: np.inf)    
    elif pk == 0:
         res = integrate.dblquad(gaussian_func_2dim_extra, th, np.inf, lambda x: th, lambda x: np.inf)   
    return res[0]

def squared_integral(th,pk=1/3):
    if pk == 1/3:
        res = integrate.dblquad(r_integral_onethird, th, np.inf, lambda x: th, lambda x: np.inf)
    elif pk == 2/3:
        res = integrate.dblquad(r_integral_twothird, th, np.inf, lambda x: th, lambda x: np.inf)
    elif pk == 1/2:
        res = integrate.dblquad(r_integral_onehalf, th, np.inf, lambda x: th, lambda x: np.inf)
    return res[0]


#THEOY FOR CO-EFFICIENTS IN INTEFERENCE TERM
def c1_theory(pk):
    frac = pk**(2)/((1-(pk)**(2))**(2))
    return frac

def c2_theory(pk):
    numer = 1
    denom = (1 - pk**(2))**(4)
    return numer/denom
        
def compute_excess_over(N,th,pk):
    i2 = two_pt(th,pk)
    erf = erf1(th)
    eo = eo_multimod(th,pk)**(2)
    
    ratio1 = 1 + (1/N)*c1
    ratio2 = (1/N)*c2_theory
    
    return ratio1*(i2**(2)) + erf**(4) - 2 * (f**(2)) * i2 + ratio2*eo


###COMPUTE PR - ALSO FOR REGULAR CEMS
from interpolate import prob_unimod_stim, prob_unimod_cont, prob_across_all, prob_across_one, prob_across_cont
compute_pr = False
if compute_pr:
    ths = np.linspace(0,2.1,20)
    #ths = [0.2]
    #K_list = [1,5,10]
    #K_list = [5]
    pr_emps = np.zeros(len(K_list))
    pr_emps_std = np.zeros(len(K_list))
    pr_theorys = np.zeros(len(K_list))
    fp_corrs = np.zeros(len(K_list))
    cods = np.zeros(len(ths))
    N=100
    M=100
    P=200
    K=5
    H=1200
    index=3
    p1 = prob_across_all(P,K)
    p2 = prob_across_one(P,K)
    p3 = prob_across_cont(P,K) 
    p4 = prob_unimod_stim(P,K)
    p5 = prob_unimod_cont(P,K)
    print("all probs",p1,p2,p3,p4,p5)
    n_trials = 5
    for i,th in enumerate(ths):
        pr_trials = []
        for n in range(n_trials):
            erf = erf1(th)
            stim = make_patterns(N,P)
            #h = random_proj_generic(H,stim,0)
            #h,h_test = generate_order_two_mixed_test_decoding(H,N,M,P)
            if index == 3:
                h,h_test = generate_order_full_mixed_test(H,N,M,P,K,0.1,0.1)
            elif index == 2:
                h,h_test = generate_order_two_mixed_test(H,N,M,P,K,0.1,0.1)
            o = 0.5*(np.sign(h-th)+1)
            #print("shape of o",o.shape)
            f = compute_sparsity(o[:,np.random.randint(o.shape[1])])
            #print("f is",f)
            
            o_in = o - f
            cov = np.matmul(o_in,o_in.T)
            pr_emp = compute_pr_eigvals(cov)
            print("pr_emp is",pr_emp)
            pr_trials.append(pr_emp)
            
            ratio = 1
            print("ratio is",ratio)
            
            if index == 3:
                pk=1/3
                feff2 = two_pt(th,pk)
                q2 = erf*(1-erf)
                i4 = feff2
                denom2_onethird = (i4 - erf**(2))**(2)/(q2**(2))
                print("denom2_onethird",denom2_onethird)
                
                pk2=2/3
                feff3 = two_pt(th,pk2)
                i4_2 = feff3
                denom2_twothird = (i4_2 - erf**(2))**(2)/(q2**(2))
                print("denom2_twothird",denom2_twothird)
                
                denom2_eo_main = (1/(3*N))*excess_over_theory(th,erf)**(2)
                print("denom2",denom2_eo)
                
                eo_acc_one = (2/(9*N))*(eo_multimod(th,1/3)**(2))/(q2**(2))
                eo_acc_cont = (2/(9*M))*(eo_multimod(th,1/3)**(2))/(q2**(2))
                eo_acc_stim = (1/(9*N))*(eo_multimod(th,2/3)**(2))/(q2**(2))
                eo_acc_cont = (1/(9*M))*(eo_multimod(th,2/3)**(2))/(q2**(2))
                
                denom2 = (p2)*(denom2_onethird + eo_acc_one) + p3*(denom2_onethird + eo_acc_cont) + p1*denom2_eo_main \
                + (p4)*(denom2_twothird + eo_acc_stim) + (p5)*(denom2_twothird + eo_acc_cont)
                
            elif index == 2:
                pk=1/4
                feff2 = two_pt(th,pk)
                q2 = erf*(1-erf)
                i4 = feff2
                denom2_onequarter = (i4 - erf**(2))**(2)/(q2**(2))
                print("denom2_onequarter",denom2_onequarter)
                
                pk2=1/2
                feff3 = two_pt(th,pk2)
                i4_2 = feff3
                denom2_onehalf = (i4_2 - erf**(2))**(2)/(q2**(2))
                print("denom2_onehalf",denom2_onehalf)
                
                denom2_eo_main = (3/(8*N))*excess_over_theory(th,erf)**(2)
                print("denom2_main",denom2_eo_main)
                
                eo_acc_one = (5/(16*N))*(eo_multimod(th,1/4)**(2))/(q2**(2))
                eo_acc_cont = (1/(8*M))*(eo_multimod(th,1/4)**(2))/(q2**(2))
                eo_acc_stim = (1/(8*N))*(eo_multimod(th,1/2)**(2))/(q2**(2))
                eo_acc_cont = (1/(4*M))*(eo_multimod(th,1/2)**(2))/(q2**(2))
                
                denom2 = (p2)*(denom2_onequarter + eo_acc_one) + p3*(denom2_onequarter + eo_acc_cont) + p1*denom2_eo_main \
                + (p4)*(denom2_onehalf + eo_acc_stim) + (p5)*(denom2_onehalf + eo_acc_cont)
                 
            Peff = P*K*K
            #Peff = P
            denom1 = (1/(H*Peff))
            #denom2 = (1/N)*excess_over_theory(th,erf1(th))**(2)
            pr_theory = 1/(denom1 + denom2 + (1/Peff) + (1/H)*ratio)
            print("pr theory",pr_theory)
            
        pr_theorys[i,j] = pr_theory
        pr_emps[i,j] = np.mean(pr_trials)
        pr_emps_std[i,j] = np.std(pr_trials)
        fp_corrs[i,j] = denom2
        cods[i] = erf
        
    plt.figure()
    plt.title(r'Dimensionality, $\mathcal{M}=3$')
    colors = itertools.cycle(('green','blue','red','black'))
    colors_ver = itertools.cycle(('lightgreen','lightskyblue','lightcoral','grey'))
    clr = next(colors)
    clr_theory = next(colors_ver)
    plt.errorbar(fp_corrs[:,i],pr_emps[:,i],yerr=pr_emps_std[:,i],color='blue',fmt='s--',
                         capsize=5, markeredgewidth=2,label=r'$K={}$'.format(K))
    #plt.plot(fp_corrs[:,i],pr_theorys[:,i],'--',color=clr_theory)
    plt.xlabel(r'$\frac{\langle \mathcal{I}_{4} \rangle}{f^{2}(1-f)^{2}}$',fontsize=14)
    plt.ylabel(r'$\mathcal{D}$',fontsize=14)
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title(r'$\langle \mathcal{I}_{4} \rangle$ vs. coding, $\mathcal{M}=3$')
    colors = itertools.cycle(('green','blue','red','black'))
    colors_ver = itertools.cycle(('lightgreen','lightskyblue','lightcoral','grey'))
    clr = next(colors)
    clr_theory = next(colors_ver)
    plt.plot(cods,fp_corrs[:,i],'o-',color='blue',label=r'$K={}$'.format(K))
    plt.xlabel(r'$f$',fontsize=14)
    plt.ylabel(r'$\frac{\langle \mathcal{I}_{4} \rangle}{f^{2}(1-f)^{2}}$',fontsize=14)
    plt.legend()
    plt.show()


"""Compute readout error & SNR """
def hebbian_contextual_decoding(H,N,M,P,th,index=3):
    """Always {0,1}"""
    n_real = 50
    errors = np.zeros(n_real)
    len_test = P
    erf = erf1(th)
    for j in range(n_real):
        stim = make_patterns(N,P)
        labels = np.zeros(P)
        for i in range(P):
            labels[i] = make_labels(0.5)
        
        if index == 1:
            h,h_test = generate_order_one_mixed_test_decoding(H,N,M,P)
        elif index == 2:
            h,h_test = generate_order_two_mixed_test_decoding(H,N,M,P)
        elif index == 3:
            h,h_test = generate_order_full_mixed_test_decoding(H,N,M,P)
        
        o = 0.5*(np.sign(h - th) + 1)
        o_test = 0.5*(np.sign(h_test - th) + 1)
        f = compute_sparsity(o[:,np.random.randint(P)])
        print("coding",f)

        o_in = o - erf
        o_test_in = o_test - erf
        
        
        w_hebb = np.matmul(o_in,labels) 
        
        labels_test = labels
        
        #print("shape hebbian weights",w_hebb.shape)
        stabs = []
        d_outs = []
        for m in range(len_test):
            stab = labels_test[m]*np.dot(w_hebb,o_test_in[:,m])
            stabs.append(stab)
            d_out = (1/(2*erf*(1-erf))) * compute_delta_out(o_test_in[:,m],o_in[:,m])
            d_outs.append(d_out)
        
        d_out_mean = np.mean(d_outs)
        d_std = np.std(d_outs)
        print("d_out_mean",d_out_mean)
        
        if index==1:
            d_in = 2/3
        elif index==2:
            d_in = 1/2
        elif index==3:
            d_in = 2/3
       
        #print("index is",index,"d_in is",d_in)
        d_out_theory = erf_full(th,d_in,erf)
        print("d_out theory",d_out_theory)
        
        err = (len(np.where(np.asarray(stabs)<0)[0]))/(len(stabs))
        #print("numer of err is",len(np.where(np.asarray(stabs)<0)[0]))
        #print("denom of err is",len(stabs))
        errors[j] = err


    err_mean = np.mean(errors)
    err_std = np.std(errors)
    print("ERR EMPIRICAL",err_mean)
    print("err std",err_std)
  
    numer_theory = (1 - d_out_theory)
    
    if index==1:
        pk = 2/3 #No 1/3 peak in contextual model
    elif index==2:
        pk = 1/2
    elif index==3:
        pk = 2/3
    
    q2 = erf*(1-erf)
    tp_extra = two_point_extra(th,pk)
    fp_theory = (tp_extra - erf**(2))**(2)
    print("fp_theory",fp_theory)


    fp_theory_in = (1/N)*excess_over_theory(th,erf)**(2) 
    print("fp_theory_in",fp_theory_in)
    
    denom_theory = P/H + P * fp_theory_in
    print("first and second term in denom",P/H,P*fp_theory_in)
  
    snr_theory = (numer_theory**(2))/denom_theory
    print("numer",numer_theory**(2))
    print("denom",denom_theory)
    print("snr_theory",snr_theory)
    err_theory = erf1(np.sqrt(snr_theory))
    print("err theory",err_theory)

    return err_mean, err_std, err_theory, f, d_out_mean, d_out_theory

#N=100
#M=100
#H=3600
#P=10
#th=0.2
#err_mean, err_std, err_theory, f, d_emp3, d_theory3 = hebbian_contextual_decoding(H,N,M,P,th,index=3)

run_sweep_ratio_ctxt = True
"""
3 sweeps:
    1. For different f
    2. For different Delta_in
    3. For different alpha
"""
if run_sweep_ratio_ctxt:
    N=100 #N larger for {+,-}
    #H=20000
    bool_ = False #True for {+,-}
    bool2_ = False
    #P=100
    if bool_:
        #thress = [0.4,0.6,1.5] #For {+,-}
        thress = list(np.linspace(0.0,0.8,8)) + [1.2,1.5,2.1] #For optimal sparsity
        #thress = [0.6]
    else:
        #thress = [0.5,1.5,1.9] #For {0,1}
        thress = np.linspace(0.01,2.9,10)
        #thress = [0.5,2.2,2.8] #For ultra-sparse
    #deltas = [0.02,0.1,0.3] #HERE THRESS "=" DELTAS
    alphas = [0.1,0.5,2.0]
    delta = 0.5
    #thress = [1,500] #HERE THRESS "=" R
    cods = np.zeros((len(thress)))
    P_arr = np.linspace(0.2*N,4*N,10)
    P_list = [int(p) for p in P_arr]
    H_arr = np.linspace(1.0*N,20*N,10)
    H_list = [int(p) for p in H_arr]
    H_list = [100,200,400,600,800,1000,1200,1400,1600,1800,2000]
    err_means = np.zeros((len(H_list),len(alphas)))
    err_stds = np.zeros((len(H_list),len(alphas)))
    err_theorys = np.zeros((len(H_list),len(alphas)))
    for j,h in enumerate(H_list):
        for i,a in enumerate(alphas):
            th=0.8
            #delta = th
            #delta = d
            H = int(h)
            #H = 2700
            P=int(a*N)
            print("P is",P)
            #print("H is",H)
            err_mean, err_std, err_theory, f, d_emp3, d_theory3 = hebbian_contextual_decoding(H,N,M,P,th,index=2)
            err_means[j,i] = err_mean
            print("EMP ERR",err_mean)
            err_theorys[j,i] = err_theory
            print("theoretical error",err_theory)
            err_stds[j,i] = err_std
            
        #cods[j] = f
    #np.savetxt("err_mean_largeN_R=10_modtheta_diffP.csv",err_means,delimiter=',')
    #np.savetxt("err_theory_largeN_R=10_modtheta_diffP.csv",err_theorys,delimiter=',')
    #np.savetxt("snrs_largeN_R=10_modtheta_diffP.csv",snrs,delimiter=',')
    #
    #
    #err_means = np.genfromtxt('err_mean_largeN_R=10_modtheta_diffP.csv',delimiter=',')
    #err_theorys = np.genfromtxt('err_theory_largeN_R=10_modtheta_diffP.csv',delimiter=',')
    colors = itertools.cycle(('green','blue','red','black'))
    colors_ver = itertools.cycle(('lightgreen','lightskyblue','lightcoral','grey'))
    plt.figure()
    plt.title(r'Contextual decoding $\mathcal{M}=2$, $f=0.2$',fontsize=12)
    for i,a in enumerate(alphas):
        clr = next(colors)
        clr_theory = next(colors_ver)
        print("lenghts",len(H_arr),len(err_means[:,i]))
        plt.errorbar((1/N)*np.asarray(H_list),err_means[:,i],yerr=0.2*err_stds[:,i],color=clr,fmt='-',
                     capsize=5, markeredgewidth=2,label=r'$\beta={}$'.format(a))
        #plt.plot((1/N)*np.asarray(H_list),err_theorys[:,i],'--',color=clr_theory)
    plt.xlabel(r'$\mathcal{R}$',fontsize=14)
    plt.ylabel(r'Readout error',fontsize=14)
    plt.legend(fontsize=10)
    plt.show()

