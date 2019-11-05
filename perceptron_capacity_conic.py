#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capacity (USING LINEAR CONSTRAINTS) - perceptron with coding
"""
#import sys
#sys.path.append('/Users/arvingopal5794/Documents/cognitive_control')
import numpy as np
from numpy import linalg as LA
import scipy as sp
from scipy.optimize import linprog
from cvxopt import matrix, solvers
import pulp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def generate_pm_with_coding(N,f):
    """f: Fraction of plus ones"""
    v = np.zeros(N)
    for i in range(N):
        if np.random.rand() > 1-f:
            v[i] = 1
        else:
            v[i] = -1
            #v[i] = 0
    return v


def make_patterns(N,P,cod=0.5):
    matrix = np.zeros((N,P))
    for i in range(P):
        vec = generate_pm_with_coding(N,cod)
        matrix[:,i] = vec
    
    return matrix

        
def make_labels(f):
    if np.random.rand()>1-f:
        lbl = 1
    else:
        lbl = -1
        
    return lbl


def perceptron_storage(patterns,kappa=0.001,cod_l=0.5,thres=0):
    """
    Patterns: P columns of patterns
    
   WORKS IF "REVISED SIMPLEX" USED (instead of "SIMPLEX")
    
    """
    
    N=patterns.shape[0]
    thres_vec = np.zeros(N)
    thres_vec.fill(thres)
    #c = [0] * N + [1] #Plus one to enforce min 1
    c = [0] * N
    w_bounds = (None,None)
    A = []
    B = []
#    Aeq1 = [0]*N + [1] #Margin is N+1th weight
#    #Aeq2 = [1]*N + [0] #Enforce normalization of simplex
#    Aeq = [Aeq1]
#    beq = [1]
    
    labels = []    
    for m in range(patterns.shape[1]):
        #label = 0.5*(1 + make_labels(0.5))
        label = make_labels(cod_l)
        labels.append(label)
        #print("label",label)
        lista = list(-label/(np.sqrt(N)) * patterns[:,m].T)
        #A.append(lista)
        A.append(lista)
        #print("A",A)
        B.append(-kappa - label*thres)
        
    
    res = linprog(c, A_ub=A, b_ub=B,bounds=w_bounds,method='revised simplex')
    
#    print('weights',res.x,res.x.shape)
#    print('patterns shape',patt.shape)
#    print('dot shape',np.dot(res.x,patt),np.dot(res.x,patt).shape)
#    converged_to_solution = np.asarray(labels).T*np.dot(res.x,patterns)-thres_vec>=kappa
#    print("dot product",np.asarray(labels).T*np.dot(res.x,patterns)-thres_vec>=kappa ) 
#    if not converged_to_solution:
#        print("Did not find solution")
    
    print("status",res.status)
    print("message",res.message)
    
    return res.x, res.status

#
#N=200
#P=400
#cod=0.5
#patt = make_patterns(N,P,cod)
#w,sol = perceptron_storage(patt,kappa=0.001,cod_l=cod)



def perceptron_storage_cone(pattern,kappa=0.,cod_l=0.5,thres = 0.):
    """
    Using the python cone program
    """
    v1 = np.array([-3.,1.,1.,-5.,]) #v analogous to patterns
    v2 = np.array([1.,2.,-1.,-1.])
    v3 = np.array([0.,-1.,3.,0.])
    v4 = np.array([1.,-1.,0.,-1.])
    a = np.vstack((v1,v2,v3))
    list_a = []
    for i in range(a.shape[0]):
        list_a.append(list(a[i,:]))
        
    #A = matrix(np.asarray(list_a))
    A = matrix(a)
    print("A",A)
    b = matrix([6.,4.,3.])
    print("b",b)
    #c1 = np.array([0.,0.,0.,0.])
    c1 = np.ones(4)
    c = matrix(c1)
    print("c",c)
    
    dims = {'l': 3, 'q': [], 's': []}
    sol = solvers.conelp(c, A, b,dims)
    print("status",sol['status'])
    print("values",sol['x'])  
    
#    N = pattern.shape[0]
#    list_patt = []
#    list_labels = []
#    list_b = []
#    for j in range(pattern.shape[1]):
#        lbl = make_labels(cod_l)
#        list_b.append(-kappa - (lbl/np.sqrt(N))*thres)
#        list_patt.append((-lbl/np.sqrt(N))*pattern[:,j])
#    mat_patt = np.vstack(list_patt[:])
#    
#    G = matrix(mat_patt)
#    print("rank mat_patt",LA.matrix_rank(mat_patt))
#    print("G",G)
#    h = matrix(list_b)
#    print("h",h)
#    C = np.asarray([1.]*N)
#    c = matrix(C)
#    print("c",c)
#    
#    #dims = {'l': N+1, 'q': [], 's': []}
#    sol = solvers.conelp(c, G, h)
#    print("status",sol['status'])
    
    return sol    

#N=5
#P=4
#patt = make_patterns(N,P,cod=0.5)
#w, sol = perceptron_storage(patt,kappa=0.,cod_l=0.5)
#sol = perceptron_storage_cone(patt,cod_l=0.5)

   
    
def simplest_case(kappa=0.001):
    """
    Simplest toy example for classification
    
    """
    
    p1 = [1,1,0]
    p2 = [1,2,0]
    p3 = [-1,-1,0]
    p_fail = [-2,-2,0] #Point that makes the program fail - same label as p1
    #A = [[-1*l for l in p1],[-1*l for l in p2],[1*l for l in p3],[-1*l for l in p_fail]]
    A = [[-1*l for l in p1],[-1*l for l in p2],[1*l for l in p3]]
    Aeq = [[0,0,1]] #First one to enforce the margin, second to enforce the normalization (simplex)
    print("shape of A",A)
    b = [-kappa,-kappa,-kappa]
    #b = [-kappa,-kappa,-kappa,-kappa]
    beq = [1]
    w_bounds = []
    for i in range(len(p1)-1):
        w_bounds.append((None,None))
    w_bounds.append((kappa,kappa)) #Set the bounds on the last weight to be equal to kappa
    c = [0,0,1]
    res = linprog(c, A, b,A_eq=Aeq,b_eq=beq,bounds=w_bounds,method="simplex")
    print("message",res.message)
    print("RESULT STATUS",res.status)
    
    w = res.x
    
    print("projection of weights onto p1",np.dot(np.asarray(res.x),np.asarray(p1)))
    print("projection of weights onto p3",np.dot(np.asarray(res.x),np.asarray(p3)))
    print("projection of weights onto p_fail",np.dot(np.asarray(res.x),np.asarray(p_fail)))
    
    ###PLOT SUCCESS (3 POINTS) VS. FAILURE (WITH P_FAIL)
    fig_suc = plt.figure()
    ax1 = fig_suc.add_subplot(111)
    legend_elements = [Line2D([0], [0], marker='o', color='black', label=r'$+1$ class',
                          markerfacecolor='black', markersize=15),
                        Line2D([0], [0], marker='o', color='green', label=r'$-1$ class',
                          markerfacecolor='green', markersize=15)]
    if kappa == 1:
        text = 'inseparable'
    else:
        text = 'separable'
    plt.title(r'Toy {} case (colinear), $\kappa = {}$'.format(text,kappa))
    ax1.scatter(p1[0],p1[1],color='black',s=100)
    ax1.scatter(p2[0],p2[1],color='black',s=100)
    ax1.scatter(p3[0],p3[1],color='green',s=100)
    #ax1.scatter(p_fail[0],p_fail[1],color='black',s=100)
    
    ax1.set_xlim(-2.2,2.2)
    ax1.set_ylim(-2.2,2.2)
    origin = [0], [0]
    z = np.array([0,0,1])
    perp = np.cross(np.asarray(w),z)
    x = np.linspace(-0.8,0.8,100)
    grad = perp[1]/perp[0]
    y = grad*x
    y2 = grad*x + kappa/(np.linalg.norm(np.asarray(w)))
    y3 = grad*x - kappa/(np.linalg.norm(np.asarray(w)))
    ax1.quiver(*origin, w[0], w[1], color=['r'], scale=21)
    ax1.annotate('({}, {})'.format(p1[0],p1[1]),xy=[l + 0.05 for l in p1[:2]], textcoords='data')
    ax1.annotate('({}, {})'.format(p2[0],p2[1]),xy=[l + 0.05 for l in p2[:2]], textcoords='data')
    ax1.annotate('({}, {})'.format(p3[0],p3[1]),xy=[l + 0.05 for l in p3[:2]], textcoords='data')
    #ax1.annotate('({}, {})'.format(p_fail[0],p_fail[1]),xy=[l + 0.05 for l in p_fail[:2]], textcoords='data')
    ax1.annotate(r'$\vec{}$ = ({}, {})'.format('w',np.round(w[0],2),np.round(w[1],2)),xy=[0.10,0])
    #ax1.annotate(r'$y = \vec{} \cdot \vec{} + \kappa/|\vec{}|$'.format('w','x','w'),xy=[0.7,-0.2])
    #ax1.annotate(r'$y = \vec{} \cdot \vec{}$'.format('w','x'),xy=[0.7,-0.8])
    #ax1.annotate(r'$y = \vec{} \cdot \vec{} - \kappa/|\vec{}|$'.format('w','x','w'),xy=[-1.0,0.15])
    ax1.plot(x,y,'-r')
    #ax1.plot(x,y2,'--r')
    #ax1.plot(x,y3,'--r')
    #plt.xticks([], [])
    #plt.yticks([], [])
    
    ax1.legend(handles=legend_elements, loc='upper left')
    
    plt.show()
    
    
    
    return w
    

def func_evaluate_capacity_counting(f=0.5):

    N_list = [50]
    len_P = 20
    n_real = 1
    sucss_matrix = np.zeros((len(N_list),len_P))
    P_lists = {}
    for i,N in enumerate(N_list):
        P_list = np.linspace(0.2*N,4*N,len_P)
        P_lists[i] = P_list
        P_crits = []
        for j,P in enumerate(P_list):
            sucs = []
            Pcaps = []
            for n in range(n_real):
                patt = make_patterns(N,int(P),cod=f)
                w, status = perceptron_storage(patt)
                if status == 0:
                    sucs.append(1)
                else:
                    sucs.append(0)
                    Pcaps.append(int(P))
            print("number in matrix",np.mean(sucs))
            sucss_matrix[i,j] = np.mean(sucs)
         
            P_crits.append(np.mean(Pcaps))
        alpha_cs = (1/N)*np.asarray(P_crits)
        print("alpha_cs",alpha_cs)

    plt.figure()
    plt.title(r'Perceptron capacity,f={}'.format(f),fontsize=16)
    for i,N in enumerate(N_list):
        plt.plot((1/N)*P_lists[i],sucss_matrix[i,:],'s-',label=r'$N={}$'.format(N))
    plt.xlabel(r'$\alpha$',fontsize=14)
    plt.ylabel('Prob of success',fontsize=14)
    plt.legend(fontsize=12)
    plt.show()
    
    
def func_evaluate_alphac_given_kappa(f=0.5):
    
    """
    Fix kappa, find what the maximum alpha is
    """
    
    #len_K = 5
    #K_list = np.linspace(0.001,1,len_K)
    len_K=1
    K_list = [0.0001]
    N = 100
    len_P = 10
    n_real = 1
    sucss_matrix = np.zeros((len_K,len_P))
    alpha_cs = []
    P_list = np.linspace(0.2*N,4*N,len_P)
    for i,K in enumerate(K_list):
        for j,P in enumerate(P_list):
            sucs = []
            for n in range(n_real):
                patt = make_patterns(N,int(P),cod=f)
                w, status = perceptron_storage(patt,kappa=K)
                if status == 0:
                    sucs.append(1)
                else:
                    print("CANNOT STORE!",int(P),"N is",N)
                    sucs.append(0)
            print("number in matrix",np.mean(sucs))
            sucss_matrix[i,j] = np.mean(sucs)
            
        ##Find alpha_c by approximating P at 0.5
        print("success matrix",sucss_matrix[i,:])
        print("first indices",np.where(sucss_matrix[i,:]<=0.55)[0])
        print("second indices",np.where(sucss_matrix[i,:]>=0.45)[0])
        ind_1 = np.where(sucss_matrix[i,:]<=0.55)[0]
        ind_2 = np.where(sucss_matrix[i,:]>=0.45)[0]
        set1 = set(ind_1)
        set2 = set(ind_2)
        if list(set1.intersection(set2)) == []:
            #Take P corresponding to p=1 and p=0, and average
            Pl = P_list[ind_2[-1]]
            print("Pl",Pl)
            Ph = P_list[ind_1[0]]
            print("Ph",Ph)
            #Pav = np.mean([Pl,Ph])
            Pav = Ph
            alpha_c = (1/N)*Pav
            
        else: 
            int_ = list(np.asarray(set1.intersection(set2)))
            ind_ = np.mean(P_list[int_])
            alpha_c = (1/N)*ind_
            
        alpha_cs.append(alpha_c)
    
    plt.figure()
    plt.title(r'Capacity curve,f={}'.format(f),fontsize=16)
    plt.plot(np.round(K_list,2),alpha_cs,'s-',label=r'$f={}$'.format(f))
    plt.xlabel(r'$\kappa$',fontsize=14)
    plt.ylabel(r'$\alpha_c$',fontsize=14)
    plt.legend(fontsize=12)
    plt.show()
    
    
def create_all_black(N,P):
    mat = np.zeros((N,P))
    for i in range(mat.shape[1]):
        mat[:,i] = np.ones(N)
        
    return mat

#k=0.001
#N=150
#P=200
#f=0.5
#patt = make_patterns(N,P,cod=f)
#w,succ = perceptron_storage(patt,kappa=0.001,cod_l=f)




