#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perceptron capacity - USING PECEPTRON LEARNING ALGORITHM
"""

from fusi_barak_rank import *
from perceptron_capacity_conic import *
from matplotlib.lines import Line2D

def make_patterns_dict(N,P,N_samples=50):
    """
    Makes dict of patterns and labels for N_s=50 samples
    """
    patterns = {}
    labels = {}
    for n in range(N_samples):
        patterns[n] = make_patterns(N,P)
    for n in range(N_samples):
        label = []
        for p in range(P):
            label.append(make_labels(0.5))
        labels[n] = label
    
    return patterns, labels

def perceptron_learning_rule(N_list,len_P,T=500,eta=0.01):
    """
    Perceptron learning rule - count number of dichotomies
    T:Epochs
    """
    num_sep = np.zeros((len(N_list),len_P))
    
    for i,N in enumerate(N_list):
        P_list = np.linspace(0.5*N,4*N,len_P)
        print("P_list",P_list)
        for j,P in enumerate(P_list):
            count = 0
            #w = np.zeros(N)
            #w = np.random.normal(0,1,(N))
            data, labels = make_patterns_dict(N,int(P))
            for n in range(len(data)):
                #print("n",n)
                #print("count",count)
                w = np.random.normal(0,1,(N))
                w_old = np.copy(w)
                #w = np.ones(N)
                diffs = []

                for t in range(T):#Epochs
                    for m in range(int(P)):
                        #print("m",m)
                        if labels[n][m] * np.dot(w,data[i][:,m]) < 0:
                            w += eta * 1/(np.sqrt(N)) * labels[n][m] * data[i][:,m]
                            diff = np.abs(w - w_old)
                            diffs.append(diff)
                            #print("updating!, epoch{}".format(t))
                            #print("diff", diff)
                        elif labels[n][m] * np.dot(w,data[i][:,m]) >= 0:
                            w += 0
                            #print("NOT updating!, epoch{}".format(t))
                            diff = np.abs(w - w_old)
                            diffs.append(diff)
                            #print("diff", diff)
                        w_old = np.copy(w)
                    
                if (diffs[-1] - diffs[-20]).all() < 10e-6:
                    print("converged for 20 steps, n = {}!".format(n))
                    count += 1   
            
            print("counts before averaging",count)           
            print("num_dichom",count/(len(data)))
            num_sep[i,j] = count/(len(data))
    
    print("Plist",P_list)
    print("(1/N)*Plist",(1/N)*P_list)        
    alpha_cs = []
    for i,N in enumerate(N_list):
        print("num_sep",num_sep[i,:])
        ind1 = set(np.where(num_sep[i,:]<=0.6)[0])
        print("ind1",ind1)
        ind2 = set(np.where(num_sep[i,:]>=0.4)[0])
        print("ind2",ind2)
        ind_int = list(ind1.intersection(ind2))
        print("intersection",ind_int)
        ind_midd = int(np.median(ind_int))
        print("ind_midd",ind_midd)
        alphac = (1/N)*P_list[ind_midd]
        alpha_cs.append(alphac)
        print("alpha_c",alphac)
    
            
    return num_sep, P_list, alpha_cs



N_list = [50]
len_P = 4
num, Plist, alpha_cs = perceptron_learning_rule(N_list,len_P)
plt.figure()
plt.title(r'Perceptron capacity,$\alpha_c = {}$'.format(np.round(alpha_cs[0],2)),fontsize=18)
for i,N in enumerate(N_list):
    plt.plot((1/N)*Plist,num[i,:],'x-',label=r'$N={}$'.format(N))
plt.xlabel(r'$\alpha$',fontsize=14)
plt.ylabel('Probability of separation',fontsize=14)
plt.legend()
plt.show()
