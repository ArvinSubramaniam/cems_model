# cems_model

The code here describes results in "Cortical Models of Expansive Non-linear Mixed Selectivity" form my Masters thesis. The files can be roughly divided as follows:

##1. Capacity: (Chapters 2 and 3)

 (a) perceptron_capacity_conic.py - gives the main functions for generating datum, labels, and performing linear program to check feasability.
 
 (b) perceptron_capacity_fix_rank.py - runs simulations to reproduce results of Shinzato,Kabashima ('08)
 
 (c) fusi_barak_rank.py - functions for generating contextual datasets (2 modalities) for the CDP.
 
 (d) context_dep_capacity.py - simulations for the CDP
 
##2. hebbian_readout.py - functions that implement two forms of simples Hebbain learning (Appendix of Chapter 4)
 
##3. Unimodal model: (Chapters 4 and 5.1)

 (a) random_expansion.py - functions for random feed-forward expansions
 
 (b) sparseness_expansion.py -  To reproduce results of Babadi, Sompolinsky (2014)
 
 (c) relu_nonlinearity.py - Order parameters, dimensionality and readout error for ReLU non-linearity (Ch. 5.1)
 
##4. dimensionality_disentanglement.py - functions that calculate dimensionality empirically and theoretically (Chapter 4)

##5. Multi-modal model: (Chapters 3 and 5)

  (a) interpolate.py - implements cems model under given interpolation scheme. Can obtain capacity, readout error, cluster size, etc.
  
  (b) contextual_decoding.py - implements contextual decoding model (Chapter 5.1)
