#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, time, os, csv, pdb
import pandas as pd
import scipy as sp
"""http://docs.scipy.org/doc/scipy-0.15.1/reference/optimize.html
   http://docs.scipy.org/doc/scipy/reference/optimize.nonlin.html"""
from scipy.optimize import minimize
from scipy import stats
nan = '\x00\x00\x00\x00\x00\x00\xf8\x7f'
import matplotlib as mpl
import matplotlib.pyplot as plt
import functools
import numdifftools as nd
import copy
import random
import pickle


from .utils import *

from .bidding_models import *

from .likelihoo_functions import *



project = "neuroshyuka" # "neuroshyuka" or "econoshyuka"

initbid = {}
if project == "neuroshyuka":
    fn = os.path.dirname(os.path.dirname(os.getcwd())) + os.path.sep + 'data' + os.path.sep + 'shyuka.csv'
elif project == "econoshyuka":
    fn = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + '/Econoshyuka/logs/econoshyuka.csv'

with open(fn, 'r') as csvfile:
    fr = csv.reader(csvfile)  
    df = [row for row in fr]

D = pd.read_csv(fn)
S = D['sid'].unique()
if project == 'econoshyuka':
    only_cond = {'d'} #  bcd{'b','c','d'}
    D = D.drop([i for i in D.index if D.ix[i,'cond'] not in only_cond])
    D2 = D
    if only_cond == {'c'}:
        fnp = os.path.dirname(os.path.dirname(os.getcwd())) + os.path.sep + 'Data' + os.path.sep + 'PrerecordedData.csv'
        with open(fnp, 'r') as csvfile:
            fr = csv.reader(csvfile)  
            df = [row for row in fr]
        Dp = pd.read_csv(fnp, header = None)      
        opp_role = {'seller_NC': list(range(24)), 'seller_SC': list(range(24,48)), 'buyer_BC': list(range(48,72)), 'seller_BC': list(range(72,96))} 
else:
    only_cond = {''};
    S2 = D.sid.unique()[18:]
    D2 = D[D['sid'].isin(S2)]

hl = D.columns.values.tolist()
print(D)

r_m = sp.mean(D.ix[:, 'profit'])
ns = D.shape[0]
a = 0.5 # learning rate
b = 1 # inverse temperature
Bbins = 101

if project == "neuroshyuka":
    initbid['BC'] = 6.554348
    initbid['NC'] = 5.131915
    initbid['SC'] = 4.959574
elif project == "econoshyuka":
    initbid['BC'] = D[(D['snb']=='BC') & (D['block_n']==1)]['bid'].mean() 
    initbid['NC'] = D[(D['snb']=='NC') & (D['block_n']==1)]['bid'].mean() 
    initbid['SC'] = D[(D['snb']=='SC') & (D['block_n']==1)]['bid'].mean()
print(initbid)



###  Value initialization  

if Bbins == 101:
    B = sp.linspace(0, 10, 101)
elif Bbins == 11:
    B = sp.linspace(0, 10, 11)

"""
# Naive cognitive hierachy level 0
initQ_CH0 = sp.array([float(buyer_profit(i)) for i in B]) #initQd_CH0 = {round2(i): buyer_profit(i) for i in B} 

thr = 5
scale = 1
initQ_CH0lPr = CH0lPrB(thr) 

# uniform initialization
q0 = 5
initQ_unif = unifB(q0)
"""

# Q (delta rule)
initQ = sp.copy(initQ_unif) # for NumPy! nitQ[:] = initQ_unif 
markets = {'SC', 'NC', 'BC'} #{'BC', 'NC', 'SC'}
Q = dict(BC = sp.copy(initQ), 
         NC = sp.copy(initQ),
         SC = sp.copy(initQ))




### Code examples for fitting and plotting 


## Fitting

# Fixed effects optimization
from optimization import mle_ffx_optim()
mle_ffx_optmin() 

# Mixed effects optimization
from optimization import mle_ffx_optim()
mle_rfx_optmin_batch() 


## Saving results to list
# efp29 contains the fit results of a mixed effects mle analysis
from fits_results_econoshyuka import efp29
l = [sp.mean(list(efp29[i][3].values())) for i in range(24)], [sp.std(list(efp29[i][3].values())) for i in range(24)] 

## Reading saved analyses
# the folder ../summarized_results contains analysis output (fitted parameters, simulations, BIC scores, etc.) 
R, isconv, ll_m, par_m = pickle.load(open('../summarized_results/rfx_econoshyuka_drlepkurnudger7_d.pkl', 'rb')) 
R, isconv, ll_m, par_m = pickle.load(open('../summarized_results/rfx_econoshyuka_dr_avf_b.pkl', 'rb')) 
R, isconv, ll_m, par_m = pickle.load(open('../summarized_results/rfx_neuroshyuka_.pkl', 'rb')) 

# calculating BIC. utility functions are in utils.py
from utils import bic
bic(-13.47, 60, 2/27) 


## Plotting simulations

# DL-type best fitting model simulation
from simulation_plots import *
shyuka.simul_plt.play_dr_nudger(fp[40]) 
# RL-type best fitting model simulation
shyuka.simul_plt.play_avf(fp[15])

















