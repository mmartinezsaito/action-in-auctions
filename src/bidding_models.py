
from .utils import *
from .main import *



###############################
###  Some benchmark models  ###
###############################

# Null model
def null_agent():
    return 1.0 / len(B)
   
# Game theoretic Nash equilibrium double auction solution  
"""Policy: choose always 9.9 in BC, choose anything in the other conditions"""
def gt_NE_agent(m):
    N = len(B)
    S = 10.0 / N 
    if m == 'BC':
        p = 1
        q = N - S     
    else:
        p = 1.0 / N 
        q = round2(sp.random.randint(N) * S)
    return p, q

# Game theoretic Bayesian Nash equilibrum double auction solution  
""" Chatterjee K, Samuelson W (1983) Bargaining under incomplete information """
""" Felli L (2002) Bilateral asymmetric information """
"""  assumes that both the buyer and the seller believe that the valuation of 
     the opponent (vs, vb respectively) is uniformly distributed on [0, len(B)]
     this is a ChInf model"""
def gt_BayesianNE_agent(m, vb = 1, vs = 1):
    N = len(B)
    S = 10.0 / len(B)
    if   m == 'BC':
        p = 1
        qb = N - S     
    elif m == 'NC' or m == 'BC':
        p = 1
        qb = (2.0/3 * vs + 1.0/12) * N
        qs = (2.0/3 * vb + 1.0/4) * N
    return p, qb    



###########################################
###  Value-based adaptive learning models # 
###########################################
# assumes buyer tries to maximize profit while treating sellers as non-agents, but 'nature'
# estimates seller behavior, including maybe reserve price density, as part of nature


## Learning model

# Model-free RL: value-based delta rule 
def dr_naive_avf_agent(a, b, Q, bi, r):
    p = boltzmann_dist(b, Q, bi) 

    Q[bi] += a * (r - Q[bi])
    return p, Q

def ql_naive_agent(a, b, Q, bi, r):
    gamma = 0 # if zero, same as dr_naive, since it's a stage game: single-shot 
    p = boltzmann_dist(b, Q, bi) 
    Q[bi] += a * (r + gamma * max(Q) - Q[bi])
    return p, Q


# Model-free RL: policy-based updating
def pu_naive_avf_agent(b, A, bi, r):
    p = boltzmann_dist(b, A, bi) 

    A[bi] += r - r_m 
    return p, A

# Model-based RL: rule understanding: acceptance if, seller reserve price > bid
# Simple delta rule batch asymmetrical updating
def dr_unilatupd_agent(a, b, Q, bi, r):
    p = boltzmann_dist(b, Q, bi) 

    if r == 0: # rejected
        for j in range(len(Q)):
            if j <= bi: 
                Q[j] += a * (0 - Q[j]) # r=0 always
            else:
                pass
    else:  # accepted
        for j in range(len(Q)):
            if j >= bi: 
                Q[j] += a * (initQ_CH0[j] - Q[j]) 
            else:
                pass
    return p, Q

def pu_unilatupd_agent(b, Q, bi, r):
    p = boltzmann_dist(b, Q, bi) 
    du = [r - r_m for q in Q]
    du_acc = [initQ_CH0[i] - r_m for i in range(len(Q))]
    if r == 0: # rejected
        for j in range(len(Q)):
            if j <= bi: 
                Q[j] += du[j]
            else:
                pass
    else:  # accepted
        for j in range(len(Q)):
            if j >= bi: 
                Q[j] += du_acc[j]
            else:
                pass
    return p, Q


### Kernel density estimation of opponent's choice pdfs ###

def kdeag_uf(Ks1 = None, Ks2 = None, Kb2 = None):
    """ computes bargainer utility function given density estimates """
    U = sp.zeros(Bbins) 
    for i in range(Bbins):
        for s1 in range(Bbins):
            if Ks2 is None and Kb2 is None:
                if B[i] < B[s1]: 
                    U[i] += 0
                else:
                    U[i] += (10 - B[i]) * Ks1[s1]
            elif Ks2 is not None and Kb2 is None:
                for s2 in range(Bbins):
                    t = min(B[s1], B[s2])
                    if B[i] < t: 
                        U[i] += 0
                    else:
                        U[i] += (10 - B[i]) * Ks1[s1] * Ks2[s2]
            elif Ks2 is None and Kb2 is not None:
                for b2 in range(Bbins):
                    t = max(B[s1], B[b2])
                    if B[i] < t: 
                        U[i] += 0
                    else:
                        U[i] += (10 - B[i]) * Ks1[s1] * Kb2[b2]
    return U


#skde = skl.neighbors.KernelDensity(bandwidth = bw, algorithm = 'auto', kernel = 'gaussian', metric = 'euclidean')
#skf = skde.fit(D.ix[i0:i, 's1_rp'].reshape(-1,1))
#plt.plot(skf.score_sample(sp.linspace(0,10,Bbins).reshape(-1,1))))
def kde_agent(beta, bid, i, m, bw = None, *args):
    """ simulates kde artificial bargainer behavior """
    if bw is None: bw = 'scott'

    i0 = D.index[0]
    if only_cond == {'d'}:
        s1 = D.ix[i0:i, 's1_rp']
        if len(s1) == 1:
            s1[i0+1] = sp.mean([s1, 10])
    else:
        s1est = args[0]
        ar = D.ix[i, 'out_bool']
        b1 = D.ix[i, 'bid']
        b1i = round2(b1)
        if len(s1est) == 1:
            s1est.append(sp.mean([s1est[0], 10]))  
        else:
            gks = stats.gaussian_kde(s1est, bw_method = bw)
            if ar:
               aestint = gks.pdf(sp.arange(0, b1i, 0.1))
               aest = sp.dot(aestint, sp.arange(0, b1i, 0.1)) / b1i 
            else:
               aestint = gks.pdf(sp.arange(b1i, 10, 0.1))
               aest = sp.dot(aestint, sp.arange(b1i, 10, 0.1)) / (10-b1i) 
            s1est.append(aest) 
        s1 = s1est
    gks = stats.gaussian_kde(s1, bw_method = bw)
    Ks1 = gks.pdf(B) 
    Ks1 /= sum(Ks1)

    if m == 'SC':
        if only_cond == {'d'}:
            s2 = D.ix[i0:i, 's2_rp'].dropna()
            if len(s2) == 1:
                s2[i+1] = sp.mean([s2[i], 10])  
        else:
            s2est = args[1]
            if len(s2est) == 1:
                s2est.append(sp.mean([s2est[0], 10]))  
            else:
                gks = stats.gaussian_kde(s2est, bw_method = bw)
                if ar:
                    aestint = gks.pdf(sp.arange(0, b1i, 0.1))
                    aest = sp.dot(aestint, sp.arange(0, b1i, 0.1)) / b1i 
                else:
                    aestint = gks.pdf(sp.arange(b1i, 10, 0.1))
                    aest = sp.dot(aestint, sp.arange(b1i, 10, 0.1)) / (10-b1i) 
                s2est.append(aest) 
            s2 = s2est
        gks = stats.gaussian_kde(s2, bw_method = bw)
        Ks2 = gks.pdf(B) 
        Ks2 /= sum(Ks2)
        U = kdeag_uf(Ks1, Ks2, None)
        s2orb2 = s2
    elif m == 'NC':
        U = kdeag_uf(Ks1, None, None)
        s2orb2 = None
    elif m == 'BC':
        if only_cond == {'d'}:
            b2 = D.ix[i0:i, 'b2_bid'].dropna()
            if len(b2) == 1:
                b2[i+1] = sp.mean([b2[i], 10])  
        else:
            b2est = args[1]
            if len(b2est) == 1:
                b2est.append(sp.mean([b2est[0], 10]))  
            else:
                gks = stats.gaussian_kde(b2est, bw_method = bw)
                if ar:
                    aestint = gks.pdf(sp.arange(0, b1i, 0.1))
                    aest = sp.dot(aestint, sp.arange(0, b1i, 0.1)) / b1i 
                else:
                    aestint = gks.pdf(sp.arange(b1i, 10, 0.1))
                    aest = sp.dot(aestint, sp.arange(b1i, 10, 0.1)) / (10-b1i) 
                b2est.append(aest) 
            b2 = b2est
        gks = stats.gaussian_kde(b2, bw_method = bw)
        Kb2 = gks.pdf(B) 
        Kb2 /= sum(Kb2)
        U = kdeag_uf(Ks1, None, Kb2)
        s2orb2 = b2
    
    opb = B[sp.argmax(U)]
    p = boltzmann_dist(beta, U, int(bid*((Bbins-1)/10)))#p = P[int(bid*((Bbins-1)/10))]
    return p, opb, s1, s2orb2, U



############################################################################
### DL-type nudgers: bumping the bid up and down in a markovian fashion  ###
############################################################################

# Logistic probability of rejection updating
def nudger_CH0lPr_agent(nud, b, Q, bi, thr):
    p = boltzmann_dist(b, Q, bi)
    if  r == 0:
        thr += nud
    else:
        thr -= nud
    Q = CH0lPrB(thr) 
    return p, Q

# Naive gaussian nudging model
def naive_gausnudger1_agent(n_up, n_dn, sig, q, bid, r):
    #sig = (n_up + n_dn) / 2
    p = sp.stats.norm.pdf(bid, loc = q, scale = sig)
    if  r == 0:
        q += n_up
    else:
        q -= n_dn
    return p, q
def naive_gausnudger2_agent(n_up, n_dn, siga, sigr, q, bid, r, prevacc):
    if prevacc:
        p = sp.stats.norm.pdf(bid, loc = q, scale = siga)
    else:
        p = sp.stats.norm.pdf(bid, loc = q, scale = sigr)
    if  r == 0:
        q += n_up
    else:
        q -= n_dn
    return p, q
# Improve using a leptokurtic distribution
# Naive laplacian(leptokurtic) nudging model
def naive_lepkurnudger1_agent(n_up, n_dn, sig, q, bid, r):
    p = sp.stats.laplace.pdf(bid, loc = q, scale = sig)
    if  r == 0:
        q += n_up
    else:
        q -= n_dn
    return p, q
def naive_lepkurnudger2_agent(n_up, n_dn, siga, sigr, q, bid, r, prevacc):
    if prevacc:
        p = sp.stats.laplace.pdf(bid, loc = q, scale = siga)
    else:
        p = sp.stats.laplace.pdf(bid, loc = q, scale = sigr)
    if  r == 0:
        q += n_up
    else:
        q -= n_dn
    return p, q



###  Linear Kalman filter  ###

#  mean-tracking rule for the optimal bid: this is not an action-value function 
#  needs a prior for mu, sb, s0 
def kalman_gausnudger1_agent(mu, sb, s0, bid, r):
    p = sp.stats.norm.pdf(bid, loc = mu, scale = sb)
    if r == 0:
        d = 0 # + sb**2 / (sb**2 + s0**2) 
    else:
        d = bid - mu    # innovation or measurement residual
    a = sb**2 / (sb**2 + s0**2) # optimal Kalman gain (with innovation covariance)
    mu += a * d # updated state estimate
    sb *= sp.sqrt(1 - a) # updated estimate covariance
    return p, mu, sb
def kalman_gausnudger2_agent(mu, sba, sbr, s0, bid, r, prevacc):
    if prevacc:
        p = sp.stats.norm.pdf(bid, loc = mu, scale = sba)
    else:
        p = sp.stats.norm.pdf(bid, loc = mu, scale = sbr)
    if r == 0:
        d = 0 # + sb**2 / (sb**2 + s0**2) 
        a = sbr**2 / (sbr**2 + s0**2) # optimal Kalman gain (with innovation covariance)
        sbr *= sp.sqrt(1 - a) # updated estimate covariance
    else:
        d = bid - mu    # innovation or measurement residual
        a = sba**2 / (sba**2 + s0**2) # optimal Kalman gain (with innovation covariance)
        sba *= sp.sqrt(1 - a) # updated estimate covariance
    a = (sba**2 + sbr**2) / (sba**2 + sbr**2 + 2*s0**2) # optimal Kalman gain (with innovation covariance)
    mu += a * d # updated state estimate
    return p, mu, sba, sbr
# Leptokurtic kalman nudgers
def kalman_lepkurnudger1_agent(mu, sb, s0, bid, r):
    p = sp.stats.laplace.pdf(bid, loc = mu, scale = sb)
    if r == 0:
        d = 0 # + sb**2 / (sb**2 + s0**2) 
    else:
        d = bid - mu    # innovation or measurement residual
    a = sb**2 / (sb**2 + s0**2) # optimal Kalman gain (with innovation covariance)
    mu += a * d # updated state estimate
    sb *= sp.sqrt(1 - a) # updated estimate covariance
    return p, mu, sb
def kalman_lepkurnudger2_agent(mu, sba, sbr, s0, bid, r, prevacc):
    if prevacc:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sba)
    else:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sbr)
    if r == 0:
        d = 0 # + sb**2 / (sb**2 + s0**2) 
        a = sbr**2 / (sbr**2 + s0**2) # optimal Kalman gain (with innovation covariance)
        sbr *= sp.sqrt(1 - a) # updated estimate covariance
    else:
        d = bid - mu    # innovation or measurement residual
        a = sba**2 / (sba**2 + s0**2) # optimal Kalman gain (with innovation covariance)
        sba *= sp.sqrt(1 - a) # updated estimate covariance
    a = (sba**2 + sbr**2) / (sba**2 + sbr**2 + 2*s0**2) # optimal Kalman gain (with innovation covariance)
    mu += a * d # updated state estimate
    return p, mu, sba, sbr


### Delta rule nudger ###

def dr_gausnudger1_agent(a, sig, mu, bid, r): # LOUSY AGENT, BE ASHAMED
    p = sp.stats.norm.pdf(bid, loc = mu, scale = sig)
    if r == 0: 
        d = 0 # + a  
    else:
        d = bid - mu  # innovation or measurement residual   
    mu += a * d # updated state estimate
    return p, mu
def dr_gausnudger2_agent(a, siga, sigr, mu, bid, r, prevacc):
    if prevacc:
        p = sp.stats.norm.pdf(bid, loc = mu, scale = siga)
    else:
        p = sp.stats.norm.pdf(bid, loc = mu, scale = sigr)
    if r == 0:
        d = 0 ## what i lost, or my counterfactual gain
    else:
        d = bid-mu   #   
    mu += a * d # updated state estimate
    #mu = min(10, max(0, mu))
    return p, mu
# Leptokurtic dr nudgers
def dr_lepkurnudger1_agent(a, sig, mu, bid, r): 
    p = sp.stats.laplace.pdf(bid, loc = mu, scale = sig)
    if r == 0: 
        d = 0   
    else:
        d = bid - mu  # innovation or measurement residual   
    mu += a * d # updated state estimate
    return p, mu
def dr_lepkurnudger2_agent(a, siga, sigr, mu, bid, r, prevacc):
    if prevacc:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = siga)
    else:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sigr)
    if r == 0:
        d = 0 ## what i lost, or my counterfactual gain
    else:
        d = bid-mu   #   
    mu += a * d # updated state estimate
    return p, mu
def dr_lepkurnudger31_agent(nd, nu, siga, sigr, mu, bid, r, prevacc):
    if prevacc:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = siga)
    else:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sigr)
    if r == 0:
        d = +nu ## what i lost, or my counterfactual gain
    else:
        d = (bid - mu) -nd  #   
    mu += d # updated state estimate
    return p, mu
def dr_lepkurnudger32_agent(a, nu, siga, sigr, mu, bid, r, prevacc):
    if prevacc:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = siga)
    else:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sigr)
    if r == 0:
        d = +nu ## what i lost, or my counterfactual gain
        mu += d
    else:
        d = (bid - mu)  #   
        mu += a*d  # updated state estimate
    return p, mu
def dr_lepkurnudger33_agent(aa, ar, siga, sigr, mu, bid, r, prevacc):
    if prevacc:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = siga)
    else:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sigr)
    d = (bid - mu)     
    if r == 0:
        mu += ar*d # updated state estimate
    else:
        mu += aa*d # updated state estimate
    return p, mu
def dr_lepkurnudger4_agent(a, nd, nu, siga, sigr, mu, bid, r, prevacc):
    if prevacc:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = siga)
    else:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sigr)
    if r == 0:
        mu += nu # updated state estimate
    else:
        d = (bid - mu)     
        mu += a * d -nd # updated state estimate
    return p, mu
def dr_lepkurnudger5_agent(a, nd, nu, siga, sigr, sign, raitc, mu, bid, r, prevacc): # OVERFITS? 
    # the pdf of a sum of random variables is the convolution of their corresponding pdfs respectively
    #  https://en.wikipedia.org/wiki/List_of_convolutions_of_probability_distributions 
    # i will use a trick with raitc to implement tradeoff sigar and sign
    if   prevacc and bid <= mu: 
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = siga) * 2 * (1 - raitc) 
    elif prevacc and bid > mu:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sign) * 2 * raitc 
    elif not prevacc and bid >= mu:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sigr) * 2 * (1 - raitc)
    elif not prevacc and bid < mu:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sign) * 2 * raitc
    if r == 0:
        mu += nu # updated state estimate
    else:
        d = (bid - mu)     
        mu += a * d -nd # updated state estimate
    return p, mu
def dr_lepkurnudger6_agent(a, siga, sigr, sign, raitc, mu, bid, r, prevacc): # OVERFITS? 
    if   prevacc and bid <= mu: 
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = siga) * 2 * (1 - raitc) 
    elif prevacc and bid > mu:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sign) * 2 * raitc 
    elif not prevacc and bid >= mu:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sigr) * 2 * (1 - raitc)
    elif not prevacc and bid < mu:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sign) * 2 * raitc
    d = (bid - mu)     
    if r > 0:
        mu += a * d # updated state estimate
    return p, mu
def dr_lepkurnudger7_agent(a, siga, sigr, sign, raitc, mu, bid, r, prevacc): 
    if   prevacc and bid <= mu: 
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = siga) * 2 * (1 - raitc) 
    elif prevacc and bid > mu:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sign) * 2 * raitc 
    elif not prevacc and bid >= mu:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sigr) * 2 * (1 - raitc)
    elif not prevacc and bid < mu:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sign) * 2 * raitc
    mu += a * (bid - mu) # updated state estimate
    return p, mu
def dr_fb1_lepkurnudger6_agent(a, sig_lt, sig_e, sig_gt, raitc, mu, bid, r, prevcob, cob): 
    if   bid > prevcob: 
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sig_gt) * 2 * raitc 
    elif bid == prevcob:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sig_e) * 2 * raitc 
    elif bid < prevcob:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sig_lt) * 2 * (1 - raitc)
    mu += a * (cob - mu) # updated state estimate
    return p, mu

