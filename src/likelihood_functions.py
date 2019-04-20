
from .main import *


# Explicit computation of L, yoked parameters
class llhf_ffx_explicit:  # numerical log-likelihood  function
    def __init__(self, mms_a, mms_b):
        if not mms_a:
            mms_a = 0, 1, 10**-2
        if not mms_b:
            mms_b = 10**-2, 1, 10**-2
        self.mms_a = mms_a
        self.mms_b = mms_b
        self.dom_a = sp.arange(*mms_a)
        self.dom_b = sp.arange(*mms_b)
        self.f = sp.zeros((len(self.dom_a), len(self.dom_b)))
         # self.f = {(a, b): 0 for a, b in zip(self.dom_a, self.dom_b)}
    def update(self, a, b, logp):
        x_a = round((a - self.mms_a[0]) / self.mms_a[2]) 
        x_b = round((b - self.mms_b[0]) / self.mms_b[2])
        self.f[x_a, x_b] += logp

def mle_ffx_yoked_explicit():
    initQ = sp.copy(initQ)
    mms_a = (0, 1.1, 0.25)
    mms_b = (0.1, 2.1, 0.5)
    Ln = llhf_explicit(mms_a, mms_b) 
    for a in Ln.dom_a:
        for b in Ln.dom_b:
            for i in range(len(D)):
                row = D.ix[i, 0]
                m = D.ix[i, 'snb']
                bid = round2(D.ix[i, 'bid'])
                if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
                    Q[m] = sp.copy(initQ)
                r = D.ix[i, 'profit']
                if not sp.isnan(D.ix[i, 'bid']): 
                    if Bbins == 101: 
                        ind = int(bid * 10) 
                    elif Bbins == 11:
                        ind = int(round(bid)) 
                    p, Q[m] = pu_naive_avf_agent(b, Q[m], ind, r)
                    #p, Q[m] = dr_naive_avf_agent(a, b, Q[m], ind, r)
                    Ln.update(a, b, sp.log10(p))
                    print(row, a, b, m, bid, r, Q[m], sp.log10(p))

    Lmax = sp.amax(Ln.f)
    mlep = sp.where(Ln.f == Lmax) #mlep = sp.argmax(Ln.f) 
    mle_a, mle_b = mlep[0] * Ln.mms_a[2] + Ln.mms_a[0], mlep[1] * Ln.mms_b[2] + Ln.mms_b[0]
    
    fig, axa = plt.subplots(2, 1)

    plt.subplot(211) # fig = plt.figure(); axs = fig.add_subplot(); axs.plot()
    levels = sp.linspace(sp.amin(-Ln.f), sp.amax(-Ln.f), 20)
    X, Y = sp.meshgrid(Ln.dom_a, Ln.dom_b)
    plt.contourf(X, Y, -Ln.f.transpose(), levels = levels, alpha = .5, cmap = plt.cm.gray) 
    plt.colorbar(format = '%.2f') 
    isoLn = plt.contour(X, Y, -Ln.f.transpose(), levels = levels, colors='black', linewidth=.5)
    plt.clabel(isoLn, inline=1, fontsize=10)
    plt.annotate('Lmax', xy = (min(mle_a), min(mle_b)), xytext = (0.1, 0.1), arrowprops = dict(facecolor='b', width=0.01, shrink=0.02))

    plt.title('NLL; Lmax=L({0},{1})={2}'.format(mle_a, mle_b, Lmax))
    plt.xlabel('a')
    plt.ylabel('b')
    #plt.xticks(())
    #plt.yticks(())
    plt.legend()
    
    ax3 = fig.add_subplot(212, projection='3d')
    ax3.plot_wireframe(X, Y, -Ln.f.transpose()) #plot_surface

    pdb.set_trace()
    plt.show() # or plt.savefig(); plt.close() 

    return mle_a, mle_b

# Bid nudging
def mle_ffx_yoked_explicit_nudging(initq = 5):
    q = dict(BC = initq,
             NC = initq,
             SC = initq)
    mms_nup = (0, 1.5, 0.5)
    mms_ndn = (0, 1.5, 0.5)
    dom_nup = sp.arange(*mms_nup)
    dom_ndn = sp.arange(*mms_ndn)
    L = sp.zeros((len(dom_nup), len(dom_ndn)))
    for nu in dom_nup:
        for nd in dom_ndn:
            for i in range(len(D)):
                row = D.ix[i, 0]
                m = D.ix[i, 'snb']
                bid = round2(D.ix[i, 'bid'])
                if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
                    q[m] = initq
                r = D.ix[i, 'profit']
                if not sp.isnan(D.ix[i, 'bid']): 
                    p, q[m] = naive_gausnudger_agent(nu, nd, q[m], bid, r)
                    L[nu, nd] += sp.log10(p) 
                    print(row, nu, nd, m, bid, r, q[m], sp.log10(p))
    Lmax = sp.amax(L)
    mlep = sp.where(L == Lmax) #mlep = sp.argmax(Ln.f) 
    mle_nup, mle_ndn = mlep[0] * mms_nup[2] + mms_nup[0], mlep[1] * mms_ndn[2] + mms_ndn[0]

    fig = plt.figure(1)
    plt.subplot(111) 
    X, Y = sp.meshgrid(dom_nup, dom_ndn)
    plt.contourf(X, Y, -L.transpose(), 8, alpha = .8, cmap = plt.cm.hot) 
    isoLn = plt.contour(X, Y, -L.transpose(), 8, colors='black', linewidth=.5)
    plt.clabel(isoLn, inline=1, fontsize=10)
    plt.annotate('Lmax', xy = mlep, xytext = (0.1, 1), arrowprops = dict(facecolor='b', shrink=0.02))

    plt.title('NLL')
    plt.xlabel('nup')
    plt.ylabel('ndn')
    plt.legend()

    ax3 = fig.add_subplot(212, projection='3d')
    ax3.plot_wireframe(X, Y, -L.transpose())
    
    pdb.set_trace()
    plt.show() 


## Local extrema search in L, yoked parameters

# Objective functions: Fixed effects likelihood functions
""" For population-level questions, treating parameters as fixed effects and thereby
conflating within- and between-subject variability can lead to serious problems 
such as overstating the true significance of results"""

def nll_ffx_dr_avf(params, D = D2):
    initQ = {}
    if len(params) == 2:
        fn = 'behvars_dr_avf4.csv'
        a = {m:params[0] for m in markets}
        b = {m:params[1] for m in markets}
        bda = initbid
        bdb, bdc = 2, 4
        initQ = {m: beta_distB(initbid[m], bdb, bdc) for m in markets}
    elif len(params) == 5:
        fn = 'behvars_dr_avf1.csv'
        a = {m:params[0] for m in markets}
        b = {m:params[1] for m in markets}
        bda, bdb, bdc = params[2:5]
        initQ = {m: beta_distB(bda, bdb, bdc) for m in markets}
    elif len(params) == 9 and 1==0:
        fn = 'behvars_dr_avf2.csv'
        a = {i[1]: params[0:6:2][i[0]] for i in enumerate(markets)}
        b = {i[1]: params[1:6:2][i[0]] for i in enumerate(markets)}
        initQ = {i[1]: unifB(params[6:9][i[0]]) for i in enumerate(markets)}
    elif len(params) == 9:
        fn = 'behvars_dr_avf3.csv'
        a = {m:params[0] for m in markets}
        b = {m:params[1] for m in markets}
        bdab = params[2:8]
        use_bdmax = True 
        if not use_bdmax:
            bdc = params[8]
        elif use_bdmax:
            bdmax = params[8]
            #bdc = newtonraphson_secant(5, (bda, bdb)) 
            try: 
                bdc = min([brent(0, 10, (bdab[2*i], bdab[2*i+1], bdmax)) for i in range(3)])
            except ValueError:
                return 10**10 
        initQ = {i[1]: beta_distB(bdab[2*i[0]], bdab[2*i[0]+1], bdc) for i in enumerate(markets)}
    Q = copy.deepcopy(initQ)

    if project == "econoshyuka":
        fn = 'econo_' + fn

    nt = len(D)
    nb = max(D.ix[:, 'block_n'])
    emv, opb, rpe, ecv, clh, cob, coe, cmr, sve = [], [], [], [], [], [], [], [], []    
    writevars = True
    if writevars:
        fid = open(fn, 'w') 
        cw = csv.writer(fid, delimiter = ',')
        cw.writerow(["opb", "clh", "rpe", "cob", "coe", "cmr", "sve", "emv", "ecv"])

    L = 0
    sL = {i:0 for i in D.sid.unique()}
    for i in D.index:
        row = D.ix[i, 0]
        bn = D.ix[i, 'block_n']
        m = D.ix[i, 'snb']
        bid = round2(D.ix[i, 'bid'])
        b2b = round2(D.ix[i, 'b2_bid'])
        s1p = round2(D.ix[i, 's1_rp'])
        s2p = round2(D.ix[i, 's2_rp'])
        r = D.ix[i, 'profit']
        si = D.ix[i, 'sid']
        if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
            Q[m] = sp.copy(initQ[m])
            acr = 0
        emv.append(max(Q[m]))
        if Bbins == 101: 
            opb.append(round2(sp.argmax(Q[m]) / 10.0))
        elif Bbins == 11:
            opb.append(round2(sp.argmax(Q[m])))
        if sp.isnan(D.ix[i, 'bid']):
            ecv.append(sp.nan)
            rpe.append(sp.nan)
            coe.append(sp.nan)
            cob.append(sp.nan)
            cmr.append(sp.nan)
            clh.append(sp.nan)
            sve.append(sp.nan)
        else: 
            if Bbins == 101: 
                ind = int(bid * 10) 
                opb.append(round2(sp.argmax(Q[m]) / 10.0))
            elif Bbins == 11:
                ind = int(round(bid))
                opb.append(round2(sp.argmax(Q[m])))
            if m == 'NC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    cob.append(s1p) 
                else: 
                    if bid >= s1p: 
                        cob.append(bid);   
                    else:
                        cob.append(bid + 1); # this is an approximation
            elif m == 'SC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    cob.append(min(s1p, s2p)) 
                else: 
                    if bid >= min(s1p, s2p):
                        cob.append(bid);   
                    else:
                        cob.append(bid + 1); # this is an approximation
            elif m == 'BC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    cob.append(max(s1p, b2b + 0.1)) 
                else: 
                    cob.append(b2b + 0.1);
            rpei = r - Q[m][ind]
            rpe.append(rpei)
            acr += r
            cmr.append(acr)
            coe.append((10.0 - cob[-1]) - r)
            sve.append((10.0 - cob[-1]) - Q[m][ind])
            ecv.append(Q[m][ind])
            coe.append(r - 10.0 + cob[-1])
            if project == "econoshyuka" and {'d'} == only_cond:
                ind = int(s1p * 10) 
                p, Q[m] = dr_unilatupd_agent(a[m], b[m], Q[m], ind, 10-ind)
            else: 
                p, Q[m] = dr_unilatupd_agent(a[m], b[m], Q[m], ind, r)
            #p, Q[m] = dr_naive_avf_agent(a[m], b[m], Q[m], ind, r)
            clh.append(sp.log10(p))
            L += sp.log10(p)
            sL[si] += sp.log10(p)
            if Bbins == 11:
                ex11L = sp.log10(0.2) if bid >= 9.5 or bid < 0.5 else sp.log10(0.1)
                L += ex11L
                sL[si] += ex11L
                clh[-1] += ex11L

    if writevars:
        csvrows = zip(opb, clh, rpe, cob, coe, cmr, sve, emv, ecv)  
        cw.writerows(csvrows)       
        fid.close()

    if len(params) == 2:
        print(a[m], b[m], L, sL)
    elif len(params) == 5:
        print(a[m], b[m], bda, bdb, bdc, L, sL)
    elif len(params) == 9 and 1==0:
        print(a, b, params[6:9], L, sL)
    elif len(params) == 9:
        print(a[m], b[m], bdab, bdc, "({})".format(params[8]), L, sL.values())

    return 10**10 if sp.isnan(L) else -L

def nll_ffx_dr_naive_avf_jac(params, D = D): # deprecated: too complex 
    initQ = {}
    if len(params) == 5:
        a = {m:params[0] for m in markets}
        b = {m:params[1] for m in markets}
        bda, bdb, bdc = params[2:5]
        initQ = {m: beta_distB(bda, bdb, bdc) for m in markets}
    elif len(params) == 9 and 1==0:
        a = {i[1]: params[0:6:2][i[0]] for i in enumerate(markets)}
        b = {i[1]: params[1:6:2][i[0]] for i in enumerate(markets)}
        initQ = {i[1]: unifB(params[6:9][i[0]]) for i in enumerate(markets)}
    elif len(params) == 9:
        a = {m:params[0] for m in markets}
        b = {m:params[1] for m in markets}
        bdab = params[2:8]
        bdc = params[8]
        initQ = {i[1]: beta_distB(bdab[2*i[0]], bdab[2*i[0]+1], bdc) for i in enumerate(markets)}
    Q = copy.deepcopy(initQ)
    dQ_dx = dict((m, unifB(0)) for m in markets)
    dQ_dc1 = dict((m, beta_distB_da(bda, bdb, bdc))for m in markets)
    dQ_dc2 = dict((m, beta_distB_db(bda, bdb, bdc)) for m in markets)
    dQ_dc3 = dict((m, beta_distB(bda, bdb, 1)) for m in markets)

    jac_a, jac_b, jac_eb1, jac_eb2, jac_eb3 = 5 * (0,)
    for i in range(len(D)):
        m = D.ix[i, 'snb']
        bid = round2(D.ix[i, 'bid'])
        r = D.ix[i, 'profit']
        if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
            Q[m] = sp.copy(initQ[m])
            dQ_da[m] = unifB(0);
        if not sp.isnan(D.ix[i, 'bid']): 
            if Bbins == 101: 
                ind = int(bid * 10) 
            elif Bbins == 11:
                ind = int(round(bid))
            Qc = sp.copy(Q[m]) 
            dQc_da   = sp.copy(dQ_dx) 
            dQc_deb1 = sp.copy(dQ_dc1)   
            dQc_deb2 = sp.copy(dQ_dc2) 
            dQc_deb3 = sp.copy(dQ_dc3) 

            _, Q[m] = dr_naive_avf_agent(a[m], b[m], Q[m], ind, r)
            dQ_da[m][ind] = dQ_da[m][ind] * (1 - Qc[ind]) + r 
            dQ_deb1[m][ind] *= (1 - a[m])  
            dQ_deb2[m][ind] *= (1 - a[m])  
            dQ_deb3[m][ind] *= (1 - a[m])  
 
            jac_a   += b[m]*dQc_da[ind]*(1+r-Qc[ind]) - b[m] * sum([boltzmann_dist(b[m], Qc, j)*dQc_da[j] for j in B])
            jac_b   += Q[m][ind] - sum([Q[m][j] * p for j in B]) 
            jac_eb1 += b[m]*dQc_deb1[ind]*(1-a[m]) - b[m] * sum([boltzmann_dist(b[m], Qc, j)*dQc_deb1[j] for j in B])  
            jac_eb2 += b[m]*dQc_deb2[ind]*(1-a[m]) - b[m] * sum([boltzmann_dist(b[m], Qc, j)*dQc_deb2[j] for j in B])   
            jac_eb3 += b[m]*dQc_deb3[ind]*(1-a[m]) - b[m] * sum([boltzmann_dist(b[m], Qc, j)*dQc_deb3[j] for j in B])  
    if len(params) == 5:
        print(a[m], b[m], bda, bdb, bdc, (-jac_a, -jac_b, -jac_eb1, -jac_eb2, -jac_eb3))
    elif len(params) == 9 and 1==0:
        print(a, b, params[6:9], (-jac_a, -jac_b, -jac_eb1, -jac_eb2, -jac_eb3))
    elif len(params) == 9:
        print(a[m], b[m], bdab, bdc, (-jac_a, -jac_b, -jac_eb1, -jac_eb2, -jac_eb3))
    
    if any(sp.isnan(i) for i in (-jac_a, -jac_b, -jac_eb1, -jac_eb2, -jac_eb3)):
        return 10**10 #sp.inf
    else:
        return -jac_a, -jac_b, -jac_eb1, -jac_eb2, -jac_eb3

def nll_ffx_pu_avf(params, D = D):
    initQ = {}
    if len(params) == 4:
        b = {m:params[0] for m in markets} 
        bda, bdb, bdc = params[1:4]
        initQ = {m: beta_distB(bda, bdb, bdc) for m in markets}
    elif len(params) == 7:
        b = {i[1]: params[0:3][i[0]] for i in enumerate(markets)}
        initQ = {i[1]: unifB(params[4:7][i[0]]) for i in enumerate(markets)}
    elif len(params) == 8:
        b = {m:params[0] for m in markets} 
        bdab = params[1:7]
        use_bdmax = True 
        if not use_bdmax:
            bdc = params[7]
        elif use_bdmax:
            bdmax = params[7]
            #bdc = newtonraphson_secant(5, (bda, bdb)) 
            try:
                bdc = min([brent(0, 10, (bdab[2*i], bdab[2*i+1], bdmax)) for i in range(3)])
            except ValueError:
                return 10**10 
        initQ = {i[1]: beta_distB(bdab[2*i[0]], bdab[2*i[0]+1], bdc) for i in enumerate(markets)}
    Q = copy.deepcopy(initQ) 

    L = 0
    for i in D.index:
        m = D.ix[i, 'snb']
        bid = round2(D.ix[i, 'bid'])
        r = D.ix[i, 'profit']
        if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
            Q[m] = sp.copy(initQ[m])
        if not sp.isnan(D.ix[i, 'bid']): 
            if Bbins == 101: 
                ind = int(bid * 10) 
            elif Bbins == 11:
                ind = int(round(bid))
            #p, Q[m] = pu_naive_avf_agent(b[m], Q[m], ind, r)
            p, Q[m] = pu_unilatupd_agent(b[m], Q[m], ind, r)
            L += sp.log10(p)
            if Bbins == 11:
                L += sp.log10(0.2) if bid >= 9.5 or bid < 0.5 else sp.log10(0.1) 
    if len(params) == 4: 
        print(b[m], bda, bdb, bdc, L)
    if len(params) == 7: 
        print(b, params[4:7], L)
    if len(params) == 8: 
        print(b[m], bdab, bdc, "({})".format(params[7]), L)

    return 10**10 if sp.isnan(L) else -L


def nll_null():
    return -sp.log10(null_agent()) * m

counter = 0
def nll_ffx_dr_nudger(params, D = D2):  # pana.nll_ffx_dr_nudger(pana.fp[38][2])
    if len(params) == 2:
        fn = 'behvars_drnudger1.csv'
        a, sig = params
        mu0 = copy.deepcopy(initbid)
        q = dict((m, mu0[m]) for m in markets)
    elif len(params) == 3:
        #nu, nd, sig = params
        #initq = copy.deepcopy(initbid)
        #q = dict((m, initq[m]) for m in markets)
        #fn = 'behvars_nvnudger1.csv'
        fn = 'behvars_drnudger2.csv'
        a, siga, sigr = params
        mu0 = copy.deepcopy(initbid)
        q = copy.deepcopy(mu0)
    elif len(params) == 4:
        nu, nd, siga, sigr = params
        initq = copy.deepcopy(initbid)
        q = dict((m, initq[m]) for m in markets)
        fn = 'behvars_nvnudger2.csv'
        mu0 = copy.deepcopy(initbid)
        q = copy.deepcopy(mu0)
        #fn = 'behvars_drnudger3.csv'
        #p1, p2, siga, sigr = params
        #mu0 = copy.deepcopy(initbid)
        #q = copy.deepcopy(mu0)
    elif len(params) == 5 and 1==0:
        fn = 'behvars_drnudger4.csv'
        a, nd, nu, siga, sigr = params
        mu0 = copy.deepcopy(initbid)
        q = copy.deepcopy(mu0)
    elif len(params) == 7:
        fn = 'behvars_drnudger5.csv'
        a, nd, nu, siga, sigr, sign, raitc = params
        mu0 = copy.deepcopy(initbid)
        q = copy.deepcopy(mu0)
    elif len(params) == 5:
        if project == 'econoshyuka':
            fn = 'behvars_drlepkurnudger7_'+list(only_cond)[0]+'.csv'
        else:
            fn = 'behvars_drlepkurnudger7.csv'
        a, siga, sigr, sign, raitc = params
        mu0 = copy.deepcopy(initbid)
        q = copy.deepcopy(mu0)
    elif len(params) == 5 and 1==0:
        a = params[0] 
        sig = params[1] 
        mu0 = {i[1]: params[2:5][i[0]] for i in enumerate(markets)}
        q = dict((m, mu0[m]) for m in markets)

    if project == "econoshyuka":
        fn = 'econo_' + fn

    opb, rpe, clh, cob, coe, cmr, sve = [], [], [], [], [], [], []
    writevars = True
    if writevars:

        cw.writerow(["opb", "clh", "rpe", "cob", "coe", "cmr", "sve"])

    L = 0
    sL = {i:0 for i in D.sid.unique()}
    for i in D.index:
        m = D.ix[i, 'snb']
        bid = round2(D.ix[i, 'bid'])
        b2b = round2(D.ix[i, 'b2_bid'])
        s1p = round2(D.ix[i, 's1_rp'])
        s2p = round2(D.ix[i, 's2_rp'])
        r = D.ix[i, 'profit']
        si = D.ix[i, 'sid']
        if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
            q[m] = mu0[m]
            acr = 0
            prevacc = {m:1 for m in markets}
            prevcob = {m:mu0[m] for m in markets}
        
        opb.append(q[m])
        if sp.isnan(D.ix[i, 'bid']): 
            cob.append(sp.nan)
            rpe.append(sp.nan)
            cmr.append(sp.nan)
            coe.append(sp.nan)
            clh.append(sp.nan)
            sve.append(sp.nan)
        else:  
            if m == 'NC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    cob.append(s1p) 
                else: 
                    if bid >= s1p: 
                        cob.append(bid);   
                    else:
                        cob.append(bid + 1); # this is an approximation
            elif m == 'SC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    cob.append(min(s1p, s2p)) 
                else: 
                    if bid >= min(s1p, s2p):
                        cob.append(bid);   
                    else:
                        cob.append(bid + 1); # this is an approximation
            elif m == 'BC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    cob.append(max(s1p, b2b + 0.1)) 
                else: 
                    cob.append(b2b + 0.1);
            rpei = r - 10.0 + q[m]
            rpe.append(rpei)
            acr += r
            cmr.append(acr)
            coe.append((10.0 - cob[-1]) - r)
            sve.append((10.0 - cob[-1]) - 10 + q[m])
            #p, q[m] = naive_gausnudger1_agent(nu, nd, sig, q[m], bid, r) # gaus/lepkur nudger
            #p, q[m] = naive_lepkurnudger2_agent(nu, nd, siga, sigr, q[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0  # gaus/lepkur nudger
            #p, q[m] = dr_gausnudger1_agent(a, sig, q[m], bid, r) # gaus
            #p, q[m] = dr_gausnudger2_agent(a, siga, sigr, q[m], bid, r, prevacc); prevacc[m] = 1 if r > 0 else 0  # gaus
            #p, q[m] = dr_lepkurnudger1_agent(a, sig, q[m], bid, r) # lepkur
            #p, q[m] = dr_lepkurnudger2_agent(a, siga, sigr, q[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 # lepkur
            #nd, nu = p1, p2; p, q[m] = dr_lepkurnudger31_agent(nd, nu, siga, sigr, q[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 
            #a, nu = p1, p2; p, q[m] = dr_lepkurnudger32_agent(a, nu, siga, sigr, q[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 
            #aa, ar = p1, p2; p, q[m] = dr_lepkurnudger33_agent(aa, ar, siga, sigr, q[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 
            #p, q[m] = dr_lepkurnudger4_agent(a, nd, nu, siga, sigr, q[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 
            #p, q[m] = dr_lepkurnudger5_agent(a, nd, nu, siga, sigr, sign, raitc, q[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 
            #p, q[m] = dr_lepkurnudger6_agent(a, siga, sigr, sign, raitc, q[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 
            p, q[m] = dr_lepkurnudger7_agent(a, siga, sigr, sign, raitc, q[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 
            #p, q[m] = dr_fb1_lepkurnudger6_agent(a, siga, sigr, sign, raitc, q[m], bid, r, prevcob[m], cob[-1]); prevcob[m] = cob[-1];  
            #p, q[m] = dr_fb2_lepkurnudger6_agent(a, siga, sigr, sign, raitc, q[m], bid, r, prevcob[m], cob[-1]); prevcob[m] = cob[-1]; prevacc[m] = 1 if r>0 else 0  
            clh.append(sp.log(p))
            L += sp.log10(p) 
            sL[si] += sp.log10(p)
 
    if writevars:
        csvrows = zip(opb, clh, rpe, cob, coe, cmr, sve)  
        cw.writerows(csvrows)       
        fid.close()
    
    global counter 
    counter += 1
    if project == 'neuroshyuka' or project == 'econoshyuka':# and counter % 100 == 0:
        if len(params) == 2:
            print(a, sig, L, sL)
        elif len(params) == 3:
            #print(a, sig, mu0, L)
            print(a, siga, sigr, L, sL)
        elif len(params) == 4:
            #print(p1, p2, siga, sigr, L, sL)
            print(nu, nd, siga, sigr, L, sL)
        elif len(params) == 5:
            print(a, siga, sigr, sign, raitc, L, sL)
        elif len(params) == 5:
            print(a, nd, nu, siga, sigr, L, sL)
        elif len(params) == 5:
            print(a, sig, mu0, L, sL)
        if len(params) == 7: 
            print(a, nd, nu, siga, sigr, sign, raitc, L, sL)
    
    return 10**10 if sp.isnan(L) or sp.isinf(L) else -L


# Kernel density estimator

def nll_ffx_kde(params, D = D):  
    if project == 'econoshyuka':
        fn = 'behvars_kde_'+list(only_cond)[0]+'.csv'
    else:
        fn = 'behvars_kde.csv'

    if len(params) == 2:
      beta, bw = params
    elif len(params) == 1:
      beta = params[0]
      bw = 'scott' # 'silverman'
  
    initbidl = {k:[v] for k, v  in initbid.items()}
    s1est = copy.deepcopy(initbidl); s2est = copy.deepcopy(initbidl); b2est = copy.deepcopy(initbidl)

    if project == "econoshyuka":
        fn = 'econo_' + fn

    opb, rpe, clh, cob, coe, cmr, sve = [], [], [], [], [], [], []
    writevars = True
    if writevars:
        fid = open(fn, 'w') 
        cw = csv.writer(fid, delimiter = ',')
        cw.writerow(["opb", "clh", "rpe", "cob", "coe", "cmr", "sve"])

    L = 0
    sL = {i:0 for i in D.sid.unique()}
    for i in D.index:
        m = D.ix[i, 'snb']
        bid = round2(D.ix[i, 'bid'])
        b2b = round2(D.ix[i, 'b2_bid'])
        s1p = round2(D.ix[i, 's1_rp'])
        s2p = round2(D.ix[i, 's2_rp'])
        r = D.ix[i, 'profit']
        si = D.ix[i, 'sid']
        if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
            opbi = initbid[m]
            acr = 0
            s1est = copy.deepcopy(initbidl); s2est = copy.deepcopy(initbidl); b2est = copy.deepcopy(initbidl)
        if sp.isnan(D.ix[i, 'bid']): 
            cob.append(sp.nan); rpe.append(sp.nan); cmr.append(sp.nan); coe.append(sp.nan); clh.append(sp.nan); sve.append(sp.nan)
        else:  
            if m == 'NC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    cob.append(s1p) 
                else: 
                    if bid >= s1p: 
                        cob.append(bid);   
                    else:
                        cob.append(bid + 1); # this is an approximation
                s2orb2est = None
            elif m == 'SC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    cob.append(min(s1p, s2p)) 
                else: 
                    if bid >= min(s1p, s2p):
                        cob.append(bid);   
                    else:
                        cob.append(bid + 1); # this is an approximation
                s2orb2est = s2est[m]
            elif m == 'BC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    cob.append(max(s1p, b2b + 0.1)) 
                else: 
                    cob.append(b2b + 0.1);
                s2orb2est = b2est[m]
            rpei = r - 10.0 + opbi
            rpe.append(rpei)
            acr += r
            cmr.append(acr)
            coe.append((10.0 - cob[-1]) - r)
            sve.append((10.0 - cob[-1]) - 10 + opb)

            
            p, opbi, s1est[m], s2orb2est, U = kde_agent(beta, bid, i, m, bw, s1est[m], s2orb2est); 
            if m == 'SC':
                s2est[m] = s2orb2est  
            elif m == 'BC':
                b2est[m] = s2orb2est  

            opb.append(opbi)
            clh.append(sp.log(p))
            L += sp.log10(p) 
            sL[si] += sp.log10(p)
        print(only_cond, beta, bw, i, p, L, sL)
 

        #plt.plot(d.pdf(sp.linspace(0, 10, Bbins)))
        #plt.show()
             
    if writevars:
        csvrows = zip(opb, clh, rpe, cob, coe, cmr, sve)  
        cw.writerows(csvrows)       
        fid.close()
    
    global counter 
    counter += 1
    if project == 'neuroshyuka' or project == 'econoshyuka' and counter % 100 == 0:
        print(only_cond, beta, bw, L, sL)
    
    return 10**10 if sp.isnan(L) or sp.isinf(L) else -L



# Objective function: Sum of squared residuals function / Least squares
def ls_ffx_dr_max(params):
    def dr_max_agent(a, Q, bi, r):
        maxvb = Q.index(max(Q)) 
        Q[bi] += a * (r - Q[bi])
        return maxvb, Q
    def nud_max_agent(a, Q, bi, r):
        maxvb = Q.index(max(Q)) 
        Q[bi] += a * (r - Q[bi])
        return maxvb, Q
    S = 0
    Q = sp.copy(initQ)
    a, b = params[0], params[1]
    for i in range(len(D)):
        r = D.ix[i, 'profit']
        bid = D.ix[i, 'bid']
        if not sp.isnan(bid): 
            if Bbins == 101: 
                ind = int(bid * 10) 
            elif Bbins == 11:
                ind = int(round(bid))
            maxvb, Q = dr_max_agent(a, Q, ind, r)
            s = (bid - maxvb)**2
            S += s 
    return sp.inf if sp.isnan(L) else -L


## Bayesian estimation of density

def emnll():
    """http://nbviewer.ipython.org/github/tritemio/notebooks/blob/master/Mixture_Model_Fitting.ipynb"""
    """ not applicable """

def nll_ffx_kalman_nudger(params):
    if len(params) == 2:
        s0 = {m: params[0] for m in markets}
        mu0 = copy.deepcopy(initbid)
        sb0 = {m: params[1] for m in markets}
        q = dict((m, [mu0[m], sb0[m]]) for m in markets)
    if len(params) == 3:
        s0, mu0, sb0 = params
        q = dict((m, [mu0, sb0]) for m in markets)
        #s0 = {m: params[0] for m in markets}
        #mu0 = copy.deepcopy(initbid)
        #sb0a = {m: params[1] for m in markets}
        #sb0r = {m: params[2] for m in markets}
        #q = dict((m, [mu0[m], sb0a[m], sb0r[m]]) for m in markets)
    elif len(params) == 5:
        s0 = {m: params[0] for m in markets}
        mu0 = {i[1]: params[1:4][i[0]] for i in enumerate(markets)}
        sb0 = {m: params[4] for m in markets}
        q = dict((m, [mu0[m], sb0[m]]) for m in markets)

    L = 0
    for i in range(len(D)):
        m = D.ix[i, 'snb']
        bid = round2(D.ix[i, 'bid'])
        r = D.ix[i, 'profit']
        if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
            q[m] = [mu0, sb0] 
            #q[m] = [mu0[m], sb0a[m], sb0r[m]]; prevacc = {m:1 for m in markets} # mu0, sb0a, sb0r, are updated independenlty for each market
        if not sp.isnan(D.ix[i, 'bid']):  
            p, q[m][0], q[m][1] = kalman_lepkurnudger1_agent(q[m][0], q[m][1], s0, bid, r) # lepkur/gaus
            #p, q[m][0], q[m][1], q[m][2] = kalman_lepkurnudger2_agent(q[m][0], q[m][1], q[m][2], s0[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 # lepkur/gaus
            L += sp.log10(p) 

    if len(params) == 2:
        print(s0[m], sb0[m], L)
    if len(params) == 3:
        print(s0, mu0, sb0, L)
        #print s0[m], sb0a[m], sb0r[m], L)
    if len(params) == 5:
        print(s0[m], mu0, sb0[m], L)

    return 10**10 if sp.isnan(L) else -L

# Bayesian estimator of static srpd
def bayesian_estimator_ffx(i):
    """https://en.wikipedia.org/wiki/Beta_distribution"""
    initQ = beta_distB(2, 3, 4) 
    a, b = 0.1, 2
    pars0 = sp.concatenate(sp.array([a, b]), initQ, axis = 0)
   
    mms_a = (0, 1.1, 0.25)
    mms_b = (0.1, 2.1, 0.5)
    Ln = llhf_explicit(mms_a, mms_b) 
    for a in Ln.dom_a:
        for b in Ln.dom_b:
            for i in range(len(D)):
                row = D.ix[i, 0]
                m = D.ix[i, 'snb']
                bid = round2(D.ix[i, 'bid'])
                if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
                    Q[m] = sp.copy(initQ)
                r = D.ix[i, 'profit']
                if not sp.isnan(D.ix[i, 'bid']): 
                    if Bbins == 101: 
                        ind = int(bid * 10) 
                    elif Bbins == 11:
                        ind = int(round(bid)) 
                    p, Q[m] = pu_naive_avf_agent(b, Q[m], ind, r)
                    #p, Q[m] = dr_naive_avf_agent(a, b, Q[m], ind, r)
                    Ln.update(a, b, sp.log10(p))
                    print(row, a, b, m, bid, r, Q[m], sp.log10(p))
