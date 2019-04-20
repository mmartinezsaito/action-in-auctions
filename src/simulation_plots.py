from .main import *
from .utils import sample_bd



## functions for plotting simulation results 
# this serves as a posterior predictive check 


def play_avf(params):
    initQ = {}
    if len(params[2]) == 5:
        a = {m:params[2][0] for m in markets}
        b = {m:params[2][1] for m in markets}
        bda, bdb, bdc = params[2][2:5]
        initQ = {m: beta_distB(bda, bdb, bdc, isab = False) for m in markets}
    elif len(params[2]) == 9: # and params[0]=='11331_ffx_dr_naive_avf_11':
        a = {m:params[2][0] for m in markets}
        b = {m:params[2][1] for m in markets}
        bdab = params[2][2:8]
        bda = params[2][2:8][0:6:2]
        bdb = params[2][2:8][1:6:2]
        use_bdmax = True 
        if not use_bdmax:
            bdc = params[2][8]
        elif use_bdmax:
            bdmax = params[2][8]
            #bdc = newtonraphson_secant(5, (bda, bdb)) 
            bdc = min([brent(0, 10, (bdab[2*i], bdab[2*i+1], bdmax)) for i in range(3)])
        initQ = {i[1]: beta_distB(bdab[2*i[0]], bdab[2*i[0]+1], bdc, isab = False) for i in enumerate(markets)}
    Q = copy.deepcopy(initQ)

    nb = max(D.loc[:, 'block_n'])
    nt = len(D)
    for lhs in ["beh", "opb", "clh", "rpe", "cob", "coe", "cmr", "sve",  "emv","ecv", "sib", "mmsee"]:
        exec(lhs + ' = {m:[[] for i in range(nb)] for m in markets}', locals(), globals())

    fn = 'simvars_' + params[0] + '.csv'
    if project == "econoshyuka":
        fn = 'econo_' + fn
    fid = open(fn, 'w') 
    cw = csv.writer(fid, delimiter = ',')
    outvars = ["beh", "opb", "clh", "rpe", "cob", "coe", "cmr", "sve", "ecm", "ecv", "sib"]
    cw.writerow(outvars)

    for i in D.index:
        m = D.ix[i, 'snb']
        bn = D.ix[i, 'block_n']
        beb = round2(D.ix[i, 'bid'])
        b2b = round2(D.ix[i, 'b2_bid'])
        s1p = round2(D.ix[i, 's1_rp'])
        s2p = round2(D.ix[i, 's2_rp'])
        beh[m][bn-1].append(beb)
        if bn == 1: # restart pars to initial guess
            Q[m] = sp.copy(initQ[m])
            acr = 0
            prevacc = {m:1 for m in markets}
        emv[m][bn-1].append(max(Q[m]))

        sbid = sample_bd(b[m], Q[m])  
        sib[m][bn-1].append(sbid)
        
        if sp.isnan(D.ix[i, 'bid']):
            ecv[m][bn-1].append(sp.nan)
            rpe[m][bn-1].append(sp.nan)
            coe[m][bn-1].append(sp.nan)
            cob[m][bn-1].append(sp.nan)
            cmr[m][bn-1].append(sp.nan)
            clh[m][bn-1].append(sp.nan)
            sve[m][bn-1].append(sp.nan)
        else:
            if Bbins == 101: 
                ind = int(sbid * 10) 
                opb[m][bn-1].append(round2(sp.argmax(Q[m]) / 10.0))
                mmsee[m][bn-1].append(round2(sp.dot(sp.linspace(0, 10, len(Q[m])), Q[m]) / sum(Q[m])))
            elif Bbins == 11:
                ind = int(round(sbid))
                opb[m][bn-1].append(round2(sp.argmax(Q[m])))
                mmsee[m][bn-1].append(round2( sp.dot(sp.linspace(0, 10, len(Q[m])), Q[m]) / sum(Q[m])))
            if m == 'NC':
                if sbid >= s1p:
                    r = 10 - sbid 
                    cob[m][bn-1].append(sbid);
                else:
                    r = 0
                    cob[m][bn-1].append(sbid+1); # approximation
            elif m == 'SC':
                if sbid >= min(s1p, s2p):
                    r = 10 - sbid
                    cob[m][bn-1].append(sbid);
                else:
                    r = 0
                    cob[m][bn-1].append(sbid + 1); # approximation
            elif m == 'BC':
                if sbid >= min(s1p, b2b):
                    r = 10 - sbid
                else:
                    r = 0
                cob[m][bn-1].append(b2b + 0.1);

            rpei = r - Q[m][ind]
            rpe[m][bn-1].append(rpei)
            acr += r
            cmr[m][bn-1].append(acr)
            ecv[m][bn-1].append(Q[m][ind])
            coe[m][bn-1].append(10.0 - cob[m][bn-1][-1] - r)
            sve[m][bn-1].append(10.0 - cob[m][bn-1][-1] - ecv[m][bn-1][-1])
            
            p, Q[m] = dr_naive_avf_agent(a[m], b[m], Q[m], ind, r)
    
            clh[m][bn-1].append(sp.log(p))
        csvrow = [beh[m][bn-1][-1], opb[m][bn-1][-1], clh[m][bn-1][-1], rpe[m][bn-1][-1], cob[m][bn-1][-1], coe[m][bn-1][-1], cmr[m][bn-1][-1], sve[m][bn-1][-1], emv[m][bn-1][-1], ecv[m][bn-1][-1], sib[m][bn-1][-1]]   
        cw.writerow(csvrow)       
    fid.close()

    fig, axa = plt.subplots(3, 2)
    cf, ca = plt.gcf(), plt.gca()
    t = sp.linspace(1, nt, nt)    
    ep = sp.linspace(1, nb, nb)
    npnanfilt =  lambda x: x[sp.logical_not(sp.isnan(x))]

    fig.tight_layout()
    fig.suptitle('{0}: ({1}). BIC:{2}'.format(params[0], params[2], params[1]), fontsize = 14)
    plt.subplot(321)
    #plt.gca().set_color_cycle(['c', 'm', 'y'])
    minmn, maxmn = 0, 0
    for m in markets:
        fil = [npnanfilt(sp.array(l)) for l in ecv[m]]
        mn = list(map(sp.mean, fil))
        sem = list(map(sp.stats.sem, fil))
        plt.errorbar(ep, mn, yerr=sem, marker='s', mfc='c', ms=3)
        plt.title('current bid simulated value')
        plt.xlabel('block number')
        plt.ylabel('action-value')
        minmn, maxmn = min(minmn, min(mn)), max(maxmn, max(mn))
    plt.gca().set_ylim([minmn-1, maxmn+1])
    plt.legend(list(markets), loc='lower left')
    plt.subplot(322)
    #plt.gca().set_color_cycle(['c', 'm', 'y'])
    minmn, maxmn = 0, 0
    for m in markets:
        fil = [npnanfilt(sp.array(l)) for l in emv[m]]
        mn = list(map(sp.mean, fil))
        sem = list(map(sp.stats.sem, fil))
        plt.errorbar(ep, mn, sem, linestyle='-', marker='s', mfc='c', ms=3)
        plt.title('simulated maximum action-value')
        plt.xlabel('block number')
        plt.ylabel('action-value')
        minmn, maxmn = min(minmn, min(mn)), max(maxmn, max(mn))
    plt.gca().set_ylim([1, 3])
    plt.legend(list(markets), loc='lower left')
    plt.subplot(323)
    plt.gca().set_prop_cycle(color=['cyan', 'magenta', 'yellow'])
    minmn, maxmn = 0, 0
    for m in markets:
        fil1 = [npnanfilt(sp.array(l)) for l in rpe[m]]
        fil2 = [npnanfilt(sp.array(l)) for l in coe[m]]
        mn1 = list(map(sp.mean, fil1))
        mn2 = list(map(sp.mean, fil2))
        sem1 = list(map(sp.stats.sem, fil1))
        sem2 = list(map(sp.stats.sem, fil2))
        plt.errorbar(ep, mn1, sem1, marker='s', mfc='c', ms=3)
        plt.errorbar(ep, mn2, sem2, marker='^', mfc='c', ms=3)
        plt.title('simulated reward prediction error and fictive error')
        plt.xlabel('block number')
        plt.ylabel('error signal (reward - action-value|fictive-value) ')
        minmn, maxmn = min(minmn, min(min(mn1, mn2))), max(maxmn, max(max(mn1, mn2)))
    plt.gca().set_ylim([minmn-1, maxmn+1])
    plt.legend(list(markets), loc='upper left')
    plt.subplot(324)
    for m in range(3):
        #bdB(bda, bdb, bdc, ax = axa[2][0]) 
        bdB(bda[m], bdb[m], bdc, ax = axa[2][0], isab = False)
    plt.title('estimated prior action-value function')
    plt.xlabel('bid')
    plt.ylabel('action-value')
    plt.subplot(325)
    #plt.gca().set_color_cycle(['c', 'm', 'y'])
    plt.gca().set_ylim([0, 10])
    for m in markets:
        fil = [npnanfilt(sp.array(l)) for l in beh[m]]
        mn = list(map(sp.mean, fil))
        sem = list(map(sp.stats.sem, fil))
        plt.errorbar(ep, mn, sem, marker='s', ms = 3) # mfc='c') 
        plt.title('actual data')
        plt.xlabel('block number')
        plt.ylabel('bid')
    plt.legend(list(markets), loc='lower left')
    plt.subplot(326)
    #plt.gca().set_prop_cycle(['c', 'm', 'y'])
    plt.gca().set_ylim([0, 10])
    for m in markets:
        fil = [npnanfilt(sp.array(l)) for l in opb[m]]
        mn = list(map(sp.mean, fil))
        sem = list(map(sp.stats.sem, fil))
        plt.errorbar(ep, mn, sem, linestyle='-',  marker='s', ms=3) # mfc='c'
        plt.title('simulated optimal bid')
        plt.xlabel('block number')
        plt.ylabel('bid')
    #plt.legend(list(markets), loc='upper left')
 
    plt.show()
    pdb.set_trace()
    plt.close('all')


def play_dr_nudger(params):
    if len(params[2]) == 3 and params[0]=='111_ffx_naive_gausnudger1':
        a, sig, mu0 = params[2]
        q = dict((m, mu0) for m in markets)
    if len(params[2]) == 3 and params[0]=='111_ffx_dr_lepkurnudger2':
        a = {m: params[2][0] for m in markets}
        siga = {m: params[2][1] for m in markets}
        sigr = {m: params[2][2] for m in markets}
        mu0 = copy.deepcopy(initbid)
        q = dict((m, mu0[m]) for m in markets)
    elif len(params[2]) == 5 and params[0]=='131_ffx_dr_gausnudger1':
        a = {m: params[2][0] for m in markets}
        sig = {m: params[2][1] for m in markets}
        mu0 = {i[1]: params[2][2:5][i[0]] for i in enumerate(markets)}
        q = dict((m, mu0[m]) for m in markets)
    elif len(params[2]) == 5 and params[0]=='11111_ffx_dr_lepkurnudger4':
        a = {m: params[2][0] for m in markets}
        nd = {m: params[2][1] for m in markets}
        nu = {m: params[2][2] for m in markets}
        siga = {m: params[2][3] for m in markets}
        sigr = {m: params[2][4] for m in markets}
        mu0 = copy.deepcopy(initbid)
        q = dict((m, mu0[m]) for m in markets)
    elif len(params[2]) == 7 and params[0]=='1111111_ffx_dr_lepkurnudger5':
        a = {m: params[2][0] for m in markets}
        nd = {m: params[2][1] for m in markets}
        nu = {m: params[2][2] for m in markets}
        siga = {m: params[2][3] for m in markets}
        sigr = {m: params[2][4] for m in markets}
        sign = {m: params[2][5] for m in markets}
        raitc = {m: params[2][6] for m in markets}
        mu0 = copy.deepcopy(initbid)
        q = dict((m, mu0[m]) for m in markets)
    #elif len(params[2]) == 5 and params[0]=='11111_ffx_dr_lepkurnudger6':
    elif len(params[2]) == 5 and params[0]=='11111_ffx_dr_lepkurnudger6_' + list(only_cond)[0]:
        a = {m: params[2][0] for m in markets}
        siga = {m: params[2][1] for m in markets}
        sigr = {m: params[2][2] for m in markets}
        sign = {m: params[2][3] for m in markets}
        raitc = {m: params[2][4] for m in markets}
        mu0 = copy.deepcopy(initbid)
        q = dict((m, mu0[m]) for m in markets)

    nt = len(D)
    nb = max(D.ix[:, 'block_n'])
    outvars = ["beh", "opb", "clh", "rpe", "cob", "coe", "cmr", "otb", "sib"]
    for lhs in outvars:
        exec(lhs + ' = {m:[[] for i in range(nb)] for m in markets}', locals(), globals())

    if project == "econoshyuka":
        fn = 'econo_simvars_' + params[0] + '.csv' #+ '_' + list(only_cond)[0] + '.csv' 
    else:
        fn = 'simvars_' + params[0] + '.csv'
    fid = open(fn, 'w') 
    cw = csv.writer(fid, delimiter = ',')
    cw.writerow(outvars)

    for i in D.index:
        bn = D.ix[i, 'block_n']
        if project == 'econoshyuka' and bn > 24:
            bn %= 24 
        m = D.ix[i, 'snb']
        beb = round2(D.ix[i, 'bid'])
        b2b = round2(D.ix[i, 'b2_bid'])
        s1p = round2(D.ix[i, 's1_rp'])
        s2p = round2(D.ix[i, 's2_rp'])
        beh[m][bn-1].append(beb)
        if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
            q[m] = mu0[m] 
            acr = 0
            prevacc = {m:1 for m in markets}
        opb[m][bn-1].append(q[m])

        #sbid = float(sp.stats.norm.rvs(loc = q[m], scale = sig[m], size = 1)) 
        #sig = siga[m] if prevacc[m] else sigr[m]; sbid = sp.stats.laplace.rvs(loc = q[m], scale = sig) 
        #sig = siga[m] if prevacc[m] else sigr[m]; sbid = float(sp.stats.laplace.rvs(loc = q[m], scale = sig, size = 1)) 
        if prevacc[m]:
            if random.random() < raitc[m]: 
                sbiv = +abs(sp.stats.laplace.rvs(loc = 0, scale = sign[m]))
            else:
                sbiv = -abs(sp.stats.laplace.rvs(loc = 0, scale = siga[m]))
            sbid = q[m] + sbiv
        else:
            if random.random() < raitc[m]: 
                sbiv = -abs(sp.stats.laplace.rvs(loc = 0, scale = sign[m]))
            else:
                sbiv = +abs(sp.stats.laplace.rvs(loc = 0, scale = sigr[m]))
            sbid = q[m] + sbiv
        
        sib[m][bn-1].append(sbid)
        if sp.isnan(D.ix[i, 'bid']):
            clh[m][bn-1].append(sp.nan)
            rpe[m][bn-1].append(sp.nan)
            coe[m][bn-1].append(sp.nan)
            cob[m][bn-1].append(sp.nan)
            cmr[m][bn-1].append(acr)
        else: 
            rint = random.randint(1,3) # 1 2 3: S N B
            if m == 'NC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    if sbid >= s1p:
                        r = 10 - sbid
                        otb[m][bn-1].append(1)
                    else:
                        r = 0
                        otb[m][bn-1].append(0)
                    cob[m][bn-1].append(s1p); # REAL FICTIVE SIGNAL
                elif project == "econoshyuka" and {'c'} == only_cond and rint in {1, 3}:
                    if rint == 1:
                        s1p = Dp[random.randint(0,15)][opp_role['seller_SC'][bn-1]]
                        s2p = Dp[random.randint(0,15)][opp_role['seller_SC'][bn-1]]
                        if s1p == 100 or s2p == 100:
                            s1p = Dp[random.randint(0,15)][opp_role['seller_SC'][bn]]
                            s2p = Dp[random.randint(0,15)][opp_role['seller_SC'][bn]]
                        if sbid >= min(s1p, s2p):
                            r = 10 - sbid
                            otb[m][bn-1].append(1)
                            cob[m][bn-1].append(sbid);  # so use instead approx
                        else:
                            r = 0
                            otb[m][bn-1].append(0)
                            cob[m][bn-1].append(sbid + 1);  # so use instead approx
                    elif rint == 3:
                        s1p = Dp[random.randint(0,15)][opp_role['seller_BC'][bn-1]]
                        b2b = Dp[random.randint(0,15)][opp_role['buyer_BC'][bn-1]]
                        if s1p == 100 or b2b == 100:
                            s1p = Dp[random.randint(0,15)][opp_role['seller_BC'][bn]]
                            b2b = Dp[random.randint(0,15)][opp_role['buyer_BC'][bn]]
                        if sbid >= s1p and sbid > b2b:
                            r = 10 - sbid
                            otb[m][bn-1].append(1)
                            cob[m][bn-1].append(b2b + 0.1);
                        else:
                            r = 0
                            otb[m][bn-1].append(0)
                            cob[m][bn-1].append(b2b + 0.1);
                else:
                    if sbid >= s1p:
                        r = 10 - sbid
                        otb[m][bn-1].append(1)
                        cob[m][bn-1].append(sbid); # so use instead approx 
                    else:
                        r = 0
                        otb[m][bn-1].append(0)
                        cob[m][bn-1].append(sbid + 1); # so use instead approx 
            elif m == 'SC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    if sbid >= min(s1p, s2p):
                        r = 10 - sbid
                        otb[m][bn-1].append(1)
                    else:
                        r = 0
                        otb[m][bn-1].append(0)
                    cob[m][bn-1].append(min(s1p, s2p)); # REAL FICTIVE SIGNAL
                elif project == "econoshyuka" and {'c'} == only_cond and rint in {2, 3}:
                    if rint == 2:
                        s1p = Dp[random.randint(0,15)][opp_role['seller_NC'][bn-1]]
                        if s1p == 100:
                            s1p = Dp[random.randint(0,15)][opp_role['seller_NC'][bn]]
                        if sbid >= s1p:
                            r = 10 - sbid
                            otb[m][bn-1].append(1)
                            cob[m][bn-1].append(sbid); # so use instead approx 
                        else:
                            r = 0
                            otb[m][bn-1].append(0)
                            cob[m][bn-1].append(sbid + 1); # so use instead approx 
                    if rint == 3:
                        s1p = Dp[random.randint(0,15)][opp_role['seller_BC'][bn-1]]
                        b2b = Dp[random.randint(0,15)][opp_role['buyer_BC'][bn-1]]
                        if s1p == 100 or b2b == 100:
                            s1p = Dp[random.randint(0,15)][opp_role['seller_BC'][bn]]
                            b2b = Dp[random.randint(0,15)][opp_role['buyer_BC'][bn]]
                        if sbid >= s1p and sbid > b2b:
                            r = 10 - sbid
                            otb[m][bn-1].append(1)
                            cob[m][bn-1].append(b2b + 0.1);
                        else:
                            r = 0
                            otb[m][bn-1].append(0)
                            cob[m][bn-1].append(b2b + 0.1);
                else:
                    if sbid >= min(s1p, s2p):
                        r = 10 - sbid
                        otb[m][bn-1].append(1)
                        cob[m][bn-1].append(sbid);  # so use instead approx
                    else:
                        r = 0
                        otb[m][bn-1].append(0)
                        cob[m][bn-1].append(sbid + 1);  # so use instead approx
            elif m == 'BC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    if sbid >= s1p and sbid > b2b:
                        r = 10 - sbid
                        otb[m][bn-1].append(1)
                    else:
                        r = 0
                        otb[m][bn-1].append(0)
                    cob[m][bn-1].append(max(b2b + 0.1, s1p)); # REAL FICTIVE ERROR
                elif project == "econoshyuka" and {'c'} == only_cond and rint in {1, 2}:
                    if rint == 1:
                        s1p = Dp[random.randint(0,15)][opp_role['seller_SC'][bn-1]]
                        s2p = Dp[random.randint(0,15)][opp_role['seller_SC'][bn-1]]
                        if s1p == 100 or s2p == 100:
                            s1p = Dp[random.randint(0,15)][opp_role['seller_SC'][bn]]
                            s2p = Dp[random.randint(0,15)][opp_role['seller_SC'][bn]]
                        if sbid >= min(s1p, s2p):
                            r = 10 - sbid
                            otb[m][bn-1].append(1)
                            cob[m][bn-1].append(sbid);  # so use instead approx
                        else:
                            r = 0
                            otb[m][bn-1].append(0)
                            cob[m][bn-1].append(sbid + 1);  # so use instead approx
                    if rint == 2:
                        s1p = Dp[random.randint(0,15)][opp_role['seller_NC'][bn-1]]
                        if s1p == 100:
                            s1p = Dp[random.randint(0,15)][opp_role['seller_NC'][bn]]
                        if sbid >= s1p:
                            r = 10 - sbid
                            otb[m][bn-1].append(1)
                            cob[m][bn-1].append(sbid); # so use instead approx 
                        else:
                            r = 0
                            otb[m][bn-1].append(0)
                            cob[m][bn-1].append(sbid + 1); # so use instead approx 
                else:
                    if sbid >= s1p and sbid > b2b:
                        r = 10 - sbid
                        otb[m][bn-1].append(1)
                        cob[m][bn-1].append(b2b + 0.1);
                    else:
                        r = 0
                        otb[m][bn-1].append(0)
                        cob[m][bn-1].append(b2b + 0.1);
            rpei = r - 10.0 + q[m]
            rpe[m][bn-1].append(rpei)
            acr += r
            cmr[m][bn-1].append(acr)
            coe[m][bn-1].append(10.0 - cob[m][bn-1][-1] - r)

            #p, q[m] = dr_nudger1_agent(a[m], sig[m], q[m], sbid, r)
            #p, q[m] = dr_lepkurnudger2_agent(a[m], siga[m], sigr[m], q[m], sbid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 
            #p, q[m] = dr_lepkurnudger4_agent(a[m], nd[m], nu[m], siga[m], sigr[m], q[m], sbid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 
            #p, q[m] = dr_lepkurnudger5_agent(a[m], nd[m], nu[m], siga[m], sigr[m], sign[m], raitc[m], q[m], sbid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 
            p, q[m] = dr_lepkurnudger6_agent(a[m], siga[m], sigr[m], sign[m], raitc[m], q[m], sbid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 

            clh[m][bn-1].append(sp.log(p))
        csvrow = [beh[m][bn-1][-1], opb[m][bn-1][-1], clh[m][bn-1][-1], rpe[m][bn-1][-1], 
                  cob[m][bn-1][-1], coe[m][bn-1][-1], cmr[m][bn-1][-1], otb[m][bn-1][-1], sib[m][bn-1][-1]]   
        cw.writerow(csvrow)       
    fid.close()

    fig, axa = plt.subplots(3, 2)
    cf, ca = plt.gcf(), plt.gca()
    t = sp.linspace(1, nt, nt)    
    ep = sp.linspace(1, nb, nb)
    npnanfilt =  lambda x: x[sp.logical_not(sp.isnan(x))]

    fig.tight_layout()
    #fig.suptitle('{0}: {1}, {2}(static gain, static obs.noise). BIC:{3}'.format(params[0], params[2][0], params[2][1], params[1]), fontsize = 14)
    fig.suptitle('{0} (L:{1})'.format(params[0], params[1]), fontsize = 14)
    fig.subplots_adjust(top = 0.90)
    plt.subplot(323)
    #axa[1, 0].set_color_cycle(['c', 'm', 'y'])
    for m in markets:
        fil = [npnanfilt(sp.array(l)) for l in clh[m]]
        mn = list(map(sp.mean, fil))
        sem = list(map(sp.stats.sem, fil))
        plt.errorbar(ep, mn, sem, marker='s', mfc='c', ms=3)
        plt.title('current bid likelihood')
        plt.xlabel('block number')
        plt.ylabel('log10 likelihood')
    plt.gca().set_ylim([min(mn)-0.2, max(mn)+0.2])
    plt.legend(list(markets), loc='upper left', prop={'size':10})
    plt.subplot(324)
    minmn, maxmn = 0, 0
    for m in markets:
        fil = [npnanfilt(sp.array(l)) for l in rpe[m]]
        mn = list(map(sp.mean, fil))
        sem = list(map(sp.stats.sem, fil))
        plt.errorbar(ep, mn, sem, marker='s', mfc='c', ms=3)
        plt.title('simulated reward prediction error')
        plt.xlabel('block number')
        plt.ylabel('error signal units')
        minmn, maxmn = min(minmn, min(mn)), max(maxmn, max(mn))
    plt.gca().set_ylim([minmn-1, maxmn+1])
    plt.subplot(325)
    ax = fig.add_subplot(325)
    tl = sp.arange(3)
    bw = 0.5
    rl = tl - bw/2.0
    ax.bar(rl, mu0.values(), bw)  
    ax.set_title('estimated initial optimal bid')
    ax.set_xlabel('market type')
    ax.set_xticks(tl)  #plt.xticks(tl, sp.array(q.keys()))
    #ax.set_xticklabels(sp.array(q.keys()))
    ax.set_ylabel('bid')
    ax.set_ylim([0, 10])
    plt.subplot(326)
    minmn, maxmn = 0, 0
    for m in markets:
        fil = [npnanfilt(sp.array(l)) for l in coe[m]]
        mn = list(map(sp.mean, fil))
        sem = list(map(sp.stats.sem, coe[m]))
        plt.errorbar(ep, mn, sem, marker='s', mfc='c', ms=3)
        plt.title('simulated counterfactual optimal (fictive) error')
        plt.xlabel('block number')
        plt.ylabel('error signal units')
        minmn, maxmn = min(minmn, min(mn)), max(maxmn, max(mn))
    plt.gca().set_ylim([minmn-1, maxmn+1])
    plt.subplot(321)
    plt.gca().set_ylim([0, 10])
    for m in markets:
        fil1 = [npnanfilt(sp.array(l)) for l in sib[m]]
        fil2 = [npnanfilt(sp.array(l)) for l in cob[m]]
        mn1 = list(map(sp.mean, fil1))
        #mn2 = list(map(sp.mean, fil2))
        sem1 = list(map(sp.stats.sem, sib[m]))
        #sem2 = list(map(sp.stats.sem, cob[m]))
        plt.errorbar(ep, mn1, sem1, linestyle='-',  marker='s', mfc='c', ms=3)
        #plt.errorbar(ep, mn2, sem2, linestyle='-',  marker='^', mfc='c', ms=3)
        plt.title('simulated bids and counterfactual(^)-optimal bids')
        plt.xlabel('block number')
        plt.ylabel('bid')
    plt.legend(['SC-prd', 'NC-prd', 'BC-prd', 'SC-fic', 'NC-fic', 'BC-fic'], loc='lower left', prop={'size':7})
    plt.subplot(322)
    plt.gca().set_ylim([0, 10])
    for m in markets:
        fil = [npnanfilt(sp.array(l)) for l in beh[m]]
        mn = list(map(sp.mean, fil))
        sem = list(map(sp.stats.sem, beh[m]))
        plt.errorbar(ep, mn, sem, marker='s', mfc='c', ms=3)
        plt.title('behavioral data')
        plt.xlabel('block number')
        plt.ylabel('bid')

    plt.show()
    pdb.set_trace()
    plt.close('all')

def play_kalman_nudger(params):
    if len(params[2]) == 3:
        s0, mu0, sb0 = params[2]
        q = dict((m, [mu0, sb0]) for m in markets)
    elif len(params[2]) == 5:
        s0 = {m: params[2][0] for m in markets}
        mu0 = {i[1]: params[2][1:4][i[0]] for i in enumerate(markets)}
        sb0 = {m: params[2][4] for m in markets}
        q = dict((m, [mu0[m], sb0[m]]) for m in markets)

    nt = len(D)
    nb = max(D.ix[:, 'block_n'])
    opb = {m:[[] for i in range(nb)] for m in markets}
    ops = {m:[[] for i in range(nb)] for m in markets}
    rpe = {m:[[] for i in range(nb)] for m in markets}
    clh = {m:[[] for i in range(nb)] for m in markets}
    beh = {m:[[] for i in range(nb)] for m in markets}
    cob = {m:[[] for i in range(nb)] for m in markets}
    coe = {m:[[] for i in range(nb)] for m in markets}

    for i in range(len(D)):
        row = D.ix[i, 0]
        bn = D.ix[i, 'block_n']
        m = D.ix[i, 'snb']
        bid = round2(D.ix[i, 'bid'])
        s1p = round2(D.ix[i, 's1_rp'])
        b2b = round2(D.ix[i, 'b2_bid'])
        s2p = round2(D.ix[i, 's2_rp'])
        r = D.ix[i, 'profit']
        beh[m][bn-1].append(bid)
        if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
            q[m] = [mu0[m], sb0[m]] 
        opb[m][bn-1].append(q[m][0])
        ops[m][bn-1].append(q[m][1])
        if sp.isnan(D.ix[i, 'bid']):
            clh[m][bn-1].append(sp.nan)
            rpe[m][bn-1].append(sp.nan)
            coe[m][bn-1].append(sp.nan)
            cob[m][bn-1].append(sp.nan)
        else: 
            if m == 'NC':
                #cob[m][bn-1].append(s1p); # REAL FICTIVE SIGNAL, ABSENT IN THIS TASK
                cob[m][bn-1].append(q[m][0]); # so use instead RPE
            elif m == 'SC':
                #cob[m][bn-1].append(min(s1p, s2p)); # REAL FICTIVE SIGNAL, ABSENT IN THIS TASK
                cob[m][bn-1].append(q[m])[0];  # so use instead RPE
            elif m == 'BC':
                #cob[m][bn-1].append(max(b2b + 0.1, s1p)); # REAL FICTIVE ERROR, BUT s1p ABSENT IN THIS TASK
                cob[m][bn-1].append(b2b + 0.1);
            coe[m][bn-1].append(10.0 - cob[m][bn-1][-1] - r)
            rpe[m][bn-1].append(r - 10.0 + q[m][0])
            p, q[m][0], q[m][1] = kalman_nudger_agent(q[m][0], q[m][1], s0[m], bid, r)
            clh[m][bn-1].append(sp.log10(p))

    fig, axa = plt.subplots(3, 2)
    cf, ca = plt.gcf(), plt.gca()
    t = sp.linspace(1, nt, nt)    
    ep = sp.linspace(1, nb, nb)
    npnanfilt =  lambda x: x[sp.logical_not(sp.isnan(x))]

    fig.tight_layout()
    plt.suptitle('{0}: {2}. BIC:{1}'.format(params[0], params[2], params[1]), fontsize = 14)
    plt.subplot(321)
    plt.gca().set_color_cycle(['c', 'm', 'y'])
    for m in markets:
        fil = [npnanfilt(sp.array(l)) for l in clh[m]]
        mn = list(map(sp.mean, fil))
        sem = list(map(sp.stats.sem, clh[m]))
        plt.errorbar(ep, mn, sem, marker='s', mfc='c', ms=3)
        plt.title('current bid likelihood')
        plt.xlabel('block number')
        plt.ylabel('log10 likelihood')
    plt.gca().set_ylim([min(mn)-0.1, max(mn)+0.1])
    plt.legend(list(markets), loc='upper left')
    plt.subplot(322)
    plt.gca().set_color_cycle(['c', 'm', 'y'])
    minmn, maxmn = 0, 0
    for m in markets:
        fil1 = [npnanfilt(sp.array(l)) for l in rpe[m]]
        fil2 = [npnanfilt(sp.array(l)) for l in coe[m]]
        mn1 = list(map(sp.mean, fil1))
        mn2 = list(map(sp.mean, fil2))
        sem1 = list(map(sp.stats.sem, rpe[m]))
        sem2 = list(map(sp.stats.sem, coe[m]))
        plt.errorbar(ep, mn1, sem1, marker='s', mfc='c', ms=3)
        plt.errorbar(ep, mn2, sem2, marker='^', mfc='c', ms=3)
        plt.title('simulated reward prediction error and counterfactual optimal (fictive) error')
        plt.xlabel('block number')
        plt.ylabel('error signal units')
        minmn, maxmn = min(minmn, min(min(mn1, mn2))), max(maxmn, max(max(mn1, mn2)))
    plt.gca().set_ylim([minmn-1, maxmn+1])
    plt.subplot(323)
    plt.bar(sp.arange(3), mu0.values())  
    #for m in markets: plt.errorbar(sp.arange(3), mu0[m], sb0[m], marker='s', ms=3)
    plt.title('estimated initial optimal bid')
    plt.xlabel('market type')
    plt.xticks(sp.arange(3), sp.array(q.keys()))
    plt.ylabel('bid')
    plt.gca().set_ylim([0, 10])
    plt.subplot(324)
    plt.gca().set_color_cycle(['c', 'm', 'y'])
    for m in markets:
        fil = [npnanfilt(sp.array(l)) for l in ops[m]]
        mn = list(map(sp.mean, fil))
        sem = list(map(sp.stats.sem, ops[m]))
        plt.errorbar(ep, mn, sem, linestyle='-',  marker='s', mfc='c', ms=3)
        plt.title('simulated optimal bid variance')
        plt.xlabel('block number')
        plt.ylabel('bid')
    plt.gca().set_ylim([min(mn), max(mn)])
    plt.subplot(325)
    plt.gca().set_color_cycle(['c', 'm', 'y'])
    plt.gca().set_ylim([0, 10])
    for m in markets:
        fil1 = [npnanfilt(sp.array(l)) for l in opb[m]]
        fil2 = [npnanfilt(sp.array(l)) for l in cob[m]]
        mn1 = list(map(sp.mean, fil1))
        mn2 = list(map(sp.mean, fil2))
        sem1 = list(map(sp.stats.sem, opb[m]))
        sem2 = list(map(sp.stats.sem, cob[m]))
        plt.errorbar(ep, mn1, sem1, linestyle='-',  marker='s', mfc='c', ms=3)
        plt.errorbar(ep, mn2, sem2, linestyle='-',  marker='^', mfc='c', ms=3)
        plt.title('simulated prediction- and counterfactual- optimal bids')
        plt.xlabel('block number')
        plt.ylabel('bid')
    plt.subplot(326)
    plt.gca().set_color_cycle(['c', 'm', 'y'])
    plt.gca().set_ylim([0, 10])
    for m in markets:
        fil = [npnanfilt(sp.array(l)) for l in beh[m]]
        mn = list(map(sp.mean, fil))
        sem = list(map(sp.stats.sem, beh[m]))
        plt.errorbar(ep, mn, sem, marker='s', mfc='c', ms=3)
        plt.title('actual data')
        plt.xlabel('block number')
        plt.ylabel('bid')
 
    plt.show()
    pdb.set_trace()
    plt.close('all')


if __name__ == "__main__":

    plt.set_cmap('gray') # bone, jet, hot, summer, spec, prism, os, mpl, ocean, seismic, flag, hsv 
    #plt.ion() # plt.ioff(); plt.draw() for Axes methods 
