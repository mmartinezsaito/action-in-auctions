####################################
###  Various function utilities  ###
####################################

def round2(n):
    return round(10 * n) / 10

def buyer_profit(bid):
    return 10 - bid

def num_hessian(x0, cost_function, epsilon = 1.e-5, linear_approx = False, *args):
    """ A numerical approximation to the Hessian matrix of cost 
    function at location x0 (hopefully, the minimum) """
    # The next line calculates an approximation to the first derivative
    f1 = sp.optimize.approx_fprime(x0, cost_function, *args) 
    # This is a linear approximation, efficient if cost function is linear
    if linear_approx:
        f1 = sp.matrix(f1)
        return f1.transpose() * f1    
    # Allocate space for the hessian
    n = x0.shape[0]
    hessian = sp.zeros ((n, n))
    # The next loop fill in the matrix
    xx = x0
    for j in xrange(n):
        xx0 = xx[j] # Store old value
        xx[j] = xx0 + epsilon # Perturb with finite difference
        # Recalculate the partial derivatives for this new point
        f2 = sp.optimize.approx_fprime(x0, cost_function, *args) 
        hessian[:, j] = (f2 - f1)/epsilon # scale...
        xx[j] = xx0 # Restore initial value of x0        
    return hessian

# Root finding algorithms
def srf(x, bdu, bdv, bdmax = 10):
    return bdmax - beta_dist_maxB(bdu, bdv, x) #x * (1 - bdmax / beta_dist_maxB(bdu, bdv, x)) 
def newtonraphson_secant(x0, args):
    """ Newton-Raphson and Secant methods: find a zero of a scalar function
      http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.newton.html#scipy.optimize.newton 
     Find a zero of the function func given a nearby starting point x0. 
     The Newton-Raphson method is used if the derivative fprime of func is provided, otherwise the secant method is used. 
     If the second order derivate fprime2 of func is provided, parabolic Halley’s method is used.
     The convergence rate of the Newton-Raphson method is quadratic, the Halley method is cubic, and the secant method is sub-quadratic. 
     This means that if the function is well behaved the actual error in the estimated zero is approximately 
      the square (cube for Halley) of the requested tolerance up to roundoff error. 
     However, the stopping criterion used here is the step size and there is no guarantee that a zero has been found. 
     Consequently the result should be verified. Safer algorithms are brentq, brenth, ridder, and bisect, but they all 
      require that the root first be bracketed in an interval where the function changes sign. The brentq algorithm is 
      recommended for general use in one dimensional problems when such an interval has been found.
    """
    func = srf
    fprime = None # secant is used if this is None
    if args[0:2] == (5, 1): 
        return args[2]
    zero = sp.optimize.newton(func, x0, fprime = fprime, args = args)
    return zero
def brent(a, b, args):
    """ Brent's method: find a root of a scalar function in a given interval
      http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.brentq.html#scipy.optimize.brentq
     Return float, a zero of f between a and b. f must be a continuous function, and [a,b] must be a sign changing interval.
     Uses the classic Brent (1973) method to find a zero of the function f on the sign changing interval [a , b]. 
      Generally considered the best of the rootfinding routines here. It is a safe version of the secant method that uses 
      inverse quadratic extrapolation. Brent’s method combines root bracketing, interval bisection, and inverse quadratic 
      interpolation. It is sometimes known as the van Wijngaarden-Dekker-Brent method. Brent (1973) claims convergence is 
      guaranteed for functions computable within [a,b].
    """
    f = srf
    if args[0:2] == (5, 1):
        return args[2]
    x0, r = sp.optimize.brentq(f, a, b, args = args, disp= True, full_output = True)   
    return x0






# Naive CHL-0 * logisticP(reject)
def logisticB(loc, scale):
    logistic_frz = sp.stats.logistic(loc = loc, scale = scale)
    return sp.array([logistic_frz.pdf(i / 10) for i in B])[::-1]
def CH0lPrB(loc):
    scale = 1
    arr_lPr = logisticB(loc, scale)
    arr_CH0 = sp.array([float(buyer_profit(i)) for i in B]) #initQd_CH0 = {round2(i): buyer_profit(i) for i in B} 
    return arr_lPr * arr_CH0

# Uniform initialization to domain mean
def unifB(q):
     return sp.array([float(q) for i in B]) #initQd_unif = {round2(i): q0 for i in B}

# Parametric Beta distribution initialization
def ab2uv(a, b):
    return a / float(a + b), -sp.log(a + b) 
def uv2ab(u, v):
    return sp.exp(-v) * u, sp.exp(-v) * (1 - u) 
"""https://en.wikipedia.org/wiki/List_of_probability_distributions"""
def beta_distB(u, v, c, isab = False):
    #u = a / (a + b) # mean 
    #v = -sp.log(a + b) # volatility, in log-space, the natural space for a variance parameter
    if not isab:
        u *= 0.1 
        v = -v * sp.log(2) 
        a, b = uv2ab(u, v)
    else:
        a, b = u, v
    sc = 10
    beta_frz = sp.stats.beta(a, b, loc = 0 - 10**-4, scale = sc + 2*10**-4)
    return sp.array([c * sc * beta_frz.pdf(i) for i in B])
def gamma(z):
    return sp.special.gamma(z)
def digamma(z): # logarithmic derivative of the gamma function
    return sp.special.psi(z)
def beta_distB_da(a, b, c):
    return beta_distB(a, b, c) * (-digamma(a + b) + digamma(a))
def beta_distB_db(a, b, c):
    return beta_distB(a, b, c) * (-digamma(a + b) + digamma(b))
def beta_dist_mode(a, b):
    if b < 1:
        return 1
    if a < 1:
        return 0
    return float(a - 1) / (a + b - 2)
def beta_dist_max(a, b):
    return sp.stats.beta.pdf(beta_dist_mode(a, b), a = a, b = b, loc = 0 - 10**-4, scale = 1 + 2*10**-4)
def beta_dist_maxB(u, v, c):
    u *= 0.1 
    v = -v * sp.log(2) 
    a, b = uv2ab(u, v)
    return c * sp.stats.beta.pdf(beta_dist_mode(a, b), a = a, b = b, loc = 0 - 10**-4, scale = 1 + 2*10**-4)

# plot beta distribution
def bdB(u, v, c, ax = None, isab = False):
    #u = a / (a + b) # mean 
    #v = -sp.log(a + b) # volatility, in log-space, the natural space for a variance parameter
    if not isab:
        u *= 0.1 
        v = -v * sp.log(2)
        a, b = uv2ab(u, v)
    else: 
        a, b = u, v
    sc = 10
    prior = functools.partial(sp.stats.beta.pdf, a = a, b = b, scale = sc)
    x = sp.linspace(0, 10, num = 200)
    plt.title("%s * Beta(%s, %s) scaled %sx%s "% (c, a, b, sc, sc))
    c *= sc
    y = c * prior(x) # c * sp.array(map(prior, x))
    plt.plot(x, y)
    if not ax:
        plt.show()


## Policy / observation model

def partition_func(b, L):
    return sum([sp.exp(b * i) for i in L])

def boltzmann_dist(b, E, i):
    Z = partition_func(b, E)
    return sp.exp(b * E[i]) / Z




def ssr_optls():
    """http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html#scipy.optimize.leastsq"""
    x0 = [0.1, 1]
    func = 0
    bounds = [(0, 1), (10**-3, 10**2)]
    res = sp.optimize.leastsq(func, x0)
    print(res.x, '\n', res.conv_x, '\n', res.mesg, '\n', res.ler)
    return res











def sample_bd(b, E):
    CZ = [partition_func(b, E[:i]) for i in range(len(E))]
    rv = random.uniform(0, CZ[-1]) 
    rind = next((i for i, x in enumerate(CZ) if x > rv), None)
    rind /= 10.0 if Bbins == 101 else rind
    return round2(rind)

def confint_fisherinfo(hes, conflev):
    obs_fisherinfo_atext = hes # assuming the hessian of the negloglhf at the min
    covmat = sp.linalg.inv(hes)  # the inverse of the hessian is an estimator of the asymptotic cov matrix
    siglev = 1 - conflev
    q = 1 - siglev/2
    critlev = -sp.stats.norm.isf(q)
    ci_vec = [critlev * sp.sqrt(covmat[i, i]) for i in range(covmat.shape[0])]
    return ci_vec 

# Bayesian Information Criterion
def bic(L, m, n):
   return -2 * L + n * sp.log10(m)

# Akaike Information Criterion corrected for finite sample bias
def caic(L, m, n):
   aic = -2 * L + 2 * n # uncorrected
   return aic + (2 * n * (n + 1)) / (m - n - 1)

