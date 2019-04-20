
from .main import *


# Random effects (summary statistics) likelihood functions

def mle_rfx_optmin_batch():
   
    def nll_ffx_1s(params):
        sD = D[D['sid'] == s]
        # nll_ffx_dr_avf  nll_ffx_dr_nudger nll_ffx_kde nll_ffx_dr_naive_nudger
        return nll_ffx_dr_avf(params, sD)

    def mle_rfx_optmin(s):
        init_pars = [0.2, 1] # yoked, dr, beta_dist(3) with mean from initbid
        bounds = [(0, 1), (0.01, 10)] 
        #init_pars = [0.2, 1] # dr, unilatupd
        #bounds = [(0, 1), (0.01, 10)] 
        #init_pars = [1, 1, 1, 1] # naive_lepkurnudger   
        #bounds = [(-10, 10), (-10, 10), (0, 10), (0, 10)] 
        #init_pars = [0.1, 1] # 11, drgausnudger1 
        #bounds = [(0, 1), (0, 10)] 
        #init_pars = [0.1, 1, 1, 1, 0.1] # 111111, dr_nudger7   
        #bounds = [(0, 1), (0, 10), (0, 10), (0, 10), (0, 1)] 
        #init_pars = [1, 1] # kde
        #bounds = [(0.1, 10), (0.1, 1)] 
        nllf = nll_ffx_1s
        method = 'L-BFGS-B'
        do_basinhopping = 0
        if not do_basinhopping:
            count = 0 
            def callbackf(params):
                nonlocal count
                print(count)
                if count % 10 == 0:
                    print(params)
                count += 1
            sp.optimize.show_options('minimize', method.lower())
            res = sp.optimize.minimize(nllf, sp.array(init_pars), bounds = bounds, 
                                       method = method, # 'L-BFGS-B' 'TNC'
                                       options = {'disp': True, 'maxiter': 10**8},
                                       callback = callbackf)
            #print(res.hess, res.hess_inv)
        else:
            minimizer_kwargs = {'method': 'L-BFGS-B', 
                                'bounds': bounds, 
                                'options':{'disp': True, 'maxiter': 10**8}}
            res = sp.optimize.basinhopping(nllf, sp.array(init_pars),
                                           minimizer_kwargs = minimizer_kwargs, 
                                           niter = 10, T = 10, stepsize =  0.1,
                                           callback = print)
        return res

    R = dict()
    S = D['sid'].unique()
    for s in S:
        print('\nSubject: ', s, '\n') 
        R[s] = mle_rfx_optmin(s)
    print(R)

    isconv = [v.success for v in R.values()]
    ll_s = [v.fun for v in R.values()]
    par_s = [v.x for v in R.values()]
    ll_m = (sp.mean(ll_s), stats.sem(ll_s, ddof=1))
    par_m = [(sp.mean([p[i] for p in par_s]), stats.sem([p[i] for p in par_s], ddof=1)) for i in range(len(par_s[0]))]
    print(par_m)

    with open('rfx_' + project + '_' + next(iter(only_cond)) + '.pkl', 'wb') as outpkl:
        pickle.dump((R, isconv, ll_m, par_m), outpkl)
    # pickle.load(open(datafile, 'rb'))

    return R




### Fixed effects numerical optimization of objective functions


def mle_ffx_optmin():
    # Minimize methods
    """http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
       http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

    ### Unconstrained minimization of multivariate scalar functions ###
    Nelder-Mead Simplex algorithm: 'Nelder-Mead'
      Simplest way to minimize a fairly well-behaved function. 
      It requires only function evaluations and is a good choice for simple minimization problems. 
      However, because it does not use any gradient evaluations, it may take longer to find the minimum.
    Powell's method: 'Powell'
      Another optimization algorithm that needs only function calls to find the minimum. 
    Broyden-Fletcher-Goldfarb-Shanno algorithm: 'BFGS'
      This routine uses the gradient of the objective function. If the gradient is not given by the user, 
      then it is estimated using first-differences. The Broyden-Fletcher-Goldfarb-Shanno (BFGS) method 
      typically requires fewer function calls than the simplex algorithm even when the gradient must be estimated. 
    Newton-Conjugate-Gradient algorithm: 'Newton-CG'
      It requires the fewest function calls and is therefore often the fastest method to minimize functions 
      of many variables. This method is a modified Newton's method and uses a conjugate gradient algorithm to 
      (approximately) invert the local Hessian. Newton's method is based on fitting the function locally to a 
      quadratic form.  
      \[ f\left(\mathbf{x}\right)\approx f\left(\mathbf{x}_{0}\right)+\nabla f\left(\mathbf{x}_{0}\right)\cdot\left(\mathbf{x}-\mathbf{x}_{0}\right)+\frac{1}{2}\left(\mathbf{x}-\mathbf{x}_{0}\right)^{T}\mathbf{H}\left(\mathbf{x}_{0}\right)\left(\mathbf{x}-\mathbf{x}_{0}\right).\]
      To take full advantage of the Newton-CG method, a function which computes the Hessian must be provided. 
      The Hessian matrix itself does not need to be constructed, only a vector which is the product of the Hessian 
      with an arbitrary vector needs to be available to the minimization routine. As a result, the user can provide
      either a function to compute the Hessian matrix, or a function to compute the product of the Hessian with an
      arbitrary vector.

    ### Constrained minimization of multivariate scalar functions ###
    Simulated annealing: 'Anneal'
      It is a probabilistic metaheuristic algorithm for global optimization. 
      It uses no derivative information from the function being optimized.
    Dog-leg trust-region algorithm for unconmin: 'dogleg'
      This algorithm requires the gradient and Hessian; furthermore the Hessian is required to be positive definite.
    Newton conjugate gradient trust-region algorithm for unconmin: 'trust-ncg'
      This algorithm requires the gradient and either the Hessian or 
      a function that computes the product of the Hessian with a given vector.
    Method L-BFGS-B: 'L-BFGS-B'
      For bound constrained minimization.
    Truncated Newton algorithm: 'TNC'
      To minimize a function with variables subject to bounds. 
      This algorithm uses gradient information; it is also called Newton Conjugate-Gradient. 
      It differs from the Newton-CG method described above as it wraps a C implementation and allows each variable 
      to be given upper and lower bounds.
    Constrained Optimization BY Linear Approximation: 'COBYLA'
      The algorithm is based on linear approximations to the objective function and each constraint. 
      The method wraps a FORTRAN implementation of the algorithm.
    Sequential Least Squares Programming Optimization Algorithm: 'SLSQP'
      This algorithm allows to deal with constrained minimization problems of the form:
      \begin{eqnarray*} \min F(x) \\ \text{subject to } & C_j(X) =  0  ,  &j = 1,...,\text{MEQ}\\
         & C_j(x) \geq 0  ,  &j = \text{MEQ}+1,...,M\\
         &  XL  \leq x \leq XU , &I = 1,...,N. \end{eqnarray*} 
       It minimizes a function of several variables with any combination of bounds, equality and inequality constraints.
       The method wraps the SLSQP Optimization subroutine originally implemented by Dieter Kraft.
       Note that the wrapper handles infinite values in bounds by converting them into large floating values.  

    ### Least square fitting: 'leastsq()' ###
      All of the previously-explained minimization procedures can be used to solve a least-squares problem provided 
      the appropriate objective function is constructed. 
      For example, suppose it is desired to fit a set of data {xi,yi} to a known model, y=f(x,p) where p is a vector 
      of parameters for the model that need to be found. A common method for determining which parameter vector gives 
      the best fit to the data is to minimize the sum of squares of the residuals. The residual is usually defined
      for each observed data-point as
        \[ei(p,yi,xi)=∥yi−f(xi,p)∥.\]
      An objective function to pass to any of the previous minization algorithms to obtain a least-squares fit is.
      \[J(p)=∑i=0N−1e2i(p).\]
      The leastsq algorithm performs this squaring and summing of the residuals automatically. It takes as an input
      argument the vector function e(p) and returns the value of p which minimizes J(p)=eTe directly. The user is also
      encouraged to provide the Jacobian matrix of the function (with derivatives down the columns or across the rows).
      If the Jacobian is not provided, it is estimated.
 
    ### Global optimization ###
      http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.basinhopping.html
    """
    init_pars = [0.2, 1] # yoked, dr, beta_dist(3) with mean from initbid
    bounds = [(0, 1), (0.01, 10)] # yoked, dr, beta_dist(3) with mean from initbid
    #init_pars = [0.2, 1, 5, 2, 4] # yoked, dr, beta_dist(3)
    #bounds = [(0, 1), (0.01, 10), (1, 100), (1, 100), (1, 6)] # yoked, dr, beta_dist(3)
    #init_pars = [1, 10, 10, 5] # yoked, pu, beta_dist(3)
    #bounds = [(0.01, 10), (1, 100), (1, 100), (1, 6)] # yoked, pu, beta_dist(3)
    #init_pars = [1, 1, 1, 1] # yoked, naive_gausnudger, initq(1) 
    #bounds = [(-10, 10), (-10, 10), (0, 10), (0, 10)] # yoked, naive_gausnudger, initq(1)
    #init_pars = [1, 5, 1] # 111, kalman,dr_nudger 
    #bounds = [(0, 10), (0, 10), (0, 10)] # 111, kalman,dr_nudger
    #init_pars = [0.2, 2, 5, 2, 5, 2, 5, 2, 5] # 33003, dr, unif(3)
    #bounds = [(0, 1), (0, 10), (1, 8), (0, 10), (1, 8), (0, 10), (1, 8), (0, 10), (2, 10)] # 33003, dr, unif(3)
    #init_pars = [0.5, 1, 5, 2, 5, 2, 5, 2, 5] # 11331, dr, bd(3)
    #bounds = [(0, 1), (0, 10), (1, 8), (0, 10), (1, 8), (0, 10), (1, 8), (0, 10), (2, 6)]  # 11331, dr, bd(3)
    #init_pars = [1, 5, 2, 5, 2, 5, 2, 5] # 1331, pu, bd(3)
    #bounds = [(0.01, 10), (0, 10), (1, 8), (0, 10), (1, 8), (0, 10), (1, 8), (2, 10)] # 1331, pu, bd(3)
    #init_pars = [1, 5, 5, 5, 1] # 131, kalman 
    #bounds = [(0, 10), (0, 10), (0, 10), (0, 10), (0, 10)] # 131, kalman
    #init_pars = [0.5, 1, 5, 5, 5] # 113, dr_nudger1 
    #bounds = [(0, 1), (0, 10), (0, 10), (0, 10), (0, 10)] # 113, dr_nudger1
    #init_pars = [0.1, 1, 5, 5, 5] # 113, dr_nudger2 
    #bounds = [(0, 1), (0, 10), (0, 10), (0, 10), (0, 10)] # 113, dr_nudger2
    #init_pars = [0.1, 1] # 11, dr_nudger1 
    #bounds = [(0, 1), (0, 10)] # 11, dr_nudger1
    #init_pars = [0.1, 1, 1] # 111, dr_nudger2 
    #bounds = [(0, 1), (0, 10), (0, 10)] # 111, dr_nudger2
    #init_pars = [0.1, 0.1, 1, 1] # 1111, dr_nudger3 
    #bounds = [(0, 1), (-10, 10), (0, 10), (0, 10)] # 1111, dr_nudger3
    #init_pars = [0.1, 0.1, 0.1, 1, 1] # 11111, dr_nudger4 
    #bounds = [(0, 1), (-10, 10), (-10, 10), (0, 10), (0, 10)] # 11111, dr_nudger4
    #init_pars = [0.1, 0.1, 0.1, 1, 1, 1, 0.1] # 11111111, dr_nudger5 
    #bounds = [(0, 1), (-10, 10), (-10, 10), (0, 10), (0, 10), (0, 10), (0, 1)] # 1111111, dr_nudger5
    #init_pars = [0.1, 1, 1, 1, 0.1] # 111111, dr_nudger6 
    #bounds = [(0, 1), (0, 10), (0, 10), (0, 10), (0, 1)] # 11111, dr_nudger6
    #init_pars = [1, 1, 1] # 111, naive_nudger1
    #bounds = [(0.1, 10), (0, 10), (0, 10)] # 111, naive_nudger1
    #init_pars = [1, 1] # 131, kalman1 
    #bounds = [(0, 10), (0, 10)] # 11, kalman1
    #init_pars = [1, 1, 1] # 111, kalman2 
    #bounds = [(0, 10), (0, 10), (0, 10)] # 111, kalman2
    #init_pars = [1, 1, 1, 1] # 1111, naive_nudger2
    #bounds = [(0, 10), (0, 10), (0, 10), (0, 10)] # 1111, naive_nudger2
    #init_pars = [1, 1] # kde
    #bounds = [(0.1, 10), (0.1, 1)] # kde
    #init_pars = [1] # kde
    #bounds = [(0.1, 10)] # kde
    #jac = nll_ffx_dr_naive_avf_jac
    method = 'L-BFGS-B'
    options = {'disp': True, 'maxiter': 10**8}
    nllf = nll_ffx_dr_avf #  nll_ffx_dr_avf nll_ffx_dr_nudger nll_ffx_kde

    do_basinhopping = 1
    if not do_basinhopping:
        sp.optimize.show_options('minimize', method.lower())
        res = sp.optimize.minimize(nllf, sp.array(init_pars), 
                                   method = method, bounds = bounds, 
                                   options = options)
        #print(res.hess, res.hess_inv)
    else:
        minimizer_kwargs = {'method': method, 'bounds': bounds, 'options': options}
        res = sp.optimize.basinhopping(nllf, sp.array(init_pars),
                                       minimizer_kwargs = minimizer_kwargs, 
                                       niter = 10, T = 10, stepsize =  0.1)
    return res
