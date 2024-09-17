import numpy as np
from numba import njit, float64, int64, prange, boolean, set_num_threads
from numba.experimental import jitclass
import matplotlib.pyplot as plt
import seaborn as sns
from interpolation import interp
from scipy.linalg import eigvals
from scipy.stats import truncnorm
from quantecon import tauchen, MarkovChain
import os
import sys
import time

# Limit the number threads used by Numba
# set_num_threads(X)

sys.path.insert(0, os.path.join(os.getcwd(), 'tnr'))
from truncated_normal_rule import truncated_normal_rule

# Some plotting setups
plt.rcParams["font.family"] = 'Times New Roman'  # 'serif', 'sans-serif', 'Times New Roman'
plt.rcParams['text.usetex'] = True


def build_dynamics(
    μ_R=1.006230606825128*(1-0.9407262699728842),  # intercept of the AR(1) process
    ρ_R=0.9407262699728842,  # slope (autocor coef) of the AR(1) process
    σ_R=0.030020587135876593*np.sqrt(1-0.9407262699728842**2),  # std dev of the AR(1) innovation
    size=13,  # number of state points for {R_t^a}
    dist=3):  # steps of std dev for Tauchen discretization
    """
    Discretize the {R_t^a} process using Tauchen's method.
    R_t^a = μ_R + ρ_R*R_{t-1}^a + σ_R*ϵ_t, {ϵ_t}~iid N(0,1).
    """
    mc = tauchen(size, ρ_R, σ_R, μ_R, dist)
    Ra_vals, Π = mc.state_values, mc.P
    return Ra_vals, Π


def gauss_quadra(n=7, trunc=5):
    """
    Calculate a Gaussian quadrature for a truncated standard normal
    distribution
    """
    option = 3
    mu, sigma = 0., 1.
    a, b = -trunc, trunc
    header = 'gauss_quad'
    truncated_normal_rule(option, n, mu, sigma, a, b, header)
    points = np.loadtxt('gauss_quad_x.txt')
    weights = np.loadtxt('gauss_quad_w.txt')
    for file in ['gauss_quad_x.txt', 'gauss_quad_w.txt', 'gauss_quad_r.txt']:
        os.remove(file)
    return points, weights


# Data for CommodityPricing
cp_data = [
    ('δ', float64),  # rate of stock deterioration
    ('λ', float64),  # price elasticity of demand
    ('κ', float64),  # unit cost of storage
    ('μ_Y', float64),  # mean of {Y_t}
    ('σ_Y', float64),  # coefficient of variation of {Y_t}
    ('trunc_Y', float64),  # nb of std at which to truncate {Y_t}
    ('p_bar', float64),    # Steady-state price
    ('Π', float64[:,:]),   # transition probability matrix for {R_t^a}
    ('Ra_vals', float64[:]),  # state points for R_t^a
    ('I_grid', float64[:]),  # grid points for I_t
    ('K', int64),  # number of I_t grid points
    ('M', int64),  # number of R_t^a states
    ('gq_points', float64[:]),   # Gaussian quadrature sample points
    ('gq_weights', float64[:]),  # Gaussian quadrature weights
    ('b', float64),  # lower bound of X_t
    ('lin_demand', boolean)  # type of demand function
]


@jitclass(cp_data)
class CommodityPricing:
    """
    A class that stores primitives for the competitive storage model
    with stochastic interest rates.
    """

    def __init__(self, Ra_vals, Π, gq_points, gq_weights,
                 δ=0.02, λ=0.1, κ=0.0, μ_Y=1.0,
                 σ_Y=0.05, trunc_Y=5., p_bar=1.0,
                 lin_demand=True,  # type of demand function
                 grid_med=0.5,    # median grid point for storage I_t
                 grid_max=2,      # maximum grid point for storage I_t
                 grid_size=100):  # size of grid points for storage I_t
        self.δ, self.λ, self.κ, self.p_bar = δ, λ, κ, p_bar
        self.μ_Y, self.σ_Y, self.trunc_Y = μ_Y, σ_Y, trunc_Y
        if lin_demand:
            self.b = np.minimum(self.μ_Y * (1 - self.σ_Y * self.trunc_Y), 0)
        else:
            self.b = self.μ_Y * (1 - self.σ_Y * self.trunc_Y)
        self.Ra_vals, self.Π = Ra_vals, Π
        self.gq_points, self.gq_weights = gq_points, gq_weights
        self.lin_demand = lin_demand

        # Set grid points for I_t: exponential grid
        tmp = (grid_med**2) / (grid_max - 2 * grid_med)
        tmp2 = np.linspace(np.log(tmp), np.log(grid_max + tmp), grid_size)
        self.I_grid = np.exp(tmp2) - tmp
        self.I_grid[0] = 0.

        self.M, self.K = len(self.Ra_vals), len(self.I_grid)

    def P(self, x):
        """Inverse demand function"""
        if self.lin_demand:
            return self.p_bar * (1.0 - (x / self.μ_Y - 1.0) / self.λ)
        else:
            return self.p_bar * (x / self.μ_Y)**(-1 / self.λ)

    def P_inv(self, p):
        """Demand function"""
        if self.lin_demand:
            return self.μ_Y * (1.0 - self.λ * (p / self.p_bar - 1.0))
        else:
            return self.μ_Y * (p / self.p_bar)**(-self.λ)


def test_stability(cp, verbose=False):
    """
    Test stability of a given instance.
    """
    R_vals = cp.Ra_vals**(1/4)
    D = np.diag(1 / R_vals)  # diag{..., (1/R_i), ...}
    G = max(np.abs(eigvals(cp.Π @ D)))  # spectral radius of ΠD
    if G < np.exp(cp.δ) and verbose:
        print("\nStability condition holds: G < exp(δ).")
    if verbose:
        print(f"\nG = {G}, exp(δ) = {np.exp(cp.δ)}")
    assert G < np.exp(cp.δ), "Stability condition failed."


@njit
def get_func(x,  # the total available supply, scalar
             m,  # index of the exogenous state, int64
             x_vals,  # X_t values for interpolation, shape=(K,M)
             y_vals,  # function evaluation at the X_t values, shape=(K,M)
             cp):  # a class that stores model information
    """
    Create demand functions via linear interpolation and linear extrapolation.
    """
    if x <= x_vals[-1, m]:  # linear interpolation in this case
        res = interp(x_vals[:, m], y_vals[:, m], x)
    else:  # linear extrapolation in this case
        slope = ((y_vals[-1, m] - y_vals[-2, m]) /
                 (x_vals[-1, m] - x_vals[-2, m]))
        res = y_vals[-1, m] + slope * (x - x_vals[-1, m])
    return np.minimum(res, cp.P_inv(0))


@njit(parallel=True)
def T(x_in,  # the total available supply X_t, shape=(K,M)
      p_in,  # the prices corresponding to the given X_t, shape=(K,M)
      cp):   # class with model information
    """
    The Coleman operator that updates the prices and
    the endogenous grid points for the amount on hand X_t.
    """
    # Simplify names
    Ra_vals, Π, M, K, b = cp.Ra_vals, cp.Π, cp.M, cp.K, cp.b
    gq_points, gq_weights = cp.gq_points, cp.gq_weights
    I_grid, P, P_inv = cp.I_grid, cp.P, cp.P_inv
    R_vals = Ra_vals**(1/4)

    # Create price function
    x_in_aug = np.vstack((b*np.ones((1,M)), x_in))     # append x=b
    p_in_aug = np.vstack((P(b)*np.ones((1,M)), p_in))  # fix price=p(b) when x=b
    d_in_aug = P_inv(p_in_aug)
    f_func = lambda x, m: P(get_func(x, m, x_in_aug, d_in_aug, cp))

    # Create empty space to store updated prices
    p_out = np.empty_like(p_in)

    # Calculate updated prices Tf(x,z) and store it in p_out
    for k in prange(K):
        I = I_grid[k]
        for j in prange(M):
            # Compute expectation
            Ez = 0.
            for m in prange(M):
                R_hat = R_vals[m]
                for n in prange(len(gq_points)):
                    Y_hat = cp.μ_Y * cp.σ_Y * gq_points[n] + cp.μ_Y
                    f_eval = f_func(Y_hat + np.exp(-cp.δ) * I, m)
                    integrand = f_eval / R_hat
                    Ez += integrand * Π[j, m] * gq_weights[n] 
            p_out[k, j] = np.exp(-cp.δ) * Ez  # Ez>=0 since f>=0 and no storage cost

    # Calculate endogenous grid points for total available supply X_t
    x_out = (I_grid + P_inv(p_out).T).T  # x = I + P^{-1}[Tf(x,z)]

    return x_out, p_out


@njit
def solve_model_time_iter(
        cp,    # class with model information
        oper,  # the operator to iterate with
        tol=1e-4,       # the level of tolerance
        max_iter=1000,  # maximum step of iteration
        verbose=True,
        print_skip=25):
    """
    Time iteration based on the endogenous grid method.
    Returns:
    --------
    x_new : the endogenous grid points for the amount on hand, shape=(K,M)
    p_new : the optimal commodity prices, shape=(K,M)
    """
    # Build initial conditions
    K, M = cp.K, cp.M
    x_init = ((cp.b+cp.I_grid) * np.ones((M, K))).T
    p_init = cp.P(x_init)

    k, error = 0, tol + 1.0
    x_vec, p_vec = x_init, p_init

    while k < max_iter and error > tol:
        x_new, p_new = oper(x_vec, p_vec, cp)
        error = np.max(np.abs(p_new - p_vec))
        k += 1

        if verbose and k % print_skip == 0:
            print("Error at iteration", k, "is", error)
        x_vec, p_vec = np.copy(x_new), np.copy(p_new)

    if k == max_iter:
        print("Failed to converge!")

    if verbose and k < max_iter:
        print("\nConverged in", k, "iterations.")

    return x_new, p_new


@njit(parallel=True)
def compute_euler_eq_error(
    cp,      # class with model information
    x_star,  # the endogonous grid point for X_t, shape=(K,M)
    p_star,  # the equilibrium price at the endogoneous grid pts, shape=(K,M)
    x_i,     # grid of available supply where to assess precision
    R_i):    # grid of gross interest rate indexes where to assess precision
    """
    Compute unit-free Euler equation error for a given solution
    """
    # Simplify names
    Ra_vals, Π, M, b = cp.Ra_vals, cp.Π, cp.M, cp.b
    gq_points, gq_weights = cp.gq_points, cp.gq_weights
    P, P_inv = cp.P, cp.P_inv
    R_vals = Ra_vals**(1/4)

    # Create empty space to store equation errors
    EE = np.empty_like(x_i)

    # Create equilibrium price function
    xstar, pstar = np.copy(x_star), np.copy(p_star)
    xstar_aug = np.vstack((b*np.ones((1,M)), xstar))    # append x=b
    pstar_aug = np.vstack((P(b)*np.ones((1,M)), pstar)) # fix price=p(b) when x=b
    dstar_aug = P_inv(pstar_aug)
    f_func = lambda x, m: P(get_func(x, m, xstar_aug, dstar_aug, cp))

    for i in prange(len(x_i)):
        d = P_inv(f_func(x_i[i], R_i[i]))
        I = x_i[i] - d
        # Compute expectation
        Ez = 0.
        for m in prange(M):
            R_hat = R_vals[m]
            for n in prange(len(gq_points)):
                Y_hat = cp.μ_Y * cp.σ_Y * gq_points[n] + cp.μ_Y
                f_eval = f_func(Y_hat + np.exp(-cp.δ) * I, m)
                integrand = f_eval / R_hat
                Ez += integrand * Π[R_i[i], m] * gq_weights[n]
        p_out = np.exp(-cp.δ) * Ez
        EE[i] = 1 - (P_inv(np.maximum(P(x_i[i]), p_out))-b) / (d-b)

    return EE


@njit
def compute_price_series(
    cp,      # class with model information
    x_star,  # the endogonous grid point for X_t, shape=(K,M)
    p_star,  # the equilibrium price at the endogoneous grid pts, shape=(K,M)
    Ra_seq,  # a time path for {R_t^a}, shape=(L+1,)
    Y_sim,   # simulated {Y_t} process, shape=(L+1,)
    L=100_000,        # length of simulated series
    burn_in=10_000):  # number of discarded samples
    """
    Simulates time series of length L for price and amount on hand processes
    {(P_t,X_t)}, given the optimal pricing rule.
    """
    P, P_inv, b, M = cp.P, cp.P_inv, cp.b, cp.M
    
    # Create equilibrium price function
    xstar, pstar = np.copy(x_star), np.copy(p_star)
    xstar_aug = np.vstack((b*np.ones((1,M)), xstar))    # append x=b
    pstar_aug = np.vstack((P(b)*np.ones((1,M)), pstar)) # fix price=p(b) when x=b
    dstar_aug = P_inv(pstar_aug)
    d_func = lambda x, m: get_func(x, m, xstar_aug, dstar_aug, cp)
    
    # Find the free-disposal threshold for total available supply: x*(z)
    x_disp = np.empty(M)
    for m in prange(M):
        if pstar[-1,m] > 0:  # linearly extrapolate and solve f*(x,z)=0
            tmp = pstar[-2,m]*xstar[-1,m] - pstar[-1,m]*xstar[-2,m]
            x_disp[m] = tmp / (pstar[-2,m] - pstar[-1,m])
        else:
            x_disp[m] = xstar[np.argmax(pstar[:,m]<=0), m]

    Ra_sim = cp.Ra_vals[Ra_seq]  # the simulated {R_t^a} process

    # Simulate {(X_t,P_t)} process
    X_sim = np.empty(L+1)
    X_sim[0] = 1.
    P_sim = np.empty_like(X_sim)
    I_sim = np.empty_like(X_sim)
    P_sim[0] = P(d_func(X_sim[0], Ra_seq[0]))
    I_sim[0] = np.minimum(X_sim[0], x_disp[Ra_seq[0]]) - d_func(X_sim[0], Ra_seq[0])

    for t in prange(L):
        X_sim[t+1] = np.exp(-cp.δ) * I_sim[t] + Y_sim[t+1]
        tmp = d_func(X_sim[t+1], Ra_seq[t+1])
        P_sim[t+1] = P(tmp)
        I_sim[t+1] = np.minimum(X_sim[t+1], x_disp[Ra_seq[t+1]]) - tmp

    return X_sim[burn_in+1:], P_sim[burn_in+1:], Ra_sim[burn_in+1:], I_sim[burn_in+1:]


@njit(parallel=True)
def calc_expectations(
        cp,           # a class that stores model information
        x_star,       # the endogenous X_t grid points, shape=(K,M)
        p_star,       # equilibrium price at endogenous (X_t,R_t^a) states, shape=(K,M)
        X_seq,        # current availability, array_like
        Ra_idx_seq):  # current annual interest rate index, shape=X_seq.shape
    """
    Calculate next-period expected price and expected variance for given states.
    """
    # Simplify names
    Π, M, δ, b, P, P_inv = cp.Π, cp.M, cp.δ, cp.b, cp.P, cp.P_inv
    gq_points, gq_weights = cp.gq_points, cp.gq_weights 
    
    # Create equilibrium price function
    x_aug = np.vstack((b*np.ones((1,M)), x_star))     # append x=b
    p_aug = np.vstack((P(b)*np.ones((1,M)), p_star))  # fix price=p(b) when x=b
    d_aug = P_inv(p_aug)
    f_func = lambda x, m: P(get_func(x, m, x_aug, d_aug, cp))

    # Initialize expectations
    EP, EV = np.zeros_like(X_seq), np.zeros_like(X_seq)
    
    for k in prange(len(X_seq)):
        X, Ra_idx = X_seq[k], Ra_idx_seq[k]    # X_t, R_t^a index
        I_t = X - cp.P_inv(f_func(X, Ra_idx))  # current storage (before free disp adjustment)
        for m in prange(M):                    # listing R_{t+1}^a states
            for n in prange(len(gq_points)):   # listing Y_{t+1} states
                # Next-period output
                Y_hat = cp.μ_Y * cp.σ_Y * gq_points[n] + cp.μ_Y
                # Next-period price f(X_{t+1}, R_{t+1}^a)
                f_eval = f_func(Y_hat + np.exp(-δ) * I_t, m)
                EP[k] += f_eval * Π[Ra_idx, m] * gq_weights[n]
                EV[k] += (f_eval**2) * Π[Ra_idx, m] * gq_weights[n]
    EV -= EP**2
    return EP, EV


def calc_diff(var, diff_type='diff'):
    ref = var[0].mean(axis=0)
    sim = var[1].mean(axis=0)
    if diff_type == 'diff':
        diff = sim - ref
    elif diff_type == 'diff_log':
        diff = np.log(sim) - np.log(ref)
    elif diff_type == 'perc':
        diff = (sim - ref) / ref
    else:
        raise Exception("Unknown diff_type")
    return diff


def find_next_Ra(cp,    # a class that stores model information
                 Ra,    # current state of {R_t^a}, an R_t^a index
                 shock, # shock to the {R_t^a} innovation {ε_t^R}
                 μ_R,   # the intercept of the AR(1) process
                 ρ_R,   # the slope (autocor coef) of the AR(1) process
                 σ_R):  # the std dev of the AR(1) innovation
    """
    Find the minimum R_{t+1}^a state that is greater than or equal to 
    μ_R + ρ_R*R_t^a + σ_R*shock. 
    """
    Ra_next_val = μ_R + ρ_R * cp.Ra_vals[Ra] + σ_R * shock
    Ra_next = np.where(Ra_next_val <= cp.Ra_vals)[0][0]
    return Ra_next


# ======================================================================== #
#                  IRF conditional on X_{t-1}, R_{t-1}                     #
# ======================================================================== #
@njit(parallel=True)
def sim_for_impulse(cp,      # class with model information
                    x_star,  # the endogenous X_t grid points, shape=(K,M)
                    p_star,  # equilibrium price at endogenous (X_t,R_t^a) states, shape=(K,M)
                    X0,      # initial availability state X_{-1}
                    Ra0,     # initial interest rate state R_{-1}^a
                    Ra_sim,  # simulated {R_t^a} indices, shape=(size,T)
                    Y_sim,   # simulated {Y_t}, shape=(size,T)
                    size,    # number of samples for averaging
                    T):      # forwarding periods
    """
    Simulating samples for computing impulse responses.
    Used by the function `irf_R2`.
    """
    δ, P, P_inv, b, M = cp.δ, cp.P, cp.P_inv, cp.b, cp.M

    # Find the free-disposal threshold for total available supply
    x_disp = np.empty(M)
    for m in prange(M):
        if p_star[-1,m] > 0:
            tmp = p_star[-2,m]*x_star[-1,m] - p_star[-1,m]*x_star[-2,m]
            x_disp[m] = tmp / (p_star[-2,m] - p_star[-1,m])
        else:
            x_disp[m] = x_star[np.where(p_star[:,m]<=0)[0][0], m]
            #x_disp[m] = x_star[np.argmax(pstar[:,m]<=0), m]

    # Create equilibrium price function
    x_aug = np.vstack((b*np.ones((1,M)), x_star))     # append x=b
    p_aug = np.vstack((P(b)*np.ones((1,M)), p_star))  # fix price=p(b) when x=b
    d_aug = P_inv(p_aug)
    f_func = lambda x, m: P(get_func(x, m, x_aug, d_aug, cp))

    P_sim = np.empty((size,T))    # simulated {P_t}, shape=(size,T)
    I_sim = np.empty_like(P_sim)  # simulated {I_t}, shape=(size,T)
    X_sim = np.empty_like(P_sim)  # simulated {X_t}, shape=(size,T)
    
    for i in prange(size):
        X_pd0 = np.minimum(X0, x_disp[Ra0])  # X_{-1} after disposal
        I0 = X_pd0 - P_inv(f_func(X0, Ra0))  # I_{-1}
        X_sim[i,0] = np.exp(-δ)*I0 + Y_sim[i,0]             # X_0
        P_sim[i,0] = f_func(X_sim[i,0], Ra_sim[i,0])        # P_0
        X_pd0 = np.minimum(X_sim[i,0], x_disp[Ra_sim[i,0]]) # X_0 after disposal
        I_sim[i,0] = X_pd0 - P_inv(P_sim[i,0])              # I_0
        for j in prange(T-1):
            X_sim[i,j+1] = np.exp(-δ)*I_sim[i,j] + Y_sim[i,j+1]    # next-period availability
            P_sim[i,j+1] = f_func(X_sim[i,j+1], Ra_sim[i,j+1])     # next-period price
            X_pd = np.minimum(X_sim[i,j+1], x_disp[Ra_sim[i,j+1]]) # next-period availability after disposal
            I_sim[i,j+1] = X_pd - P_inv(P_sim[i,j+1])              # next-period inventory 
    
    return P_sim, X_sim, I_sim  # shape=(size,T)


def irf_R(cp,            # a class that stores model information
          x_star,        # the endogenous X_t grid points, shape=(K,M)
          p_star,        # equilibrium price at endogenous (X_t,R_t^a) states, shape=(K,M)
          init_state,    # the initial state from which to calculate IRF
          shocks_irf,    # the shocks that constitute the impulse, an R_t^a index
          Y_seed=5678,   # random seed for simulating {Y_t}
          diff_type='perc',  # method of differentiation to compute IRF
          size=100_000,  # size of the sample over which to average
          T=16):         # forwarding periods
    """
    Calculating impulse responses to interest rate shocks.
    The IRFs are calculated conditional on X_{t-1}.
    """
    P, P_inv, b = cp.P, cp.P_inv, cp.b
    X0, Ra0 = init_state[0], init_state[1]  # X_{-1}, R_{-1}^a
    Y_sim = truncnorm.rvs(a=-cp.trunc_Y,
                          b=cp.trunc_Y,
                          size=(size,T),
                          loc=cp.μ_Y,
                          scale=cp.μ_Y*cp.σ_Y,
                          random_state=Y_seed)  # simulated {Y_t}, shape=(size,T)
    
    # The simulated {R_t^a} process with and without impulse shocks, shape=(size,T)
    mc_Ra = MarkovChain(cp.Π)
    Ra_sim_base = np.empty((size,T+1), dtype=int)
    Ra_sim_irf = np.empty((size,T), dtype=int)
    for i in range(size):
        Ra_sim_base[i,:] = mc_Ra.simulate(T+1, init=Ra0, random_state=i)
        Ra_sim_irf[i,:] = mc_Ra.simulate(T, init=shocks_irf, random_state=i)
    Ra_samp_set = [Ra_sim_base[:,1:], Ra_sim_irf]
    X_samp_set, P_samp_set = [], []
    I_samp_set, I_pcg_samp_set = [], []
    EP_samp_set, EV_samp_set = [], []
    
    for k in range(len(Ra_samp_set)):
        Ra_sim = Ra_samp_set[k]  # simulated {R_t^a} index samples, shape=(size,T)
        
        # Simulate {P_t}, {X_t}, {I_t} processes, shape=(size,T) for each
        P_sim, X_sim, I_sim = sim_for_impulse(cp, x_star, p_star, X0, Ra0, 
                                              Ra_sim, Y_sim, size, T)
        
        I_pcg_sim = I_sim/(X_sim-b)  # simulated {I_t/(X_t-b)}, shape=(size,T)
        X_samp_set.append(X_sim)  # append the simulated {X_t}
        P_samp_set.append(P_sim)  # append the simulated {P_t}
        I_samp_set.append(I_sim)  # append the simulated {I_t}
        I_pcg_samp_set.append(I_pcg_sim)  # append the simulated {I_t/(X_t-b)}
        EP_sim, EV_sim = calc_expectations(cp,
                                           x_star,
                                           p_star,
                                           X_sim.flatten(),
                                           Ra_sim.flatten())
        EP_samp_set.append(EP_sim.reshape(I_sim.shape))
        EV_samp_set.append(EV_sim.reshape(I_sim.shape))

    impul_I_pcg = calc_diff(I_pcg_samp_set, 'diff')
    impul_P = calc_diff(P_samp_set, diff_type)
    impul_I = calc_diff(I_samp_set, diff_type)
    impul_X = calc_diff(X_samp_set, diff_type)
    impul_ES = calc_diff(np.sqrt(EV_samp_set), diff_type)
    
    return impul_P, impul_I, impul_I_pcg, impul_X, impul_ES


def irf_vs_params(δ,  # depreciation rate
                  λ,  # demand elasticity
                  L,  # length of time series samples to approx stationary dist
                  burn_in,      # No of discarded samples to approx stat dist
                  shocks_irf,   # shock to the {R_t^a} innovation
                  Ra0,          # the initial interest rate state R_{-1}^a
                  X0_perc,      # percentile of the initial availability state X_{-1}
                  Ra_vals,      # the discrete {R_t^a} states
                  Π,            # the prob trans matrix of discrete {R_t^a} process
                  Ra_seq,       # the simulated {R_t^a} index seq to approx stat dist
                  Y_sim,        # the simulated {Y_t} sequence to approx stat dist
                  gq_points,    # Gaussian quadrature sample points
                  gq_weights,   # Gaussian quadrature weights
                  Y_seed=5678,  # random seed of {Y_t} for computing IRFs
                  size=100_000, # MC sample size for computing IRFs
                  periods=1,    # forecasting periods of IRF
                  diff_type='perc'):
    """
    IRFs under different (δ,λ) parameter setups.
    """
    cp = CommodityPricing(δ=δ, λ=λ, κ=0.0, μ_Y=1.0, σ_Y=0.05,
                          Ra_vals=Ra_vals, Π=Π, grid_size=100,
                          gq_points=gq_points, gq_weights=gq_weights)
    test_stability(cp, verbose=False)
    x_star, p_star = solve_model_time_iter(cp, T, verbose=False)
    # Compute the stationary distribution of (X_t,R_t^a)
    X_sim, P_sim, Ra_sim, I_sim = compute_price_series(cp, x_star, p_star, Ra_seq,
                                                       Y_sim, L=L, burn_in=burn_in)
    # Compute key moments
    ac_P = np.corrcoef(P_sim[1:], P_sim[:-1])[0,1]  # autocorrelation coef of P
    std_P = np.std(P_sim)                           # standard deviation of P
    cv_P = std_P / np.mean(P_sim)                   # coef of variation of P
    cor_RP = np.corrcoef(Ra_sim, P_sim)[0,1]        # correlation coef of R and P
    moments = (ac_P, std_P, cv_P, cor_RP)
    
    # Compute impulse responses
    X0 = X_sim.mean() if X0_perc==None else np.quantile(X_sim, X0_perc) 
    res = irf_R(cp, x_star, p_star, [X0, Ra0], shocks_irf,
                Y_seed=Y_seed, diff_type=diff_type, size=size, T=periods)
    impul_P, impul_I, impul_I_pcg, impul_X, impul_ES = res
    
    return impul_P, impul_I, impul_I_pcg, impul_X, impul_ES, moments
