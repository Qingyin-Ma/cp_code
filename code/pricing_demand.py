import numpy as np
from numba import njit, float64, int64, prange, boolean, set_num_threads
from numba.experimental import jitclass
import matplotlib.pyplot as plt
from interpolation import interp
from scipy.linalg import eigvals
from scipy.stats import truncnorm
from quantecon import MarkovChain
from quantecon.markov.approximation import discrete_var
import os
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), 'tnr'))
from truncated_normal_rule import truncated_normal_rule

# Limit the number threads used by Numba
# set_num_threads(X)

# Some plotting setups
plt.rcParams["font.family"] = 'Times New Roman'  # 'serif', 'sans-serif', 'Times New Roman'
plt.rcParams['text.usetex'] = True

# Some global parameters
μ_R = 1.006230606825128     # stationary mean of the AR(1) interest rate process
ρ_R = 0.9407262699728842    # autocor coef of the AR(1) interest rate process
σ_R = 0.030020587135876593  # stationary std dev of the AR(1) interest rate process
ρ_EA = 0.52  # slope of the economic activity process with respect to its lag value
γ = 0.95     # slope of the economic activity process with respect to interest rate

def build_dynamics(μ_R=μ_R, ρ_R=ρ_R, σ_R=σ_R, ρ_EA=ρ_EA, γ=γ, 
                   grid_sizes=[13, 11],  # number of state points per dimension
                   random_state=1234):   # random seed for discretizing the VAR(1) process
    """
    Discretize the exogenous Markov process and 
    compute the probability transition matrix. 
    """
    B = np.array([[ρ_R, 0], [0, ρ_EA]])
    A = np.array([[1, 0], [γ, 1]])
    Σ = np.array([[σ_R * np.sqrt(1 - ρ_R**2), 0], [0, 1e-8]])
    A_tilde = np.linalg.inv(A) @ B
    Σ_tilde = np.linalg.inv(A) @ Σ
    mc = discrete_var(A=A_tilde, C=Σ_tilde, grid_sizes=grid_sizes,
                      random_state=random_state)
    mc.state_values[:,0] = mc.state_values[:,0] + μ_R
    Π = mc.P
    Ra_vals = mc.state_values[:,0]
    EA_vals = mc.state_values[:,1]
    return Π, Ra_vals, EA_vals


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
    ('α', float64),  # effect of economic activity on commodity demand
    ('p_bar', float64),  # Steady-state price
    ('Π', float64[:,:]),  # transition probability matrix for {R_t}
    ('Ra_vals', float64[:]),  # state points for R_t^a
    ('EA_vals', float64[:]),  # state points for EA_t
    ('I_grid', float64[:]),  # grid points for I_t
    ('K', int64),  # number of I_t grid points
    ('M', int64),  # number of R_t states
    ('gq_points', float64[:]),  # Gaussian quadrature sample points
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

    def __init__(
            self, Ra_vals, EA_vals, Π, gq_points, gq_weights,
            δ=0.02, λ=0.1, κ=0.0, μ_Y=1.0,
            σ_Y=0.05, trunc_Y=5., p_bar=1.0,
            α=0,
            lin_demand=True,  # type of demand function
            grid_med=0.5,    # median grid point for storage I_t
            grid_max=2,      # maximum grid point for storage I_t
            grid_size=100):  # size of grid points for storage I_t
        # Simplify names
        self.δ, self.λ, self.κ, self.p_bar = δ, λ, κ, p_bar
        self.μ_Y, self.σ_Y, self.trunc_Y = μ_Y, σ_Y, trunc_Y
        self.α = α
        if lin_demand:
            self.b = np.minimum(self.μ_Y * (1 - self.σ_Y * self.trunc_Y), 0)
        else:
            self.b = self.μ_Y * (1 - self.σ_Y * self.trunc_Y)
        self.Ra_vals, self.EA_vals, self.Π = Ra_vals, EA_vals, Π
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
    Create demand function via linear interpolation and linear extrapolation.
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
    Ra_vals, EA_vals, Π = cp.Ra_vals, cp.EA_vals, cp.Π
    M, K, b, α, δ = cp.M, cp.K, cp.b, cp.α, cp.δ
    gq_points, gq_weights = cp.gq_points, cp.gq_weights
    I_grid, P, P_inv = cp.I_grid, cp.P, cp.P_inv
    R_vals = Ra_vals**(1/4)

    # Create price function
    x_in_aug = np.vstack((b*np.ones((1,M)), x_in))     # append x=b
    p_in_aug = np.vstack((P(b)*np.ones((1,M)), p_in))  # fix price=P(b) when x=b
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
                EA_hat = EA_vals[m]
                for n in prange(len(gq_points)):
                    Y_S_hat = cp.μ_Y * cp.σ_Y * gq_points[n] + cp.μ_Y
                    Y_hat = Y_S_hat - α * EA_hat
                    f_eval = f_func(Y_hat + np.exp(-δ) * I, m)
                    integrand = f_eval / R_hat
                    Ez += integrand * Π[j,m] * gq_weights[n]
            p_out[k,j] = np.exp(-δ) * Ez  # Ez>=0 since f>=0 and no storage cost

    # Calculate endogenous grid points for total available supply X_t
    x_out = (I_grid + P_inv(p_out).T).T  # x = I + P^{-1}[Tf(x,z)]

    return x_out, p_out


@njit
def solve_model_time_iter(
        cp,  # class with model information
        oper,  # the operator to iterate with
        tol=1e-4,  # the level of tolerance
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


@njit
def compute_price_series(
        cp,         # class with model information
        x_star,     # the endogonous grid point for X_t, shape=(K,M)
        p_star,     # the equilibrium price at the endogoneous grid pts, shape=(K,M)
        Z_seq,      # a time path for exogenous state Z_t, shape=(L+1,)
        Y_S_sim,    # simulated {Y_t^S} process, shape=(L+1,)
        L=100_000,  # length of simulated series
        burn_in=10_000):  # number of discarded samples
    """
    Simulates time series of length L for price and amount on hand processes
    {(P_t,X_t)}, given the optimal pricing rule.
    """
    M, b, α, δ = cp.M, cp.b, cp.α, cp.δ
    P, P_inv = cp.P, cp.P_inv

    # Create equilibrium price function
    xstar, pstar = np.copy(x_star), np.copy(p_star)
    xstar_aug = np.vstack((b*np.ones((1,M)), xstar))
    pstar_aug = np.vstack((P(b)*np.ones((1,M)), pstar))
    dstar_aug = P_inv(pstar_aug)
    f_func = lambda x, m: P(get_func(x, m, xstar_aug, dstar_aug, cp))

    # Find the free-disposal threshold for total available supply: x*(z)
    x_disp = np.empty(M)
    for m in prange(M):
        if pstar[-1,m] > 0:  # linearly extrapolate and solve f*(x,z)=0
            tmp = pstar[-2,m]*xstar[-1,m] - pstar[-1,m]*xstar[-2,m]
            x_disp[m] = tmp / (pstar[-2,m] - pstar[-1,m])
        else:
            x_disp[m] = xstar[np.argmax(pstar[:,m]<=0), m]

    Ra_sim = cp.Ra_vals[Z_seq]    # the simulated {R_t^a} process
    EA_sim = cp.EA_vals[Z_seq]    # the simulated {EA_t} process
    Y_sim = Y_S_sim - α * EA_sim  # net supply shock

    # Simulate {(X_t,P_t)} process
    X_sim = np.empty(L+1)
    X_sim[0] = 1.
    P_sim = np.empty_like(X_sim)
    I_sim = np.empty_like(X_sim)
    P_sim[0] = f_func(X_sim[0], Z_seq[0])
    I_sim[0] = X_sim[0] - P_inv(f_func(X_sim[0], Z_seq[0]))

    for t in prange(L):
        X_sim[t+1] = np.exp(-δ) * I_sim[t] + Y_sim[t+1]
        P_sim[t+1] = f_func(X_sim[t+1], Z_seq[t+1])
        I_sim[t+1] = (np.minimum(X_sim[t+1], x_disp[Z_seq[t+1]])
                      - P_inv(f_func(X_sim[t+1], Z_seq[t+1])))

    return X_sim[burn_in+1:], P_sim[burn_in+1:], Ra_sim[burn_in+1:], EA_sim[burn_in+1:], I_sim[burn_in+1:]


@njit(parallel=True)
def calc_expectations(
        cp,          # a class that stores model information
        x_star,      # the endogenous X_t grid points, shape=(K,M)
        p_star,      # equilibrium price at endogenous (X_t,Z_t) states, shape=(K,M)
        X_seq,       # current availability, array_like
        Z_idx_seq):  # current index of exogenous state, shape=X_seq.shape
    """
    Calculate next-period expected price and expected variance for given states.
    """
    # Simplify names
    Π, M, δ, b, α = cp.Π, cp.M, cp.δ, cp.b, cp.α
    gq_points, gq_weights = cp.gq_points, cp.gq_weights
    P, P_inv = cp.P, cp.P_inv

    # Create equilibrium price function
    x_aug = np.vstack((b*np.ones((1,M)), x_star))     # append x=b
    p_aug = np.vstack((P(b)*np.ones((1,M)), p_star))  # fix price=p(b) when x=b
    d_aug = P_inv(p_aug)
    f_func = lambda x, m: P(get_func(x, m, x_aug, d_aug, cp))

    # Initialize expectations
    EP, EV = np.zeros_like(X_seq), np.zeros_like(X_seq)
    EcovPR, ER = np.zeros_like(X_seq), np.zeros_like(X_seq)

    for k in prange(len(X_seq)):
        X, Z_idx = X_seq[k], Z_idx_seq[k]  # X_t, Z_t index
        I_t = X - P_inv(f_func(X, Z_idx))  # current storage (before free disp adjustment)
        for m in prange(M):                # listing R_{t+1} states
            Y_D_hat = α * cp.EA_vals[m]
            R_hat = (cp.Ra_vals[m])**(1/4)
            ER[k] += Π[Z_idx, m] * R_hat
            for n in prange(len(gq_points)):  # listing Y_{t+1} states
                # Next-period output
                Y_S_hat = cp.μ_Y * cp.σ_Y * gq_points[n] + cp.μ_Y
                Y_hat = Y_S_hat - Y_D_hat
                # Next-period price
                f_hat = np.maximum(f_func(Y_hat + np.exp(-δ) * I_t, m), 0)
                EP[k] += f_hat * Π[Z_idx, m] * gq_weights[n]
                EV[k] += (f_hat**2) * Π[Z_idx, m] * gq_weights[n]
                EcovPR[k] += R_hat * f_hat * Π[Z_idx, m] * gq_weights[n]
    EV -= EP**2
    EcovPR -= EP * ER
    return EP, EV, EcovPR


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


# ======================================================================== #
#                   IRF conditional on X_{t-1}, Z_{t-1}                    #
# ======================================================================== #
@njit
def find_closest(array, value):
    "Find the nearest value in an array."
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


@njit
def find_Zs(Ra_sim,   # simulated {R_t} sequence
            Ra_perc,  # targeted percentile of R_t
            Ra_vals,  # discrete R_t state values
            EA_sim,   # simulated {EA_t} sequence
            EA_perc,  # targeted percentile of EA_t
            EA_vals,  # discrete EA_t state values
            shock,    # shock to the interest rate
            γ=γ,      # slope of economic activity w.r.t interest rate
            μ_R=μ_R,  # stationary mean of the AR(1) interest rate process
            ρ_R=ρ_R,  # autocor coef of the AR(1) interest rate process
            σ_R=σ_R,  # stationary std dev of the AR(1) interest rate process
            ρ_EA=ρ_EA):  # slope of the economic activity process w.r.t its lag value          
    """
    Find the initial exogenous state, and the new exogenous state after the shock.
    """
    # Initial exogenous state index
    Ra0 = np.percentile(Ra_sim, Ra_perc)  # fix R_{-1}^a at the given percentile
    EA0 = np.percentile(EA_sim, EA_perc)  # fix EA_{-1} at the given percentile
    Z0_idx = np.where((EA_vals == EA0) & (Ra_vals == Ra0))[0][0]  # find Z_{-1} index

    # New exogenous state index after the interest rate shock
    Ra_next = μ_R*(1-ρ_R) + ρ_R * Ra0 + shock   # theoretical interest rate level after the shock
    Ra_irf = find_closest(Ra_vals, Ra_next)     # approximate interest rate level after the shock
    EA_next = ρ_EA * EA0 - γ * (Ra_next - μ_R)  # theoretical economic activity after the shock
    EA_irf = find_closest(EA_vals, EA_next)     # approximate economic activity after the shock
    Z_irf_idx = np.where((Ra_vals == Ra_irf) & (EA_vals == EA_irf))[0][0]  # find exog state index after the shock

    return Z0_idx, Z_irf_idx


@njit(parallel=True)
def sim_for_impulse(
        cp,       # class with model information
        x_star,   # the endogenous X_t grid points, shape=(K,M)
        p_star,   # equilibrium price at endogenous (X_t,Z_t) states, shape=(K,M)
        X0,       # initial availability state X_{-1}, shape=scalar
        Z0,       # index of initial exogenous state Z_{-1}, shape=scalar
        Z_sim,    # simulated indices of {Z_t}, shape=(size,T)
        Y_S_sim,  # simulated {Y_t^S} samples, shape=(size,T)
        size,     # number of samples for averaging
        T):       # forwarding periods
    """
    Simulating samples for computing impulse responses.
    Used by the function `irf_R`.
    """
    δ, P, P_inv, b, α = cp.δ, cp.P, cp.P_inv, cp.b, cp.α
    M, EA_vals = cp.M, cp.EA_vals

    # Find the free-disposal threshold for total available supply
    x_disp = np.empty(M)
    for m in prange(M):
        if p_star[-1,m] > 0:
            tmp = p_star[-2,m]*x_star[-1,m] - p_star[-1,m]*x_star[-2,m]
            x_disp[m] = tmp / (p_star[-2,m] - p_star[-1,m])
        else:
            x_disp[m] = x_star[np.where(p_star[:,m] <= 0)[0][0], m]

    # Create equilibrium price function
    x_aug = np.vstack((b*np.ones((1,M)), x_star))  # append x=b
    p_aug = np.vstack((P(b)*np.ones((1,M)), p_star))  # fix price=p(b) when x=b
    d_aug = P_inv(p_aug)
    f_func = lambda x, m: P(get_func(x, m, x_aug, d_aug, cp))

    P_sim = np.empty((size,T))    # simulated {P_t}, shape=(size,T)
    I_sim = np.empty_like(P_sim)  # simulated {I_t}, shape=(size,T)
    X_sim = np.empty_like(P_sim)  # simulated {X_t}, shape=(size,T)
    Y_sim = np.empty_like(P_sim)  # simulated {Y_t}, shape=(size,T)

    for i in prange(size):
        X_pd0 = np.minimum(X0, x_disp[Z0])  # X_{-1} after disposal
        I0 = X_pd0 - P_inv(f_func(X0, Z0))  # I_{-1}
        Y_sim[i,0] = Y_S_sim[i,0] - α*EA_vals[Z_sim[i,0]]   # Y_0
        X_sim[i,0] = np.exp(-δ)*I0 + Y_sim[i,0]             # X_0
        P_sim[i,0] = f_func(X_sim[i,0], Z_sim[i,0])         # P_0
        X_pd0 = np.minimum(X_sim[i,0], x_disp[Z_sim[i,0]])  # X_0 after disposal
        I_sim[i,0] = X_pd0 - P_inv(P_sim[i,0])              # I_0
        for j in prange(T-1):
            Y_sim[i,j+1] = Y_S_sim[i,j+1] - α*EA_vals[Z_sim[i,j+1]]  # next-period Y_t
            X_sim[i,j+1] = np.exp(-δ)*I_sim[i,j] + Y_sim[i,j+1]      # next-period availability
            P_sim[i,j+1] = f_func(X_sim[i,j+1], Z_sim[i,j+1])        # next-period price
            X_pd = np.minimum(X_sim[i,j+1], x_disp[Z_sim[i,j+1]])    # next-period availability after disposal
            I_sim[i,j+1] = X_pd - P_inv(P_sim[i,j+1])                # next-period inventory

    return P_sim, X_sim, I_sim  # shape=(size,T)


def irf_R(cp,            # a class that stores model information
          x_star,        # the endogenous X_t grid points, shape=(K,M)
          p_star,        # equilibrium price at endogenous (X_t,Z_t) states, shape=(K,M)
          init_state,    # the initial state from which to calculate IRF
          shocks_irf,    # the shocks that constitute the impulse, an R_t index
          Y_seed=5678,   # random seed for simulating {Y_t^S}
          diff_type='perc',  # method of differentiation to compute IRF
          size=100_000,  # size of the sample over which to average
          T=16):         # forwarding periods
    """
    Calculating impulse responses to interest rate shocks.
    The IRFs are calculated conditional on X_t.
    """
    b = cp.b
    X0, Z0 = init_state[0], init_state[1]  # X_{-1}, Z_{-1}
    Y_S_sim = truncnorm.rvs(a=-cp.trunc_Y,
                            b=cp.trunc_Y,
                            size=(size,T),
                            loc=cp.μ_Y,
                            scale=cp.μ_Y*cp.σ_Y,
                            random_state=Y_seed)  # simulated {Y_t}, shape=(size,T)

    # The simulated {R_t} process with and without impulse shocks, shape=(size,T)
    mc_Z = MarkovChain(cp.Π)
    Z_sim_base = np.empty((size,T+1), dtype=int)
    Z_sim_irf = np.empty((size,T), dtype=int)
    for i in range(size):
        Z_sim_base[i,:] = mc_Z.simulate(T+1, init=Z0, random_state=i)
        Z_sim_irf[i,:] = mc_Z.simulate(T, init=shocks_irf, random_state=i)
    Z_samp_set = [Z_sim_base[:,1:], Z_sim_irf]
    X_samp_set, P_samp_set = [], []
    I_samp_set, I_pcg_samp_set = [], []
    EP_samp_set, EV_samp_set = [], []

    for k in range(len(Z_samp_set)):
        Z_sim = Z_samp_set[k]  # simulated {Z_t} index samples, shape=(size,T)

        # Simulate {P_t}, {X_t}, {I_t} processes, shape=(size,T) for each
        P_sim, X_sim, I_sim = sim_for_impulse(cp, x_star, p_star, X0, Z0,
                                              Z_sim, Y_S_sim, size, T)
        I_pcg_sim = I_sim / (X_sim-b)     # simulated {I_t/(X_t-b)}, shape=(size,T)
        X_samp_set.append(X_sim)          # append the simulated {X_t}
        P_samp_set.append(P_sim)          # append the simulated {P_t}
        I_samp_set.append(I_sim)          # append the simulated {I_t}
        I_pcg_samp_set.append(I_pcg_sim)  # append the simulated {I_t/(X_t-b)}
        EP_sim, EV_sim, _ = calc_expectations(cp,
                                              x_star,
                                              p_star,
                                              X_sim.flatten(),
                                              Z_sim.flatten())
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
                  α,  # effect of economic activity on commodity demand
                  L,  # length of time series samples to approx stationary dist
                  burn_in,  # No of discarded samples to approx stat dist
                  Z0,       # the initial exogenous state Z_{-1}
                  Z_irf,    # the exogenous state after the shock
                  X0_perc,  # percentile of the initial availability state X_{-1}
                  Ra_vals,  # the discrete {R_t^a} states
                  EA_vals,  # the discrete {EA_t} states
                  Π,        # the prob trans matrix of discrete {Z_t} process
                  Z_seq,    # the simulated {Z_t} index seq to approx stat dist
                  Y_S_sim,  # the simulated {Y_t^S} sequence to approx stat dist
                  gq_points,    # Gaussian quadrature sample points
                  gq_weights,   # Gaussian quadrature weights
                  n_expectations=0,  # number of elements on which to calculate expectations
                  Y_seed=5678,       # random seed of {Y_t^S} for computing IRFs
                  size=100_000,      # MC sample size for computing IRFs
                  periods=1,         # forecasting periods of IRF
                  diff_type='perc'):
    """
    IRFs under different (δ,λ) parameter setups.
    """
    cp = CommodityPricing(δ=δ, λ=λ, κ=0.0, μ_Y=1.0, σ_Y=0.05, α=α,
                          Ra_vals=Ra_vals, EA_vals=EA_vals, Π=Π, grid_size=100,
                          gq_points=gq_points, gq_weights=gq_weights)
    test_stability(cp, verbose=False)
    x_star, p_star = solve_model_time_iter(cp, T, verbose=False)
    # Compute the stationary distribution of (X_t,Z_t)
    X_sim, P_sim, Ra_sim, EA_sim, I_sim = compute_price_series(cp, x_star, p_star, Z_seq,
                                                               Y_S_sim, L=L, burn_in=burn_in)
    # Compute key moments
    ac_P = np.corrcoef(P_sim[1:], P_sim[:-1])[0,1]  # autocorrelation coef of P
    std_P = np.std(P_sim)                           # standard deviation of P
    cv_P = std_P / np.mean(P_sim)                   # coef of variation of P
    cor_RP = np.corrcoef(Ra_sim, P_sim)[0,1]        # correlation coef of Ra and P
    mean_EcovPR = float('nan')

    # Compute covariance prices-interest rates
    if n_expectations > 0:
        idx_ss = np.linspace(0, L - burn_in - 1, n_expectations).astype(int)
        Z_sim = Z_seq[burn_in+1:]
        idx_ss = np.linspace(0, L - burn_in - 1, n_expectations).astype(int)
        Z_sim = Z_seq[burn_in+1:]
        _, _, EcovPR_sim = calc_expectations(cp, x_star, p_star, X_sim[idx_ss], Z_sim[idx_ss])
        mean_EcovPR = np.mean(EcovPR_sim)

    moments = np.array([ac_P, std_P, cv_P, cor_RP, mean_EcovPR])

    # Compute impulse responses
    X0 = X_sim.mean() if X0_perc == None else np.percentile(X_sim, X0_perc)

    res = irf_R(cp, x_star, p_star, [X0, Z0], Z_irf, Y_seed=Y_seed,
                diff_type=diff_type, size=size, T=periods)
    impul_P, impul_I, impul_I_pcg, impul_X, impul_ES = res

    return impul_P, impul_I, impul_I_pcg, impul_X, impul_ES, moments
