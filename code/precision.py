import numpy as np
import pandas as pd
from pricing import build_dynamics, gauss_quadra, T, CommodityPricing, \
    test_stability, solve_model_time_iter, compute_price_series, \
    compute_euler_eq_error
from quantecon import MarkovChain
from scipy.stats import truncnorm


def compute_lee(Ns, Ks, δs, λs):
    """
    Compute Euler equation error (in log) for given grid size and parameters
    """
    L, burn_in = 20_000, 1_000

    lEE_results = np.zeros((2, len(Ns), len(Ks), len(δs), len(λs)))

    # Gaussian quadrature sample points and weights
    gq_points, gq_weights = gauss_quadra()

    for i, N in enumerate(Ns):
        for j, K in enumerate(Ks):
            for k, δ in enumerate(δs):
                for l, λ in enumerate(λs):
                    # State points and prob trans matrix of {R_t^a}
                    Ra_vals, Π = build_dynamics(size=N)

                    # Solve the stochastic interest rate storage model
                    cp = CommodityPricing(δ=δ, λ=λ, κ=0., σ_Y=0.05,
                                          Ra_vals=Ra_vals, Π=Π,
                                          gq_points=gq_points,
                                          gq_weights=gq_weights, grid_size=K)
                    test_stability(cp, verbose=False)
                    x_star, p_star = solve_model_time_iter(cp, T, tol=1e-4,
                                                           verbose=False)

                    # Simulate {R_t} indices
                    mc_R = MarkovChain(Π)
                    R_seq = mc_R.simulate(L + 1, random_state=1234)

                    # Simulate {Y_t} process
                    Y_sim = truncnorm.rvs(a=-cp.trunc_Y,
                                          b=cp.trunc_Y,
                                          size=L+1,
                                          loc=cp.μ_Y,
                                          scale=cp.μ_Y*cp.σ_Y,
                                          random_state=1)

                    X_sim, _, _, _ = compute_price_series(cp, x_star, p_star,
                                                          R_seq, Y_sim, L=L,
                                                          burn_in=burn_in)
                    R_seq = R_seq[burn_in+1:]
                    Y_sim = Y_sim[burn_in+1:]

                    EE = compute_euler_eq_error(cp, x_star, p_star, X_sim,
                                                R_seq)
                    lEE = np.log10(np.abs(EE))

                    lEE_results[:, i, j, k, l] = [np.max(lEE),
                                                  np.quantile(lEE, 0.95)]
                    lEE_results = lEE_results.round(2)

    return lEE_results


lEE_NK = compute_lee(Ns=[7, 51, 101], Ks=[100, 200, 1_000],
                     δs=[0.02], λs=[0.06]).reshape(2, 9, order='F')
lEE_NK = pd.DataFrame(lEE_NK, index=['max', '95\%'])

lEE_NK.to_csv(path_or_buf='../output/tables/lee_nk.tex',
              header=False, lineterminator='\\\\\n', sep="&", quotechar=' ')

lEE_δλ = compute_lee(Ns=[51], Ks=[100],
                     δs=[0.01, 0.02, 0.05],
                     λs=[0.03, 0.06, 0.15]).reshape(2, 9, order='F')
lEE_δλ = pd.DataFrame(lEE_δλ, index=['max', '95\%'])

lEE_δλ.to_csv(path_or_buf='../output/tables/lee_delta_lambda.tex',
              header=False, lineterminator='\\\\\n', sep="&", quotechar=' ')
