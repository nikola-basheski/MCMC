import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import time
import random
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import probplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from scipy.optimize import minimize

# ============================ #
#      SET SEED FOR REPRODUCIBILITY
# ============================ #
np.random.seed(65)
random.seed(65)

# ============================ #
#         DATA PREPROCESSING
# ============================ #
ticker = "AAPL"
data = yf.download(ticker, start="1980-01-01", end="2023-01-01")

# ============================ #
#    HESTON MODEL SIMULATION USING QE SCHEME
# ============================ #
def heston_model_sim_qe(S0, v0, mu, kappa, theta, sigma, rho, T, N, n_paths):
    if N < 1:  # Edge case: no steps to simulate
        return (np.array([[S0]*n_paths]), np.array([[v0]*n_paths]))
    
    dt = T / N
    S = np.zeros((N+1, n_paths))
    v = np.zeros((N+1, n_paths))
    S[0, :] = S0
    v[0, :] = max(v0, 1e-12)
    psi_c = 1.5  # threshold for QE scheme

    for t in range(1, N+1):
        vt = v[t-1, :]
        E = np.exp(-kappa * dt)
        m = theta + (vt - theta) * E
        s2 = (vt * sigma**2 * E / kappa) * (1 - E) + theta * sigma**2 / (2 * kappa) * (1 - E)**2
        s2 = np.where(s2 < 0, 0.0, s2)
        psi = s2 / (m**2 + 1e-12)
        psi = np.where(psi <= 1e-12, 1e-12, psi)
        
        v_next = np.zeros(n_paths)
        idx = psi <= psi_c
        if np.any(idx):
            tmp = 2.0 / psi[idx] - 1.0
            tmp_sqrt = 2.0 / psi[idx] * tmp
            tmp_sqrt = np.where(tmp_sqrt < 0, 0.0, tmp_sqrt)
            b2 = np.maximum(tmp + np.sqrt(tmp_sqrt), 0.0)
            a = m[idx] / (1 + b2 + 1e-12)
            Z = np.random.randn(np.sum(idx))
            vq = a * (np.sqrt(b2) + Z)**2
            v_next[idx] = np.maximum(vq, 0.0)
        
        idx2 = ~idx
        if np.any(idx2):
            p = (psi[idx2] - 1.0) / (psi[idx2] + 1.0)
            p = np.where(p < 0, 0.0, np.where(p > 1, 1.0, p))
            U = np.random.rand(np.sum(idx2))
            beta = (1.0 - p) / (m[idx2] + 1e-12)
            denom = np.maximum((1.0 - U), 1e-12)
            expo = -np.log((1.0 - p) / denom) / (beta + 1e-12)
            v_next[idx2] = np.where(U <= p, 0.0, expo)
        
        v_next = np.where(np.isnan(v_next), 1e-12, v_next)
        v_next = np.maximum(v_next, 1e-12)
        v[t, :] = v_next

        Z1 = np.random.randn(n_paths)
        S[t, :] = S[t-1, :] + mu * S[t-1, :] * dt + np.sqrt(v[t-1, :]) * S[t-1, :] * np.sqrt(dt) * Z1

    return S, v

# ============================ #
#   BAYESIAN MCMC: PRIOR & LIKELIHOOD
# ============================ #
def log_prior(params):
    mu, log_kappa, log_theta, log_sigma, atanh_rho = params
    kappa = np.exp(log_kappa)
    theta = np.exp(log_theta)
    sigma = np.exp(log_sigma)
    rho = np.tanh(atanh_rho)
    if kappa <= 0 or theta <= 0 or sigma <= 0 or abs(rho) >= 1:
        return -np.inf
    lp = -0.5 * (mu)**2
    lp += -0.5 * ((log_kappa + 20) / 0.005)**2
    lp += -0.5 * ((log_theta - np.log(0.005)) / 0.005)**2
    lp += -0.5 * ((20*log_sigma - np.log(0.0020)) / 0.005)**2
    lp += -0.5 * (atanh_rho / 0.4)**2
    return lp

def log_likelihood(params, returns, dv, v_prev, dt):
    mu = params[0]
    kappa = np.exp(params[1])
    theta = np.exp(params[2])
    sigma = np.exp(params[3])
    rho = np.tanh(params[4])
    
    log_lik = 0.0
    n = len(returns)
    for i in range(n):
        if v_prev[i] <= 0:
            return -np.inf
        m1 = (mu - 0.5 * v_prev[i]) * dt
        m2 = kappa * (theta - 0.05*v_prev[i]) * dt
        x1 = returns[i]
        x2 = dv[i]
        C11 = v_prev[i] * dt
        C22 = (sigma**2) * v_prev[i] * dt
        C12 = rho * sigma * v_prev[i] * dt
        det = C11 * C22 - C12**2
        if det <= 1e-12:
            return -np.inf
        inv11 = C22 / det
        inv12 = -C12 / det
        inv22 = C11 / det
        
        diff1 = x1 - m1
        diff2 = x2 - m2
        quad = diff1**2 * inv11 + 2 * diff1 * diff2 * inv12 + diff2**2 * inv22
        log_lik += -0.5 * (np.log(det) + quad + 2 * np.log(2*np.pi))
    return log_lik

def metropolis_heston(iterations, returns, dv, v_prev, dt):
    params = np.array([0.05, np.log(0.4), np.log(0.012), np.log(0.014), np.arctanh(-0.3)])
    proposals = np.array([0.003, 0.05, 0.05, 0.05, 0.015])
    samples = np.zeros((iterations, 5))
    accepted = 0
    current_log_post = log_prior(params) + log_likelihood(params, returns, dv, v_prev, dt)
    
    for i in range(iterations):
        new_params = params + proposals * np.random.randn(5)
        new_log_post = log_prior(new_params) + log_likelihood(new_params, returns, dv, v_prev, dt)
        
        if np.exp(new_log_post - current_log_post) > np.random.rand():
            params = new_params
            current_log_post = new_log_post
            accepted += 1
        
        samples[i] = params
        
        if i % 100 == 0:
            print(f"Iter {i}, Acceptance Rate: {accepted/(i+1):.2f}")
    return samples

# ============================ #
#   MAXIMUM LIKELIHOOD ESTIMATION
# ============================ #
def mle_estimation(returns, dv, v_prev, dt, init_params):
    def neg_log_like(params):
        ll = log_likelihood(params, returns, dv, v_prev, dt)
        return -ll if np.isfinite(ll) else 1e20
    
    result = minimize(neg_log_like, init_params, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'disp': True})
    if not result.success:
        print("MLE failed to converge")
    return result.x

# ============================ #
#      CROSS-VALIDATION & MODEL
# ============================ #
train_data = data.copy()
train_data['Log_Returns'] = np.log(train_data['Close']).diff()
train_data['Volatility'] = train_data['Log_Returns'].rolling(21).std() * np.sqrt(252)
train_data['v_t'] = (train_data['Volatility'] ** 2).shift(1)
train_data['dv_t'] = (train_data['Volatility'] ** 2) - train_data['v_t']
train_data.dropna(inplace=True)

n_splits = 10
tscv = TimeSeriesSplit(n_splits=n_splits)
dt = 1/252

# Storage for results
mcmc_params, mle_params = [], []
mcmc_times, mle_times = [], []
mcmc_rmse, mle_rmse = [], []
mcmc_mae, mle_mae = [], []
all_residuals_mcmc, all_residuals_mle = [], []
vol_std, return_std = [], []

for fold, (train_index, test_index) in enumerate(tscv.split(data)):
    print(f"\n=== Fold {fold+1}/{n_splits} ===")
    train_fold = train_data.iloc[train_index]
    test_fold = data.iloc[test_index].copy()
    
    returns_train = train_fold['Log_Returns'].values[1:]
    dv_train = train_fold['dv_t'].values[1:]
    v_prev_train = train_fold['v_t'].values[1:]
    
    # MCMC Estimation
    start_mcmc = time.time()
    samples = metropolis_heston(1000, returns_train, dv_train, v_prev_train, dt)
    burn_in = 500
    posterior_means = samples[burn_in:].mean(axis=0)
    mcmc_time = time.time() - start_mcmc
    mcmc_times.append(mcmc_time)
    
    # MLE Estimation
    start_mle = time.time()
    init_params = np.array([0.05, np.log(0.4), np.log(0.012), np.log(0.14), np.arctanh(-0.3)])
    mle_result = mle_estimation(returns_train, dv_train, v_prev_train, dt, init_params)
    mle_time = time.time() - start_mle
    mle_times.append(mle_time)
    
    # Parameter extraction
    params_mcmc = [
        posterior_means[0],
        np.exp(posterior_means[1]),
        np.exp(posterior_means[2]),
        np.exp(posterior_means[3]),
        np.tanh(posterior_means[4])
    ]
    params_mle = [
        mle_result[0],
        np.exp(mle_result[1]),
        np.exp(mle_result[2]),
        np.exp(mle_result[3]),
        np.tanh(mle_result[4])
    ]
    mcmc_params.append(params_mcmc)
    mle_params.append(params_mle)
    
    # Simulation and evaluation
    test_fold['Log_Returns'] = np.log(test_fold['Close']).diff()
    test_fold['Volatility'] = test_fold['Log_Returns'].rolling(21).std() * np.sqrt(252)
    test_fold['Volatility'] = test_fold['Volatility'].bfill()
    market_vol_test = test_fold['Volatility'].values
    N_sim = len(market_vol_test)
    T_sim = N_sim / 252.0
    
    # MCMC Simulation
    S0_test = train_fold['Close'].iloc[-1]
    last_train_vol = train_fold['Volatility'].iloc[-1]
    v0_test = max(last_train_vol**2, 1e-12)
    S_sim_mcmc, v_sim_mcmc = heston_model_sim_qe(S0_test, v0_test, *params_mcmc, T_sim, N_sim, 100)
    sim_vol_mcmc = np.sqrt(v_sim_mcmc.mean(axis=1)[1:])
    sim_vol_mcmc = np.nan_to_num(sim_vol_mcmc, nan=0.0)
    
    # MLE Simulation
    S_sim_mle, v_sim_mle = heston_model_sim_qe(S0_test, v0_test, *params_mle, T_sim, N_sim, 100)
    sim_vol_mle = np.sqrt(v_sim_mle.mean(axis=1)[1:])
    sim_vol_mle = np.nan_to_num(sim_vol_mle, nan=0.0)
    
    # Calculate metrics
    min_length = min(len(sim_vol_mcmc), len(market_vol_test))
    mkt_slice = market_vol_test[:min_length]
    
    # MCMC metrics
    mcmc_slice = sim_vol_mcmc[:min_length]
    rmse_mcmc = np.sqrt(np.mean((mkt_slice - mcmc_slice)**2))
    mae_mcmc = np.mean(np.abs(mkt_slice - mcmc_slice))
    mcmc_rmse.append(rmse_mcmc)
    mcmc_mae.append(mae_mcmc)
    vol_std.append(np.std(test_fold['Volatility']))
    return_std.append(np.std(test_fold['Close']))
    all_residuals_mcmc.extend(mkt_slice - mcmc_slice)
    
    # MLE metrics
    mle_slice = sim_vol_mle[:min_length]
    rmse_mle = np.sqrt(np.mean((mkt_slice - mle_slice)**2))
    mae_mle = np.mean(np.abs(mkt_slice - mle_slice))
    mle_rmse.append(rmse_mle)
    mle_mae.append(mae_mle)
    all_residuals_mle.extend(mkt_slice - mle_slice)
    
    # Plotting
    plt.figure(figsize=(12, 4))
    plt.plot(test_fold.index[:min_length], mkt_slice, 'k--', label="Market Vol")
    plt.plot(test_fold.index[:min_length], mcmc_slice, 'b-', label="MCMC Vol")
    plt.plot(test_fold.index[:min_length], mle_slice, 'r-', label="MLE Vol")
    plt.title(f"Fold {fold+1}: Market vs. Estimated Volatilities")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.legend()
    plt.show()

# ============================ #
#      COMPARISON ANALYSIS
# ============================ #
# Parameter comparison
param_names = ['μ', 'κ', 'θ', 'σ', 'ρ']
mcmc_params_arr = np.array(mcmc_params)
mle_params_arr = np.array(mle_params)

plt.figure(figsize=(15, 8))
for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.boxplot([mcmc_params_arr[:,i], mle_params_arr[:,i]], labels=['MCMC', 'MLE'])
    plt.title(param_names[i])
plt.suptitle("Parameter Estimates Comparison")
plt.tight_layout()
plt.show()

# Performance metrics comparison
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(mcmc_rmse, 'b-o', label='MCMC RMSE')
plt.plot(mle_rmse, 'r-s', label='MLE RMSE')
plt.title("RMSE Comparison Across Folds")
plt.xlabel("Fold Number")
plt.ylabel("RMSE")
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(['MCMC', 'MLE'], [np.mean(mcmc_times), np.mean(mle_times)])
plt.title("Average Computation Time")
plt.ylabel("Time (seconds)")
plt.show()

# Residual diagnostics
plt.figure(figsize=(15, 6))
plt.subplot(2, 2, 1)
plt.plot(all_residuals_mcmc, alpha=0.6, label='MCMC')
plt.plot(all_residuals_mle, alpha=0.6, label='MLE')
plt.title("Residuals Over Time")
plt.legend()

plt.subplot(2, 2, 2)
plt.hist(all_residuals_mcmc, bins=30, alpha=0.6, label='MCMC')
plt.hist(all_residuals_mle, bins=30, alpha=0.6, label='MLE')
plt.title("Residual Distribution")
plt.legend()

plt.subplot(2, 2, 3)
probplot(all_residuals_mcmc, plot=plt, fit=True)
plt.title("MCMC QQ-Plot")

plt.subplot(2, 2, 4)
probplot(all_residuals_mle, plot=plt, fit=True)
plt.title("MLE QQ-Plot")
plt.tight_layout()
plt.show()

print("\n=== Final Metrics ===")
print(f" Average STD: {np.nanmean(vol_std):.4f}")
print(f"MCMC Average RMSE: {np.nanmean(mcmc_rmse):.4f}")
print(f"MLE Average RMSE: {np.nanmean(mle_rmse):.4f}")
print(f"\nMCMC Average Time: {np.mean(mcmc_times):.1f}s")
print(f"MLE Average Time: {np.mean(mle_times):.1f}s")
