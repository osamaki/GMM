#!/usr/bin/env python3

import numpy as np
from scipy.special import logsumexp, digamma
import sys
np.random.seed(10)

def init_params(K):
    N, D = X.shape
    # number of cluster
    #K = 2
    M = np.random.normal(0, 3, (K, D)) # (K, D)
    beta = np.ones(K) # (K)
    nu = np.array([D for _ in range(K)]) # (K)
    W = np.array([np.identity(D) for _ in range(K)]) # (K, D, D)
    alpha = np.ones(K) #(K)
    #  degree of freedom
    dof = (D + 1+ 1 + D*(D-1)/2 + 1) * K
    params = {}
    params["alpha"] = alpha
    params["beta"] = beta
    params["nu"] = nu
    params["m"] = M
    params["W"] = W
    return N, D, dof, params #dof, M, beta, nu, W, alpha
    
def calc_log_likelihood(X, pi, Mu, Lambda):
    # (10000, K)
    exponents = -0.5 * np.array([[ (x_n - mu_k).T @ Lambda_k @ (x_n - mu_k) for mu_k, Lambda_k in zip(Mu, Lambda)] for x_n in X])
    
    # (K)
    weight = np.copy(pi)
    for k, Lambda_k in enumerate(Lambda):
        try:
            det = np.linalg.det(Lambda_k)
        except:
            dl = 1e-7
            while dl < 1:
                l = 1 + dl
                try:
                    # regularization
                    det = np.linalg.det(Lambda_k + np.identity(Lambda_k.shape[0]) * l)
                    break
                except:
                    dl *= 10
        weight[k] *= np.sqrt(det)
    logsum_Likelihood = logsumexp(exponents, b=weight, axis=1)
    log_likelihood = -X.shape[1] / 2 * np.log(2*np.pi) * X.shape[0] + np.sum(logsum_Likelihood)
      
    return log_likelihood
    
def variational_bayes(X, params, max_iter=1000, convergence=1e-2):
    alpha = params["alpha"]
    beta = params["beta"]
    M = params["m"]
    nu = params["nu"]
    W = params["W"]

    prev_log_likelihood = 0
    for i in range(max_iter):
        # E step
        E_Lambda = np.array([nu_k * W_k for nu_k, W_k in zip(nu, W)]) # (K, D, D)
        E_logdet_Lambda = np.array([np.sum(digamma([(nu_k + 1 - (d + 1))*0.5 for d in range(X.shape[1])])) \
                                    + X.shape[1] * np.log(2) + np.log(np.linalg.det(W_k)) for nu_k, W_k in zip(nu, W)]) #(K)
        E_Lambda_Mu = np.array([nu_k * W_k @ M_k for nu_k, W_k, M_k in zip(nu, W, M)]) # (K, D)
        E_MuT_Lambda_Mu = np.array([nu_k * M_k @ W_k @ M_k for nu_k, M_k, W_k in zip(nu, M, W)]) # (K)
        sum_alpha = np.sum(alpha)
        E_log_pi = np.array([digamma(alpha_k) - digamma(sum_alpha) for alpha_k in alpha]) # (K)
        
        # (10000, K)
        exponents = np.array([[-0.5*x_n @ E_Lambda_k @ x_n + x_n @ E_Lambda_Mu_k - 0.5*E_MuT_Lambda_Mu_k \
                               + 0.5*E_logdet_Lambda_k + E_log_pi_k \
                                      for E_Lambda_k, E_logdet_Lambda_k, E_Lambda_Mu_k, E_MuT_Lambda_Mu_k, E_log_pi_k \
                                      in zip(E_Lambda, E_logdet_Lambda, E_Lambda_Mu, E_MuT_Lambda_Mu, E_log_pi)] for x_n in X])
        # (10000)
        log_normalization = logsumexp(exponents, axis=1)
        # (10000, K)
        log_Gamma = (exponents.T - log_normalization).T
        Gamma = np.exp(log_Gamma)
        
        pi = alpha / sum_alpha
        log_likelihood = calc_log_likelihood(X, pi, M, E_Lambda)
        print(f"\titer:{i}   log likelihood:{log_likelihood}")

        
        # M step
        # (K)
        S_1 = np.array([np.sum(Gamma[:,k]) for k in range(K)])
        # (K)
        S_x = np.sum([[gamma_k * x_n for gamma_k in gamma] for x_n, gamma in zip(X, Gamma)], axis=0)
        # (10000, 10000, 10000)
        x_xT = np.array([x_n[:, np.newaxis] @ x_n[np.newaxis, :] for x_n in X])
        # (K, 10000, 10000)
        S_xx = np.sum([[gamma_k * xn_xnT for gamma_k in gamma] for xn_xnT, gamma in zip(x_xT, Gamma)], axis=0)

        new_beta = S_1 + beta
        new_M = np.array([(S_x_k + beta_k * M_k)/new_beta_k for S_x_k, beta_k, new_beta_k, M_k in zip(S_x, beta, new_beta, M)])
        new_W_inv = np.array([S_xx_k + beta_k * M_k[:, np.newaxis] @ M_k[np.newaxis, :] \
                    - new_beta_k * new_M_k[:, np.newaxis] @ new_M_k[np.newaxis, :] + np.linalg.inv(W_k) \
                    for S_xx_k, beta_k, new_beta_k, M_k, new_M_k, W_k in zip(S_xx, beta, new_beta, M, new_M, W)])
        beta = new_beta
        M = new_M
        for i, new_W_inv_k in enumerate(new_W_inv):
            try:
                W[i, :, :] = np.linalg.inv(new_W_inv_k)
            except:
                W[i, :, :] = np.linalg.pinv(new_W_inv_k)
        nu = S_1 + nu
        alpha = S_1 + alpha
        
        if abs(prev_log_likelihood - log_likelihood) < convergence:
            break
        prev_log_likelihood = log_likelihood
    ret_params = {}
    ret_params["alpha"] = alpha
    ret_params["beta"] = beta
    ret_params["nu"] = nu
    ret_params["m"] = M
    ret_params["expectation_of_pi"] = pi
    ret_params["expectation_of_Lambda"] = E_Lambda
    ret_params["expectation_of_mu"] = M
    ret_params["W"] = W
    return log_likelihood, Gamma, ret_params

K = 7
max_iter = 500
convergence = 1e-2   
if __name__ == "__main__":
    args = sys.argv
    try:
        in_file = args[1]
    except:
        in_file = "x.csv"
    try:
        z_file = args[2]
    except:
        z_file = "z.csv"
    try:
        param_file = args[3]
    except:
        param_file = "params.dat"
        
    X = np.loadtxt(in_file, delimiter=",")
    best_n_cluster = -1
    best_bic = float('inf')
    best_gamma = None
    best_params = None
    n_try = 5
    print("#cluster:", K)
    for i in range(n_try):
        N, D, dof, params = init_params(K)
        log_likelihood, Gamma, params = variational_bayes(X, params, max_iter, convergence)
        if not np.isnan(log_likelihood):
            break
        if i == n_try-1:
            print("failed to calculate.")
        else:
            print("failed to calculate. retry.")            
    bic = float(inf) if np.isnan(log_likelihood) else -2 * log_likelihood + dof * np.log(N)
    print("\tBIC:", bic, "\n")
    if bic < best_bic:
        best_n_cluster = K
        best_bic = bic
        best_gamma = Gamma
        best_params = params

    np.savetxt(z_file, Gamma, delimiter=",")
    print("saved posterior probabilities of z_n to", z_file)

    with open(param_file, "w") as fout:
        for k, v in params.items():
            fout.write(k+"\n")
            if len(v.shape) == 3:
                for vv in v:
                    np.savetxt(fout, vv)
                    fout.write("\n")
                fout.write("\n\n")
            else:
                np.savetxt(fout, v)
                fout.write("\n\n")
    print("saved parameters to", param_file)

