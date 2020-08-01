#!/usr/bin/env python3

import numpy as np
from scipy.stats import wishart
from scipy.special import logsumexp, digamma
import sys
np.random.seed(10)

def init_params(K):
    N, D = X.shape
    # number of cluster
    #K = 2
    Mu = np.random.normal(0, 3, (K, D))
    Lambda = np.array([np.identity(D) for _ in range(K)])
    pi = np.array([1. / K for _ in range(K)])
    M = np.random.normal(0, 3, (K, D)) # (K, D)
    beta = np.ones(K) # (K)
    nu = np.array([D for _ in range(K)]) # (K)
    W = np.array([np.identity(D) for _ in range(K)]) # (K, D, D)
    alpha = np.ones(K) #(K)

    params = {}
    params["mu"] = Mu
    params["Lambda"] = Lambda
    params["pi"] = pi
    params["m"] = M
    params["beta"] = beta
    params["nu"] = nu
    params["W"] = W
    params["alpha"] = alpha
    return N, D, params
    
def calc_log_likelihood(X, pi, Mu, Lambda):
    # (10000, K)
    exponents = -0.5 * np.array([[ (x_n - mu_k).T @ Lambda_k @ (x_n - mu_k) for mu_k, Lambda_k in zip(Mu, Lambda)] for x_n in X])
    
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
    
def gibbs_sampling(X, params, max_iter=1000, convergence=1e-3):
    Mu = params["mu"]
    Lambda = params["Lambda"]
    pi = params["pi"]
    M = params["m"]
    beta = params["beta"]
    nu = params["nu"]
    W = params["W"]
    alpha = params["alpha"] 
    prev_log_likelihood = 0
    for i in range(max_iter):
        # E step
        # (10000, K)
        exponents = np.array([[-0.5 * (x_n - mu_k).T @ Lambda_k @ (x_n - mu_k) + 0.5 * np.log(np.linalg.det(Lambda_k)) + np.log(pi_k) \
                               for mu_k, Lambda_k, pi_k in zip(Mu, Lambda, pi)] for x_n in X])

        logsum_Likelihood = logsumexp(exponents, axis=1)
        log_Likelihood = exponents  # (10000, K)
        log_Gamma = (log_Likelihood.T - logsum_Likelihood).T
        Gamma = np.exp(log_Gamma)
        
        categorys = list(range(pi.shape[0])) # (K)
        identity = np.identity(pi.shape[0]) # (K)
        Z = np.array([identity[np.random.choice(categorys, p=gamma)] for gamma in Gamma])
        
        log_likelihood = calc_log_likelihood(X, pi, Mu, Lambda)
        print(f"\titer:{i}   log likelihood:{log_likelihood}")

        # M step
        S_1 = np.array([np.sum(Z[:,k]) for k in range(K)])
        S_x = np.sum([[Z[n,k] * X[n] for k in range(Z.shape[1])] for n in range(X.shape[0])], axis=0)
        x_xT = np.array([X[n, :, np.newaxis] @ X[n, np.newaxis, :] for n in range(X.shape[0])])
        S_xx = np.sum([[Z[n,k] * x_xT[n] for k in range(Z.shape[1])] for n in range(Z.shape[0])], axis=0)
        
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
        
        Lambda = np.array([wishart(df=nu_k, scale=W_k).rvs(1) for nu_k, W_k in zip(nu, W)])
        Mu = np.array([np.random.multivariate_normal(M[k], np.linalg.inv(beta[k]*Lambda[k]), 1)[0] for k in range(pi.shape[0])])
        pi = np.random.dirichlet(alpha, 1)[0]
        
        if abs(prev_log_likelihood - log_likelihood) < convergence:
            break
        prev_log_likelihood = log_likelihood
    ret_params = {}
    ret_params["mu"] = Mu
    ret_params["Lambda"] = Lambda
    ret_params["pi"] = pi
    ret_params["alpha"] = alpha
    ret_params["beta"] = beta
    ret_params["nu"] = nu
    ret_params["m"] = M
    ret_params["pi"] = pi
    ret_params["W"] = W
    return log_likelihood, Gamma, ret_params
    
K = 4
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
    n_try = 5
    print("#cluster:", K)
    for i in range(n_try):
        N, D, params = init_params(K)
        log_likelihood, Gamma, params = gibbs_sampling(X, params, max_iter, convergence)
        if not np.isnan(log_likelihood):
            break
        if i == n_try-1:
            print("failed to calculate.")
        else:
            print("failed to calculate. retry.")


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
