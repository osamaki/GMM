#!/usr/bin/env python3

import numpy as np
from scipy.special import logsumexp
import sys
np.random.seed(10)

def init_params(K):
    N, D = X.shape
    # number of cluster
    #K = 2
    Mu = np.random.normal(0, 3, (K, D))
    Lambda = np.array([np.identity(D) for _ in range(K)])
    pi = np.array([1. / K for _ in range(K)])
    #  degree of freedom
    dof = (D + D*(D-1)/2) * K + K-1
    params = {}
    params["pi"] = pi
    params["mu"] = Mu
    params["Lambda"] = Lambda
    return N, D, dof, params
    
def EM_algorithm(X, params, max_iter=500, convergence=1e-2):
    Mu = params["mu"]
    Lambda = params["Lambda"]
    pi = params["pi"]
    prev_log_likelihood = 0
    for i in range(max_iter):
        # E step
        exponents = -0.5 * np.array([[ (x_n - mu_k).T @ Lambda_k @ (x_n - mu_k) for mu_k, Lambda_k in zip(Mu, Lambda)] for x_n in X])
        weight = np.copy(pi)
        for k, Lambda_k in enumerate(Lambda):
            try:
                tmp = np.sqrt(np.linalg.det(Lambda_k))
            except:
                dl = 1e-7
                while dl < 1:
                    l = 1 + dl
                    try:
                        # regularization
                        tmp = np.sqrt(np.linalg.det(Lambda_k + np.identity(Lambda_k.shape[0]) * l))
                        break
                    except:
                        dl *= 10
            weight[k] *= tmp
        logsum_Likelihood = logsumexp(exponents, b=weight, axis=1)
        
        log_likelihood = -X.shape[1] / 2 * np.log(2*np.pi) * X.shape[0] + np.sum(logsum_Likelihood)
        print(f"\titer:{i}   log likelihood:{log_likelihood}")
        if np.isnan(log_likelihood):
            return log_likelihood, None, None
        log_Likelihood = np.log(weight) + exponents  # (10000, K)
        log_Gamma = (log_Likelihood.T - logsum_Likelihood).T
        Gamma = np.exp(log_Gamma)
        
        # M step
        S_1 = np.array([np.sum(Gamma[:,k]) for k in range(K)])
        S_x = np.sum([[gamma_k * x_n for gamma_k in gamma] for x_n, gamma in zip(X, Gamma)], axis=0)
        x_xT = np.array([x_n[:, np.newaxis] @ x_n[np.newaxis, :] for x_n in X])
        S_xx = np.sum([[gamma_k * xn_xnT for gamma_k in gamma] for xn_xnT, gamma in zip(x_xT, Gamma)], axis=0)

        pi = S_1 / N
        Mu = (S_x.T / S_1).T
        mu_muT = np.array([mu_k[:, np.newaxis] @ mu_k[np.newaxis, :] for mu_k in Mu])
        Lambda_inv = (S_xx.T / S_1).T - mu_muT
        for i, Lambda_inv_k in enumerate(Lambda_inv):
            try:
                Lambda[i, :, :] = np.linalg.inv(Lambda_inv_k)
            except:
                Lambda[i, :, :] = np.linalg.pinv(Lambda_inv_k)

        if abs(prev_log_likelihood - log_likelihood) < convergence:
            break
        prev_log_likelihood = log_likelihood
    ret_params = {}
    ret_params["pi"] = pi
    ret_params["mu"] = Mu
    ret_params["Lambda"] = Lambda
    return log_likelihood, Gamma, ret_params

max_iter = 500
convergence = 1e-2
min_K = 1
max_K = 8
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
    for K in range(min_K, max_K+1):
        print("#cluster:", K)
        for i in range(n_try):
            N, D, dof, params = init_params(K)
            log_likelihood, Gamma, params = EM_algorithm(X, params, max_iter, convergence)
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
    
    print("Best #clster:", best_n_cluster)
    print("\tBIC:", best_bic)

    np.savetxt(z_file, best_gamma, delimiter=",")
    print("saved posterior probabilities of z_n in", z_file)

    with open(param_file, "w") as fout:
        for k, v in best_params.items():
            fout.write(k+"\n")
            if k=="Lambda":
                for vv in v:
                    np.savetxt(fout, vv)
                    fout.write("\n")
                fout.write("\n\n")
            else:
                np.savetxt(fout, v)
                fout.write("\n\n")
    print("saved parameters in", param_file)
