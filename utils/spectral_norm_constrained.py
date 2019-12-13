"""
This file contains helper functions to solve the spectral norm constrained
data poisoning problem.
"""
import numpy as np
import scipy
from scipy import optimize
import ipdb

def solve_spectral_norm_constrained(X_ogi, y_ogi, eps):
    #The derivation for this method is detailed in the accompanying paper.
    U, Sigma, V = np.linalg.svd(X_ogi)

    #Alter sigma a little bit
    k = len(Sigma)
    n = len(U)
    diff_length = n - k
    attach = np.zeros(diff_length)
    Sigma = np.hstack((Sigma, attach))

    d_vec = U.T@y_ogi
    success, r_star = solve_line_search_problem(d_vec, Sigma, eps)
    if success == 0:
        print("Can make zero matrix")
        return np.zeros(X_ogi.shape), r_star
    elif success == 1:
        print("Impossible to do any poisoning")
        return X_ogi, r_star
    else :
        r_star = r_star.root
        lambda_star = find_lambda_from_r(r_star, Sigma, d_vec)
        mu_star = lambda_star * r_star
        c_star = lambda_star*d_vec
        Sigma = np.diag(Sigma)
        c_star += mu_star*Sigma@Sigma@d_vec
        u_star = U@c_star
        P = find_projection_map(u_star)
        X_poisoned = (X_ogi.T @ P).T
        return X_poisoned, r_star

def solve_line_search_problem(d, Sigma, eps):
    success = 0
    #Solves the specific line search problem as detailed in the paper

    #In this case, X can be made to 0
    if eps > Sigma[0]:
        return success, None

    #In this case poisoning is completely ineffective
    if eps < Sigma[-1]:
        return 1, None

    #Define nested function whose roots we wish to find acc to optimality conds
    def search_func(r):
        summation = 0
        for i in range(len(d)):
            num = d[i]**2 * (Sigma[i]**2 - eps**2)
            deno = (1 + r*Sigma[i]**2)**2
            summation += num/deno
        return summation

    roots = optimize.root_scalar(search_func, x0 = 1, x1 = 50, maxiter = 10000)
    success = 2
    return success, roots

def find_projection_map(v):
    #Finds a map to project onto the orthogonal complement of vector v
    d = len(v)
    #TODO Need to ensure that there is no datatype issue here.
    I = np.eye(d)
    V_reg = I - (v@v.T)/np.linalg.norm(v)
    V_orthonormal = scipy.linalg.orth(V_reg.T)
    return V_orthonormal@V_orthonormal.T

def find_lambda_from_r(r, Sigma, d):
    summation = 0
    for i in range(len(d)):
        num = d[i]**2
        deno = (1 + r*Sigma[i]**2)**2
        summation += num/deno
    return np.sqrt(summation)

