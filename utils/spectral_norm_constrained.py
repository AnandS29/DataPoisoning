"""
This file contains helper functions to solve the spectral norm constrained
data poisoning problem.
"""
import numpy as np
import scipy

def solve_spectral_norm_constrained(X_ogi, y_ogi, eps):
    #The derivation for this method is detailed in the accompanying paper.
    U, Sigma, V = np.linalg.svd(X_ogi)
    d = U.T@y
    success, r_star = solve_line_search_problem(d, Sigma, eps)
    if success == 0:
        return np.zeros(X_ogi.shape)
    elif success == 1:
        return X_ogi
    else :
        lambda_star = find_lambda_from_r(r, sigma, d, Sigma)
        mu_star = lambda_star * r_star
        c_star = (lambda_star +  mu_star*Sigma@Sigma)@d
        u_star = U@c_star
        P = find_projection_map(u_star)
        X_poisoned = (X_ogi.T @ P).T
        return X_poisoned

def solve_line_search_problem(d, Sigma, eps):
    success = 0
    #Solves the specific line search problem as detailed in the paper

    #In this case, X can be made to 0
    if eps > Sigma:
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
            summation += num./deno
        return summation

    roots = scipy.optimize.root_scalar(search_func, maxiter = 10000)
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
