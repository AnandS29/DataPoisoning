#Thiss file contains helper functions for finding intervals for bounding boxes
import numpy as np

def find_interval_bounds_infty_uncertainty(X, rho):
    n, d = X.shape
    U_upper, U_lower = np.zeros((d,d)), np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            x_i, x_j = X[:, i], X[:, j]
            upper_u_ij, lower_u_ij = find_interval_bound_for_uij(x_i, x_j, rho)
            U_upper[i,j], U_lower[i,j]  = upper_u_ij, lower_u_ij
    return U_upper, U_lower

def find_interval_bound_for_uij(x_i, x_j, rho):
    d = len(x_i)
    max_list = []
    min_list = []
    for k in range(d):
        x_ik, x_jk = x_i[k], x_j[k]
        min_ij, max_ij = solve_one_dim_opt_problem(x_ik, x_jk, rho)
        max_list.append(max_ij)
        min_list.append(min_ij)
    return sum(max_list), sum(min_list)

def solve_one_dim_opt_problem(x_ik, x_jk, rho):
    objective_fxn = lambda a,b: x_ik*x_jk + x_ik*a + x_jk*b + a*b
    d_ik_options, d_jk_options = [-rho, rho], [-rho, rho]
    values = []
    for a in d_ik_options:
        for b in d_jk_options:
            value = objective_fxn(a, b)
            values.append(value)

    return np.min(values), np.max(values)
