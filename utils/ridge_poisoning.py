import numpy as np
import cvxpy as cp
from interval_bounds import *


l = lambda X: np.linalg.norm(X@np.linalg.pinv(X)@y-y)**2

def solve_nominal(X, y, lamb=10):
    n, m = X.shape
    w = cp.Variable((m,1))
    nominal_obj = cp.Minimize(cp.square(cp.norm(X@w - y, 2)) + lamb*cp.square(cp.norm(w,2)))
    constraints = []
    prob = cp.Problem(nominal_obj, constraints)
    prob.solve()
    return prob.value, w.value

def solve_robust(X,y, rho, lambda_):
    # data, labels = X,y
    n,d = X.shape
    w_rob = cp.Variable((d,1))

    abs_resids = cp.abs(X@w_rob - y)
    robper = rho*cp.norm(w_rob, p=2)*np.ones((n,1))
    ridge_pen = lambda_*cp.norm(w_rob, p=2)**2
    obj = cp.norm(abs_resids + robper, p=2)**2 + ridge_pen

    prob = cp.Problem(cp.Minimize(obj))
    prob.solve()
    return prob.value, w_rob.value

def solve_poisoned_relaxed(X_nominal, y, lamb=10, rho=10, mu=10):
    n, m = X_nominal.shape[0], X_nominal.shape[1]
    X = cp.Variable((n,m))
    M = cp.Variable((m+1,m+1), PSD=True)
    N = cp.Variable((m+n,m+n), PSD=True)
    t = cp.Variable(1)
    U = cp.Variable((m,m), PSD=True)
    poisoned_obj = cp.Minimize(t - cp.square(cp.norm(y)) + mu*cp.trace(U))
    constraints = [
        M[0:m,0:m] == U - lamb*np.eye(m),
        M[m,m] == t,
        M[m:m+1,0:m] == y.T@X,
        M[0:m,m:m+1] == X.T@y,
        N[0:m,0:m] == U,
        N[m:,m:] == np.eye(n),
        N[m:,0:m] == X,
        N[0:m,m:] == X.T,
        
    ]
    for i in range(n):
        constraints.append(cp.norm(X[i]-X_nominal[i],2) <= rho)
    prob = cp.Problem(poisoned_obj, constraints)
    prob.solve()
    return -1*prob.value, X.value, U.value

def solve_poisoned_relaxed_interval_bounded(X_nominal, y, lamb=10, rho=10, mu=10):
    U_lower, U_upper = find_interval_bounds_infty_uncertainty(X_nominal, rho)
    n, m = X_nominal.shape
    X = cp.Variable((n,m))
    M = cp.Variable((m+1,m+1), PSD=True)
    N = cp.Variable((m+n,m+n), PSD=True)
    t = cp.Variable(1)
    U = cp.Variable((m,m), PSD=True)
    poisoned_obj = cp.Minimize(t - cp.square(cp.norm(y)))
    constraints = [
        M[0:m,0:m] == U - lamb*np.eye(m),
        M[m,m] == t,
        M[m:m+1,0:m] == y.T@X,
        M[0:m,m:m+1] == X.T@y,
        N[0:m,0:m] == U,
        N[m:,m:] == np.eye(n),
        N[m:,0:m] == X,
        N[0:m,m:] == X.T,
        U <= U_upper, 
        U >= U_lower
    ]
    for i in range(n):
        constraints.append(cp.norm(X[i]-X_nominal[i],2) <= rho)
    prob = cp.Problem(poisoned_obj, constraints)
    prob.solve()
    return -1*prob.value, X.value, U.value

def solve_poisoned_bounded(X_nominal, y, lamb=10, rho=10, mu=0, upper_bound=10):
    X = cp.Variable((n,m))
    M = cp.Variable((m+1,m+1), PSD=True)
    N = cp.Variable((m+n,m+n), PSD=True)
    t = cp.Variable(1)
    U = cp.Variable((m,m), PSD=True)
    poisoned_obj = cp.Minimize(t - cp.square(cp.norm(y)) + mu*cp.trace(U))
    constraints = [
        M[0:m,0:m] == U - lamb*np.eye(m),
        M[m,m] == t,
        M[m:m+1,0:m] == y.T@X,
        M[0:m,m:m+1] == X.T@y,
        N[0:m,0:m] == U,
        N[m:,m:] == np.eye(n),
        N[m:,0:m] == X,
        N[0:m,m:] == X.T,
        t - cp.square(cp.norm(y)) + mu*cp.trace(U) <= upper_bound
    ]
    for i in range(n):
        constraints.append(cp.norm(X[i]-X_nominal[i],2) <= rho)
    prob = cp.Problem(poisoned_obj, constraints)
    prob.solve()
    return -1*prob.value, X.value, U.value
