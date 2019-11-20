import numpy as np

def gen_synthetic_multinomial(n0,n1,q,r,j):
    # n0,n1 is the number of samples from class 0, 1 resp.
    # q,r is are the theta^+/-
    
    X_train = []
    y_train = []
    
    X_train.extend(np.random.multinomial(j, q, size=n0))
    y_train.extend([1]*n0)
    
    X_train.extend(np.random.multinomial(j, r, size=n1))
    y_train.extend([-1]*n1)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    return X_train, y_train

def gen_synthetic_normal(n,d,sigma):
    
    X = np.random.normal(0, 1, n*d).reshape(n, d)
    w = np.random.normal(0, 1, d)
    y = X@w + np.random.normal(0, sigma, n)
    
    return X, np.array([y]).T, np.array([w]).T