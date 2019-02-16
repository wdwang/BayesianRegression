import numpy as np
import matplotlib
import math
import scipy.stats as ss
import utils

import matplotlib.pyplot as plt


def get_EI(mu, var, f, zeta=0):
        """
        This function computes the expected improvement (EI) according to "A Tutorial on Bayesian Optimization of
        Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning", by
        Eric Brochu, Vlad M. Cora, Nando de Freitas, URL: https://arxiv.org/abs/1012.2599.

        Parameters:
        mu: Mean
        var: Variance
        f: Observed values so far
        zeta: Exploitation vs exploration parameter, See reference for more information

        Return:
        EI at the submitted values
        """

        delta = np.maximum(0, (np.subtract(mu,f) - zeta))
        sigma = np.sqrt(var)
        q = delta / sigma

        # Get cdf, pdf from scipy and compute EI
        cdf = ss.norm.cdf(q)
        pdf = ss.norm.pdf(q)
        ei = delta * cdf + sigma * pdf

        return ei


def get_ucb(mu, var, beta_t):
    '''
    Implementation of the UCB acq. function according to
    http://www-stat.wharton.upenn.edu/~skakade/papers/ml/bandit_GP_icml.pdf.

    Parameters:
    mu: Mean
    var: Variance
    beta_t: Scalar to balance between exploration and exploitation

    Return:
    UCB at the submitted values
    '''
    return mu + np.sqrt(beta_t) * np.sqrt(var)


def get_MI(X, Y_true, locs, k, var_A, divide):
    '''
    Calculate the n-dimensional MI according to equation 6 in Krause, Singh, Guestrin - 2008 - Near-Optimal Sensor Placements in
    Gaussian Processes Theory, Efficient Algorithms and Empirical Studies.

    Parameters:
    X:       Set of all possible inputs
    Y_true:  Ground truth values
    locs:    Set of already placed sensors/measurement points
    k:       Kernel that is currently used in context (GPFlow kernel!)
    var_A:   Variance on input space based on posterior after observed values
    divide:  Whether to use the approximation or the explicit formula from the paper

    Return:
    mi:      Mutual Information for every possible location in X
    X:       X axis without already chosen locations
    '''

    # Find indices of already chosen locations
    ind = utils.findIndices(X,locs)
    # Remove chosen locations from from input space
    X_tmp = utils.removefromArray(X,ind)
    Y_tmp = utils.removefromArray(Y_true.reshape(-1, 1),ind)
    # Predict input space
    mi = []
    var_Abar = []
    h_yAbar = []
    
    for x in X:
        # Remove location x
        ind = utils.findIndices(X_tmp,np.array([x]))
        X_tmp2 = utils.removefromArray(X_tmp,ind)
        Y_tmp2 = utils.removefromArray(Y_tmp,ind)
        
        x = x.reshape(1,x.size)
        val = utils.cond_var(x, X_tmp2, k)[0]
        var_Abar.append(val)
        h_yAbar.append(utils.cond_entropy(val))
            
    h_yA = utils.cond_entropy(var_A)

    if divide:
        for sig_A, sig_Abar in zip(var_A,np.asarray(var_Abar)):
            mi.append(np.divide(sig_A,sig_Abar))
    else:
        for sig_A, sig_Abar in zip(h_yA,h_yAbar):
            mi.append(np.subtract(sig_A,sig_Abar))

    return np.asarray(mi), np.asarray(var_Abar), h_yA, np.asarray(h_yAbar)