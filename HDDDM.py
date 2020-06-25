# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import t


from kernel_two_sample_test import kernel_two_sample_test, MMD2u
from sklearn.metrics import pairwise_distances, pairwise_kernels
def test_independence_k2st(X, Y, alpha=0.005):
    sigma2 = np.median(pairwise_distances(X, Y, metric='euclidean'))**2
    _, _, p_value = kernel_two_sample_test(X, Y, kernel_function='rbf', gamma=1.0/sigma2, verbose=False)

    return True if p_value <= alpha else False

def compute_mmd2u(X, Y):
    m = len(X)
    n = len(Y)
    XY = np.vstack([X, Y])
    sigma2 = np.median(pairwise_distances(X, Y, metric='euclidean'))**2
    K = pairwise_kernels(XY, metric='rbf', gamma=1./sigma2)
    
    return MMD2u(K, m, n)


def compute_histogram(X, n_bins):
    return np.array([np.histogram(X[:, i], bins=n_bins, density=False)[0] for i in range(X.shape[1])])

def compute_hellinger_dist(P, Q):
    return np.mean([np.sqrt(np.sum(np.square(np.sqrt(P[i, :] / np.sum(P[i, :])) - np.sqrt(Q[i, :] / np.sum(Q[i, :]))))) for i  in range(P.shape[0])])


# Hellinger Distance Drift Detection Method
class HDDDM():
    def __init__(self, X, gamma=1., alpha=None, use_mmd2=False, use_k2s_test=False):
        if gamma is None and alpha is None:
            raise ValueError("Gamma and alpha can not be None at the same time! Please specify either gamma or alpha")

        self.drift_detected = False
        self.use_mmd2 = use_mmd2
        self.use_k2s_test = use_k2s_test

        self.gamma = gamma
        self.alpha = alpha
        self.n_bins = int(np.floor(np.sqrt(X.shape[0])))

        # Initialization
        self.X_baseline = X
        self.hist_baseline = compute_histogram(X, self.n_bins)
        self.n_samples = X.shape[0]
        self.dist_old = 0.
        self.epsilons = []
        self.t_denom = 0

    def add_batch(self, X):
        self.t_denom += 1
        self.drift_detected = False

        # Compute histogram and the Hellinger distance to the baseline histogram
        hist = compute_histogram(X, self.n_bins)
        dist = compute_hellinger_dist(self.hist_baseline, hist)
        if self.use_mmd2:
            dist = compute_mmd2u(self.X_baseline, X)
        n_samples = X.shape[0]

        # Compute test statistic
        eps = dist - self.dist_old
        self.epsilons.append(eps)

        epsilon_hat = (1. / (self.t_denom)) * np.sum(np.abs(self.epsilons))
        sigma_hat = np.sqrt(np.sum(np.square(np.abs(self.epsilons) - epsilon_hat)) / (self.t_denom))
        
        beta = 0.
        if self.gamma is not None:
            beta = epsilon_hat + self.gamma * sigma_hat
        else:
            beta = epsilon_hat + t.ppf(1.0 - self.alpha / 2, self.n_samples + n_samples - 2) * sigma_hat / np.sqrt(self.t_denom)

        # Test for drift
        drift = np.abs(eps) > beta
        if self.use_k2s_test:
            drift = test_independence_k2st(self.X_baseline, X, alpha=self.alpha)  # Testing for independence: Use the kernel two sample test!

        if drift == True:
            self.drift_detected = True

            self.t_denom = 0
            self.epsilons = []
            self.n_bins = int(np.floor(np.sqrt(n_samples)))
            self.hist_baseline = compute_histogram(X, self.n_bins)
            self.n_samples = n_samples
            self.X_baseline = X
        else:
            self.hist_baseline += hist
            self.n_samples += n_samples
            self.X_baseline = np.vstack((self.X_baseline, X))
    
    def detected_change(self):
        return self.drift_detected
