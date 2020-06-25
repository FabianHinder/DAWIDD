# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import ttest_rel
from sklearn.svm import SVC

from joblib import Parallel, delayed
import multiprocessing


def test_independence(X,y, n_itr=10, p_val=0.00001, n_sel=50):
    Z = np.concatenate( (X, np.linspace(0,1,X.shape[0]).reshape(-1,1) ), axis=1)
    svm = SVC(gamma=2, C=1, kernel="rbf")
    s1,s2 = [],[]
    for _ in range(n_itr):
        sel = np.random.choice(range(Z.shape[0]),size=min(n_sel,int(2*Z.shape[0]/3)),replace=False)
        if len(np.unique(y[sel])) == 1: # Number classes has to be greater than one!
            continue

        svm.fit(Z[sel] ,y[sel])
        s1.append(svm.score(Z ,y))
        s2.append(svm.score( np.concatenate( (X, np.random.random(X.shape[0]).reshape(-1,1) ), axis=1) ,y))
    
    if len(s1) == 0 or len(s2) == 0:
        return True
    elif (np.array(s1)-np.array(s2)).var() == 0:
        return abs(np.mean(s1) - np.mean(s2)) < 0.000001
    else:
        return ttest_rel(s1,s2)[1] > p_val
