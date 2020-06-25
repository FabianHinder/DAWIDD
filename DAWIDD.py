# -*- coding: utf-8 -*-
import numpy as np
from svm_test import test_independence as svm_independence_test


def test_independence(X, Y, Z=None):
    return svm_independence_test(X, Y)


class DAWIDD():
    """
    Implementation of the dynamic-adapting-window-independence-drift-detector (DAWIDD)
    
    Parameters
    ----------
    max_window_size : int, optional
        The maximal size of the window. When reaching the maximal size, the oldest sample is removed.

        The default is 90
    min_window_size : int, optional
        The minimal number of samples that is needed for computing the hypothesis test.

        The default is 70
    min_p_value : int, optional
        The threshold of the p-value - not every test outputs a p-value (sometimes only 1.0 <=> independent and 0.0 <=> not independent are returned)

        The default is 0.001
    """
    def __init__(self, max_window_size=90, min_window_size=70, min_p_value=0.001):
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.min_p_value = min_p_value

        self.X = []
        self.n_items = 0
        self.min_n_items = self.min_window_size / 4.

        self.drift_detected = False
    
    # You have to overwrite this function if you want to use a different test for independence
    def _test_for_independence(self):
        t = np.array(range(self.n_items)) / (1. * self.n_items)
        t /= np.std(t)
        t = t.reshape(-1, 1)

        X = np.array(self.X)
        X_ = X[:,:-1].reshape(X.shape[0], -1)
        Y = X[:, -1].reshape(-1, 1)
        return test_independence(X_, Y.ravel())

    def set_input(self, x):
        self.add_batch(x)

        return self.detected_change()

    def add_batch(self, x):
        self.drift_detected = False

        # Add item
        self.X.append(x.flatten())
        self.n_items += 1
        
        # Is buffer full?
        if self.n_items > self.max_window_size:
            self.X.pop(0)
            self.n_items -= 1

        # Enough items for testing for drift?
        if self.n_items >= self.min_window_size:
            # Test for drift
            p = self._test_for_independence()

            if p <= self.min_p_value:
                self.drift_detected = True
                
                # Remove samples until no drift is present!
                while p <= self.min_p_value and self.n_items >= self.min_n_items:
                    # Remove old samples
                    self.X.pop(0)
                    self.n_items -= 1


    def detected_change(self):
        return self.drift_detected
