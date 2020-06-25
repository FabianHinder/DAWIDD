# -*- coding: utf-8 -*-
import numpy as np
from skmultiflow.drift_detection import EDDM, DDM
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.data import STAGGERGenerator, RandomRBFGenerator, SEAGenerator
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle


# Blinking X
def create_blinking_X_dataset(n_samples_per_concept=200, n_concepts=4):
    def labeling_a(n_samples_per_class):
        X0 = np.concatenate((np.arange(0.4, 20.4, step=20. / 50.), np.arange(-20., 0., step=20. / 50.)), axis=0).reshape(-1, 1)
        X0 = np.concatenate([X0, -1. * X0], axis=1)
        Y0 = np.array([0 for _ in range(n_samples_per_class)])

        X1 = np.concatenate((np.arange(0.4, 20.4, step=20. / 50.), np.arange(-20., 0., step=20. / 50.)), axis=0).reshape(-1, 1)
        X1 = np.concatenate([X1, 1. * X1], axis=1)
        Y1 = np.array([1 for _ in range(n_samples_per_class)])

        return np.concatenate([X0, X1], axis=0), np.concatenate([Y0, Y1], axis=0)

    def labeling_b(n_samples_per_class):
        X0 = np.concatenate((np.arange(0.4, 20.4, step=20. / 50.), np.arange(-20., 0., step=20. / 50.)), axis=0).reshape(-1, 1)
        X0 = np.concatenate([X0, -1. * X0], axis=1)
        Y0 = np.array([1 for _ in range(n_samples_per_class)])

        X1 = np.concatenate((np.arange(0.4, 20.4, step=20. / 50.), np.arange(-20., 0., step=20. / 50.)), axis=0).reshape(-1, 1)
        X1 = np.concatenate([X1, 1. * X1], axis=1)
        Y1 = np.array([0 for _ in range(n_samples_per_class)])

        return np.concatenate([X0, X1], axis=0), np.concatenate([Y0, Y1], axis=0)
    
    # Start with labeling a, then switch to labeling b, then again to labeling a, ....
    X_stream = []
    Y_stream = []
    concept_drifts = []

    t = 0
    label_a = True
    for _ in range(n_concepts):
        data_stream_X, data_stream_Y = labeling_a(int(n_samples_per_concept / 2)) if label_a else labeling_b(int(n_samples_per_concept / 2))
        data_stream_X, data_stream_Y = shuffle(data_stream_X, data_stream_Y)
        label_a = not label_a
        t += n_samples_per_concept

        X_stream.append(data_stream_X)
        Y_stream.append(data_stream_Y)
        concept_drifts.append(t)
    concept_drifts.pop()

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)), "drifts": np.array(concept_drifts)}


# Mixed RBF
def create_mixed_rbf_dataset(n_wrong_samples_per_class=120, n_correct_samples_per_class=240, n_concepts=4):
    def mixed_rbf_blob_data(n_wrong_samples, n_correct_samples):
        X = []
        Y = []

        centerA = np.array([[-2., 2.]])
        centerB = np.array([[2., -2.]])

        # Class A
        x, _ = make_blobs(n_samples=n_wrong_samples, n_features=2, centers=centerA, cluster_std=0.5)
        X.append(x)
        Y.append([1 for _ in range(n_wrong_samples)])
        
        x, _ = make_blobs(n_samples=n_correct_samples, n_features=2, centers=centerA, cluster_std=0.8)
        X.append(x)
        Y.append([0 for _ in range(n_correct_samples)])

        # Class B
        x, _ = make_blobs(n_samples=n_wrong_samples, n_features=2, centers=centerB, cluster_std=0.5)
        X.append(x)
        Y.append([0 for _ in range(n_wrong_samples)])
        
        x, _ = make_blobs(n_samples=n_correct_samples, n_features=2, centers=centerB, cluster_std=0.8)
        X.append(x)
        Y.append([1 for _ in range(n_correct_samples)])

        return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)
    
    def unmixed_rbf_blob_data(n_wrong_samples, n_correct_samples):
        X = []
        Y = []

        centerA = np.array([[-2., 2.]])
        centerA2 = np.array([[2., 5.0]])
        centerB = np.array([[2., -2.]])
        centerB2 = np.array([[5.5, 2.]])

        # Class A
        x, _ = make_blobs(n_samples=n_wrong_samples, n_features=2, centers=centerA2, cluster_std=0.5)
        X.append(x)
        Y.append([1 for _ in range(n_wrong_samples)])
        
        x, _ = make_blobs(n_samples=n_correct_samples, n_features=2, centers=centerA, cluster_std=0.8)
        X.append(x)
        Y.append([0 for _ in range(n_correct_samples)])

        # Class B
        x, _ = make_blobs(n_samples=n_wrong_samples, n_features=2, centers=centerB, cluster_std=0.5)
        X.append(x)
        Y.append([0 for _ in range(n_wrong_samples)])
        
        x, _ = make_blobs(n_samples=n_correct_samples, n_features=2, centers=centerB2, cluster_std=0.8)
        X.append(x)
        Y.append([1 for _ in range(n_correct_samples)])

        return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)

    # Start with a mixed sampels, unmix it, mix it again, ...
    X_stream = []
    Y_stream = []
    concept_drifts = []

    t = 0
    mixed = True
    for _ in range(n_concepts):
        data_stream_X, data_stream_Y = mixed_rbf_blob_data(n_wrong_samples_per_class, n_correct_samples_per_class) if mixed else unmixed_rbf_blob_data(n_wrong_samples_per_class, n_correct_samples_per_class)
        data_stream_X, data_stream_Y = shuffle(data_stream_X, data_stream_Y)
        mixed = not mixed
        t += 2 * n_wrong_samples_per_class + n_correct_samples_per_class

        X_stream.append(data_stream_X)
        Y_stream.append(data_stream_Y)
        concept_drifts.append(t)
    concept_drifts.pop()

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)), "drifts": np.array(concept_drifts)}


# Rotating hyperplane dataset
def create_rotating_hyperplane_dataset(n_samples_per_concept=200, concepts=np.arange(0.0, 5.0, 1.0)):
    def create_hyperplane_dataset(n_samples, n_dim=2, plane_angle=0.45):
        w = np.dot(np.array([[np.cos(plane_angle), -np.sin(plane_angle)], [np.sin(plane_angle), np.cos(plane_angle)]]), np.array([1.0, 1.0]))
        X = np.random.uniform(-1.0, 1.0, (n_samples, n_dim))
        Y = np.array([1 if np.dot(x, w) >= 0 else 0 for x in X])
        
        return X, Y
    
    X_stream = []
    Y_stream = []
    concept_drifts = []
    
    t = 0
    for a in concepts:
        data_stream_X, data_stream_Y = create_hyperplane_dataset(n_samples=n_samples_per_concept, plane_angle=a)
        data_stream_X, data_stream_Y = shuffle(data_stream_X, data_stream_Y)
        t += n_samples_per_concept

        X_stream.append(data_stream_X)
        Y_stream.append(data_stream_Y)
        concept_drifts.append(t)
    concept_drifts.pop()

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)), "drifts": np.array(concept_drifts)}


# SEA
def create_sea_drift_dataset(n_samples_per_concept=200, concepts=[0, 1, 2, 3]):
    X_stream = []
    Y_stream = []
    concept_drifts = []
    
    t = 0
    gen = SEAGenerator()
    gen.prepare_for_use()
    for _ in concepts:
        if t != 0:
            concept_drifts.append(t)

        X, y = gen.next_sample(batch_size=n_samples_per_concept)
        X_stream.append(X);Y_stream.append(y)

        gen.generate_drift()
        
        t += n_samples_per_concept

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)), "drifts": np.array(concept_drifts)}


# STAGGER
def create_stagger_drift_dataset(n_samples_per_concept=500, n_concept_drifts=3):
    X_stream = []
    Y_stream = []
    concept_drifts = []

    t = 0
    gen = STAGGERGenerator()
    gen.prepare_for_use()
    for _ in range(n_concept_drifts):
        if t != 0:
            concept_drifts.append(t)
        
        X, y = gen.next_sample(batch_size=n_samples_per_concept)
        X_stream.append(X);Y_stream.append(y)

        gen.generate_drift()
        t += n_samples_per_concept

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)), "drifts": np.array(concept_drifts)}


# Random rbf
def create_rbf_drift_dataset(n_samples_per_concept=500, n_concept_drifts=3):
    X_stream = []
    Y_stream = []
    concept_drifts = []

    t = 0
    for _ in range(n_concept_drifts):
        if t != 0:
            concept_drifts.append(t)
        
        gen = RandomRBFGenerator(n_features=5, n_centroids=10)
        gen.prepare_for_use()
        X, y = gen.next_sample(batch_size=n_samples_per_concept)
        X_stream.append(X);Y_stream.append(y)

        t += n_samples_per_concept

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)), "drifts": np.array(concept_drifts)}


# Gaussians with color mixing
def create_mixing_gaussians_dataset(n_samples_per_concept=500, concepts=[(0,1.),(1,1.),(0,.5),(1,1.),(0,1.)]):
    X_stream = []
    Y_stream = []
    concept_drifts = []

    t = 0
    for concept_type, p in concepts:
        if t != 0:
            concept_drifts.append(t)
        
        X = np.concatenate((
                np.random.multivariate_normal((-3,-3),((1,0),(0,1)),int(n_samples_per_concept*75/200)),
                np.random.multivariate_normal((-3, 3),((1,0),(0,1)),int(n_samples_per_concept*25/200)),
                np.random.multivariate_normal(( 3,-3),((1,0),(0,1)),int(n_samples_per_concept*25/200)),
                np.random.multivariate_normal(( 3, 3),((1,0),(0,1)),int(n_samples_per_concept*75/200))))
        if concept_type == 0:
            y = np.concatenate((
                np.random.choice([-1,1], size=int(n_samples_per_concept*100/200), p=[p,1-p]),
                np.random.choice([-1,1], size=int(n_samples_per_concept*100/200), p=[1-p,p])))
        else:
            y = np.concatenate((
                np.random.choice([-1,1], size=int(n_samples_per_concept*75/200), p=[p,1-p]),
                np.random.choice([-1,1], size=int(n_samples_per_concept*25/200), p=[1-p,p]),
                np.random.choice([-1,1], size=int(n_samples_per_concept*25/200), p=[p,1-p]),
                np.random.choice([-1,1], size=int(n_samples_per_concept*75/200), p=[1-p,p])))

        perm = np.random.permutation(X.shape[0])
        X_stream.append(X[perm,:]);Y_stream.append(y[perm])

        t += n_samples_per_concept

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)), "drifts": np.array(concept_drifts)}


# Two mixing Gaussians mixtures
def create_two_mixing_gaussians_dataset(n_samples_per_concept=500, concepts=[(1.,.5),(.5,1.),(.5,.5),(.5,1.),(1.,.5)]):
    X_stream = []
    Y_stream = []
    concept_drifts = []

    t = 0
    for p, q in concepts:
        if t != 0:
            concept_drifts.append(t)
        
        X = np.concatenate((
                np.random.multivariate_normal((-3,-3),((1,0),(0,1)),int(n_samples_per_concept/4)),
                np.random.multivariate_normal((-3, 3),((1,0),(0,1)),int(n_samples_per_concept/4)),
                np.random.multivariate_normal(( 3,-3),((1,0),(0,1)),int(n_samples_per_concept/4)),
                np.random.multivariate_normal(( 3, 3),((1,0),(0,1)),int(n_samples_per_concept/4))))
        y = np.concatenate((
            np.random.choice([-1,1], size=int(n_samples_per_concept/4), p=[p,1-p]),
            np.random.choice([-1,1], size=int(n_samples_per_concept/4), p=[1-p,p]),
            np.random.choice([-1,1], size=int(n_samples_per_concept/4), p=[q,1-q]),
            np.random.choice([-1,1], size=int(n_samples_per_concept/4), p=[1-q,q])))

        perm = np.random.permutation(X.shape[0])
        X_stream.append(X[perm,:]);Y_stream.append(y[perm])

        t += n_samples_per_concept

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)), "drifts": np.array(concept_drifts)}


# Real world data
import load_rw_data

def create_weather_drift_dataset(n_max_length=1000, n_concept_drifts=3):
    X,y = load_rw_data.read_data_weather()
    return create_controlled_drift_dataset(X,y, n_max_length, n_concept_drifts)

def create_forest_cover_drift_dataset(n_max_length=1000, n_concept_drifts=3):
    X,y = load_rw_data.read_data_forest_cover_type()
    return create_controlled_drift_dataset(X,y, n_max_length, n_concept_drifts)

def create_electricity_market_drift_dataset(n_max_length=1000, n_concept_drifts=3):
    X,y = load_rw_data.read_data_electricity_market()
    return create_controlled_drift_dataset(X,y, n_max_length, n_concept_drifts)

def create_controlled_drift_dataset(X,y=None, n_max_length=1000, n_concept_drifts=3):
    if X.shape[0] > n_max_length:
        sel = np.random.choice(range(X.shape[0]),n_max_length,replace=False); sel.sort()
        X_stream = X[sel]
        if y is not None:
            Y_stream = y[sel]
    else:
        X_stream = X
        if y is not None:
            Y_stream = y

    concept_drifts = np.random.choice(range(X_stream.shape[0]),n_concept_drifts,replace=False); concept_drifts.sort()
    t0 = 0
    for i in range(concept_drifts.shape[0]+1):
        t1 = concept_drifts[i] if i < concept_drifts.shape[0] else X_stream.shape[0]
        perm = np.random.permutation(t1-t0)+t0

        X_stream[t0:t1] = X_stream[perm]
        if y is not None:
            Y_stream[t0:t1] = Y_stream[perm]
        t0 = t1
    
    return {"data": (X_stream, Y_stream.reshape(-1, 1)), "drifts": concept_drifts}


# Drift detection
class DriftDetectorSupervised():
    def __init__(self, clf, drift_detector, training_buffer_size):
        self.clf = clf
        self.drift_detector = drift_detector
        self.training_buffer_size = training_buffer_size
        self.X_training_buffer = []
        self.Y_training_buffer = []
        self.changes_detected = []

    def apply_to_stream(self, X_stream, Y_stream):
        self.changes_detected = []

        collect_samples = False
        T = len(X_stream)
        for t in range(T):
            x, y = X_stream [t,:], Y_stream[t,:]
            
            if collect_samples == False:
                self.drift_detector.add_element(self.clf.score(x, y))

                if self.drift_detector.detected_change():
                    self.changes_detected.append(t)
                    
                    collect_samples = True
                    self.X_training_buffer = []
                    self.Y_training_buffer = []
            else:
                self.X_training_buffer.append(x)
                self.Y_training_buffer.append(y)

                if len(self.X_training_buffer) >= self.training_buffer_size:
                    collect_samples = False
                    self.clf.fit(np.array(self.X_training_buffer), np.array(self.Y_training_buffer))
        
        return self.changes_detected


class DriftDetectorUnsupervised():
    def __init__(self, drift_detector, batch_size):
        self.drift_detector = drift_detector
        self.batch_size = batch_size
        self.changes_detected = []

    def apply_to_stream(self, data_stream):
        self.changes_detected = []
        n_data_stream_samples = len(data_stream)

        t = 0
        while t < n_data_stream_samples:
            end_idx = t+self.batch_size
            if end_idx >= n_data_stream_samples:
                end_idx = n_data_stream_samples

            batch = data_stream[t:end_idx, :]
            self.drift_detector.add_batch(batch)

            if self.drift_detector.detected_change():
                self.changes_detected.append(t)
            
            t += self.batch_size
        
        return self.changes_detected



# Evaluation
def evaluate(true_concept_drifts, pred_concept_drifts, tol=50):
    false_alarms = 0
    drift_detected = 0
    drift_not_detected = 0
    delays = []

    # Check for false alarms
    for t in pred_concept_drifts:
        b = False
        for dt in true_concept_drifts:
            if dt <= t and t <= dt + tol:
                b = True
                break
        if b is False:  # False alarm
            false_alarms += 1
    
    # Check for detected and undetected drifts
    for dt in true_concept_drifts:
        b = False
        for t in pred_concept_drifts:
            if dt <= t and t <= dt + tol:
                b = True
                drift_detected += 1
                delays.append(t - dt)
                break
        if b is False:
            drift_not_detected += 1

    return {"false_alarms": false_alarms, "drift_detected": drift_detected, "drift_not_detected": drift_not_detected, "delays": delays}


# Classifier
from sklearn.svm import SVC
class Classifier():
    def __init__(self, model=SVC(C=1.0, kernel='linear')):
        self.model = model
        self.flip_score = False
    
    def fit(self, X, y):
        self.model.fit(X, y.ravel())
    
    def score(self, x, y):
        s = int(self.model.predict([x]) == y)
        if self.flip_score == True:
            return 1 - s
        else:
            return s

    def score_set(self, X, y):
        return self.model.score(X, y.ravel())
