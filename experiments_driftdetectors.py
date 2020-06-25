# -*- coding: utf-8 -*-
import os
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.naive_bayes import GaussianNB
from skmultiflow.drift_detection import ADWIN, EDDM, DDM, PageHinkley
from experiments_utils import Classifier, DriftDetectorUnsupervised, DriftDetectorSupervised, evaluate, create_mixed_rbf_dataset, create_two_mixing_gaussians_dataset, create_blinking_X_dataset, create_rbf_drift_dataset, create_rotating_hyperplane_dataset, create_sea_drift_dataset, create_stagger_drift_dataset, create_weather_drift_dataset, create_forest_cover_drift_dataset, create_electricity_market_drift_dataset,create_mixing_gaussians_dataset,create_mixing_gaussians_dataset
from DAWIDD import DAWIDD as SWIDD
from HDDDM import HDDDM  


def analyse_data(data,n_dec=2, show_var=True):
    data = np.array(data)

    if data.shape[0] == 0:
        return "--"

    mean, var = data.mean().round(n_dec), data.var().round(n_dec)

    if not(show_var) or var == 0:
        return "$"+str(mean)+"$"
    else:
        return "$"+str(mean)+" (\\pm  "+str(var)+")$"


if __name__ == "__main__":
    n_itr = 80

    def run_test_drift_detectors():
        results = {"RotatingHyperplane": {},
                "SEA":  {},
                "STAGGER":  {},
                "RandomRBF":  {},
                "BlinkingX": {},
                "MixingGaussians": {}, 
                "TwoMixingGaussians": {}, 
                "Weather": {},
                "ForestCover": {},
                "ElectricityMarket": {}}


        datasets = [("RotatingHyperplane", create_rotating_hyperplane_dataset(n_samples_per_concept=200, concepts=[0.0, 2.0, 4.0, 6.0])),
                    ("SEA", create_sea_drift_dataset(n_samples_per_concept=400, concepts=[0, 1, 2, 0])),
                    ("STAGGER", create_stagger_drift_dataset(n_samples_per_concept=500, n_concept_drifts=4)),
                    ("RandomRBF", create_rbf_drift_dataset(n_samples_per_concept=200, n_concept_drifts=4)),
                    ("BlinkingX", create_blinking_X_dataset(n_concepts=4)),
                    ("MixingGaussians",create_mixing_gaussians_dataset(n_samples_per_concept=200, concepts=[(0,1.),(1,1.),(0,.5),(1,1.),(0,1.)])),
                    ("TwoMixingGaussians",create_two_mixing_gaussians_dataset(n_samples_per_concept=200, concepts=[(1.,.5),(.5,1.),(.5,.5),(.5,1.),(1.,.5)])),
                    ("Weather",create_weather_drift_dataset(n_concept_drifts=4)),
                    ("ForestCover",create_forest_cover_drift_dataset(n_concept_drifts=4)),
                    ("ElectricityMarket",create_electricity_market_drift_dataset(n_concept_drifts=4))]

        def test_on_data_set(data_desc, D):
            r = {data_desc: {"HDDDM": [], "SWIDD": [], "EDDM": [], "DDM": [], "ADWIN": [], "PageHinkley": []}}

            training_buffer_size = 100  # Size of training buffer of the drift detector
            n_train = 200   # Initial training set size

            concept_drifts = D["drifts"]
            X, Y = D["data"]
            data_stream = np.concatenate((X, Y.reshape(-1, 1)), axis=1)


            X0, Y0 = X[0:n_train, :], Y[0:n_train, :]   # Training dataset
            data0 = data_stream[0:n_train,:]

            X_next, Y_next = X[n_train:, :], Y[n_train:, :]  # Test set
            data_next = data_stream[n_train:,:]

            # Run unsupervised drift detector  
            dd = DriftDetectorUnsupervised(HDDDM(data0, gamma=None, alpha=0.005), batch_size=50)
            changes_detected = dd.apply_to_stream(data_next)
            
            # Evaluation
            scores = evaluate(concept_drifts, changes_detected)
            r[data_desc]["HDDDM"].append(scores)

            dd = DriftDetectorUnsupervised(SWIDD(max_window_size=300, min_window_size=100), batch_size=1)
            changes_detected = dd.apply_to_stream(data_next)
            
            # Evaluation
            scores = evaluate(concept_drifts, changes_detected)
            r[data_desc]["SWIDD"].append(scores)

            # Run supervised drift detector
            model = GaussianNB()
            
            # EDDM
            drift_detector = EDDM()

            clf = Classifier(model)
            clf.flip_score = True
            clf.fit(X0, Y0.ravel())

            dd = DriftDetectorSupervised(clf=clf, drift_detector=drift_detector, training_buffer_size=training_buffer_size)
            changes_detected = dd.apply_to_stream(X_next, Y_next)
            
            # Evaluation
            scores = evaluate(concept_drifts, changes_detected)
            r[data_desc]["EDDM"].append(scores)

            # DDM
            drift_detector = DDM(min_num_instances=30, warning_level=2.0, out_control_level=3.0)
            
            clf = Classifier(model)
            clf.flip_score = True
            clf.fit(X0, Y0.ravel())

            dd = DriftDetectorSupervised(clf=clf, drift_detector=drift_detector, training_buffer_size=training_buffer_size)
            changes_detected = dd.apply_to_stream(X_next, Y_next)
            
            # Evaluation
            scores = evaluate(concept_drifts, changes_detected)
            r[data_desc]["DDM"].append(scores)

            # ADWIN
            drift_detector = ADWIN(delta=2.)

            clf = Classifier(model)
            clf.fit(X0, Y0.ravel())

            dd = DriftDetectorSupervised(clf=clf, drift_detector=drift_detector, training_buffer_size=training_buffer_size)
            changes_detected = dd.apply_to_stream(X_next, Y_next)
            
            # Evaluation
            scores = evaluate(concept_drifts, changes_detected)
            r[data_desc]["ADWIN"].append(scores)
            
            # PageHinkley
            drift_detector = PageHinkley()
            
            clf = Classifier(model)
            clf.flip_score = True
            clf.fit(X0, Y0.ravel())

            dd = DriftDetectorSupervised(clf=clf, drift_detector=drift_detector, training_buffer_size=training_buffer_size)
            changes_detected = dd.apply_to_stream(X_next, Y_next)
            
            # Evaluation
            scores = evaluate(concept_drifts, changes_detected)
            r[data_desc]["PageHinkley"].append(scores)
        
            return r

        # Test all data sets
        r_all_datasets = Parallel(n_jobs=-1)(delayed(test_on_data_set)(data_desc, D) for data_desc, D in datasets)
        for r_data in r_all_datasets:
            for k in r_data.keys():
                results[k] = r_data[k]
        
        return results

    # Run tests in parallel
    all_results = Parallel(n_jobs=-1)(delayed(run_test_drift_detectors)() for _ in range(n_itr))

    # Merge results
    myresults = {"RotatingHyperplane": {"HDDDM": [], "SWIDD": [], "EDDM": [], "DDM": [], "ADWIN": [], "PageHinkley": []},
               "SEA":  {"HDDDM": [], "SWIDD": [], "EDDM": [], "DDM": [], "ADWIN": [], "PageHinkley": []},
               "STAGGER":  {"HDDDM": [], "SWIDD": [], "EDDM": [], "DDM": [], "ADWIN": [], "PageHinkley": []},
               "RandomRBF":  {"HDDDM": [], "SWIDD": [], "EDDM": [], "DDM": [], "ADWIN": [], "PageHinkley": []},
               "BlinkingX": {"HDDDM": [], "SWIDD": [], "EDDM": [], "DDM": [], "ADWIN": [], "PageHinkley": []},
               "MixingGaussians": {"HDDDM": [], "SWIDD": [], "EDDM": [], "DDM": [], "ADWIN": [], "PageHinkley": []}, 
               "TwoMixingGaussians": {"HDDDM": [], "SWIDD": [], "EDDM": [], "DDM": [], "ADWIN": [], "PageHinkley": []}, 
               "Weather": {"HDDDM": [], "SWIDD": [], "EDDM": [], "DDM": [], "ADWIN": [], "PageHinkley": []},
               "ForestCover": {"HDDDM": [], "SWIDD": [], "EDDM": [], "DDM": [], "ADWIN": [], "PageHinkley": []},
               "ElectricityMarket": {"HDDDM": [], "SWIDD": [], "EDDM": [], "DDM": [], "ADWIN": [], "PageHinkley": []}}
    for r in all_results:
        for k in r.keys():
            for m in r[k].keys():
                myresults[k][m] += r[k][m]

    # Final evaluation over all runs
    models = ["SWIDD", "HDDDM", "EDDM", "DDM", "ADWIN"]
    for dataset in myresults.keys():
        print("\\hline")
        print("\\multirow{5}{*}{\\rotatebox[origin=c]{90}{"+str(dataset)+"}}")
        for model in models:
            false_alarms = []
            drift_detected = []
            drift_not_detected = []
            delay = []
            for r in myresults[dataset][model]:
                false_alarms.append(r["false_alarms"])
                drift_detected.append(r["drift_detected"])
                drift_not_detected.append(r["drift_not_detected"])
                delay += r["delays"]
            false_alarms = analyse_data(false_alarms)
            drift_detected = analyse_data(drift_detected)
            drift_not_detected = analyse_data(drift_not_detected)
            delay = analyse_data(delay, show_var=False)

            print("&  {0} & {1} & {2} & {3} & {4} \\\\".format(model, drift_detected, drift_not_detected, false_alarms, delay))
        print("\\hline")
        print("")

    # Compute ranks
    print("Rank")

    from itertools import combinations
    def sub_lists(my_list):
        subs = []
        for i in range(0, len(my_list)+1):
            temp = [list(x) for x in combinations(my_list, i)]
            if len(temp)>0:
                subs.extend(temp)
        return subs

    base_data = []
    for sel_datasets in [l for l in sub_lists( list(myresults.keys()) ) if len(l) > 0 and all(i in l for i in base_data)]:
        data = []
        models = ["SWIDD", "HDDDM", "EDDM", "DDM", "ADWIN"]
        for model in models:
            data_ = []
            for dataset in sel_datasets:
                for r in myresults[dataset][model]:
                    data_.append( [r["drift_detected"],-r["drift_not_detected"],-r["false_alarms"],-np.mean(r["delays"]) if len(r["delays"]) > 0 else -np.infty] )
            data.append(data_)
        data = np.array(data)
        rank = np.empty( data.shape )
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                rank[:,i,j] = rankdata(-data[:,i,j], method="average" )

        print("  ",sel_datasets)
        mean_rank = rank.mean(axis=1).round(1)
        for i,model in enumerate(models):
            drift_detected, drift_not_detected, false_alarms, delay = mean_rank[i][0],mean_rank[i][1],mean_rank[i][2],mean_rank[i][3]
            print("    {0} & ${1}$ & ${2}$ & ${3}$ & ${4}$ \\\\".format(model, drift_detected, drift_not_detected, false_alarms, delay))
