# -*- coding: utf-8 -*-
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn_lvq import GlvqModel

import random

from utils import score, get_turned_on_features, arrange_results_in_latex_table, load_data_iris, load_data_breast_cancer, load_data_houseprices, load_data_wine, load_data_digits
from linear_models import compute_strict_pertinent_positive, compute_pertinent_positive as compute_pertinent_positive_of_linear_model, improve_pertinent_positive as improve_pertinent_positive_of_linear_model
from lvq_models import compute_pertinent_positive as compute_pertinent_positive_of_lvq_model, improve_pertinent_positive as improve_pertinent_positive_of_lvq_model
from quadratic_models import compute_pertinent_positive as compute_pertinent_positive_of_quadratic_model, improve_pertinent_positive as improve_pertinent_positive_of_quadratic_model

from ceml.sklearn import generate_counterfactual

n_kfold_splits = 3


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: <dataset_desc> <model_desc>")
    else:
        dataset_desc = sys.argv[1]
        model_desc = sys.argv[2]

        # Load data
        X, y, base_values = None, None, None

        if dataset_desc == "iris":
            X, y, base_values = load_data_iris()
        elif dataset_desc == "houseprices":
            X, y, base_values = load_data_houseprices()
        elif dataset_desc == "breastcancer":
            X, y, base_values = load_data_breast_cancer()
        elif dataset_desc == "wine":
            X, y, base_values = load_data_wine()
        elif dataset_desc == "digits":
            X, y, base_values = load_data_digits()
        print(f"Dimensionality: {X.shape[1]}")

        # Results
        scores_pp = {"sparsity": [], "closeness": []}
        scores_pp_plus = {"sparsity": [], "closeness": []}
        scores_features_overlap = []
        scores_pp_strict = []

        original_samples = []
        pertinent_positives = []

        # k-fold cross validation
        kf = KFold(n_splits=n_kfold_splits, shuffle=True)
        for train_indices, test_indices in kf.split(X):
            X_train, X_test, y_train, y_test = X[train_indices, :], X[test_indices], y[train_indices], y[test_indices]
            print(f"Train size: {X_train.shape}\nTest size: {X_test.shape}")

            # Preprocessing (could be inverted - but we do not care xD)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Fit model
            model, compute_pertinent_positive, improve_pertinent_positive = None, None, None

            if model_desc == "logreg":
                model = LogisticRegression(multi_class='multinomial')   # OK
                compute_pertinent_positive = compute_pertinent_positive_of_linear_model
                improve_pertinent_positive = improve_pertinent_positive_of_linear_model
            elif model_desc == "glvq":
                model = GlvqModel(prototypes_per_class=3, max_iter=100)
                compute_pertinent_positive = compute_pertinent_positive_of_lvq_model
                improve_pertinent_positive = improve_pertinent_positive_of_lvq_model
            elif model_desc == "qda":
                model = QuadraticDiscriminantAnalysis(store_covariance=True, reg_param=1.0)
                compute_pertinent_positive = compute_pertinent_positive_of_quadratic_model
                improve_pertinent_positive = improve_pertinent_positive_of_quadratic_model

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            print(f"F1-score: {f1_score(y_test, y_pred)}")
            print()
            
            # Classify zero vector
            y_zerovector = model.predict(np.zeros(X_train.shape[1]).reshape(1, -1))[0]
            print(f"Classification of the zero vector: {y_zerovector}")

            # Compute pertinent positives and counterfactual explanations of all test samples
            n_wrong_classification = 0
            n_failures = 0
            n_cf_failures = 0
            for i in range(X_test.shape[0]):
                x_orig, y_orig = X_test[i,:],y_test[i]
                if model.predict([x_orig]) != y_orig:
                    n_wrong_classification += 1
                    continue

                if y_orig == y_zerovector:  # Ignore class of zero vector -> zero vector would be the sparstest and most trivial (valid) pertinent positive
                    continue

                # Compute counterfactual explanation
                cf_turned_on_features = set([])
                try:
                    y_target = 1 if y_orig == 0 else 0
                    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=y_target, return_as_dict=False)
                    if model.predict([x_cf]) != y_target:
                        n_cf_failures += 1
                    cf_turned_on_features = get_turned_on_features(delta, base_values)
                except Exception as ex:
                    n_cf_failures += 1

                # Compute pertinent positives
                x_pp = compute_pertinent_positive(x_orig, y_orig, base_values, model)
                if x_pp is None:
                    n_failures += 1
                    continue
                if model.predict([x_pp]) != y_orig:
                    n_failures += 1
                    continue
                x_pp_plus = improve_pertinent_positive(x_orig, y_orig, base_values, x_pp, model) # Improve pertinent positive by a "local search"
                if model.predict([x_pp_plus]) != y_orig:
                    n_failures += 1
                    continue

                # Compute scores
                pp_turned_on_features = get_turned_on_features(x_pp, base_values)
                scores_features_overlap.append(len(cf_turned_on_features.intersection(pp_turned_on_features)))

                s_pp = score(x_orig, x_pp)
                scores_pp["sparsity"].append(s_pp[0]);scores_pp["closeness"].append(s_pp[1])
                s_pp_plus = score(x_orig, x_pp_plus)
                scores_pp_plus["sparsity"].append(s_pp_plus[0]);scores_pp_plus["closeness"].append(s_pp_plus[1])

                # If possible - compute a globally optimal solution for comparison
                if model_desc == "logreg":
                    x_pp_strict = compute_strict_pertinent_positive(x_orig, y_orig, base_values, model)
                    s_pp_strict = score(x_orig, x_pp_strict)
                    scores_pp_strict.append(s_pp_strict[0])

                # Save samples
                original_samples.append(x_orig)
                pertinent_positives.append(x_pp_plus)

            print(f"Number of missclassifications (skiped): {n_wrong_classification}")
            print(f"Number of failures (skiped): {n_failures}")
            print(f"Number of invalid counterfactuals: {n_cf_failures}")

        # Save samples
        #np.savez("data.npz", x_orig=np.array(original_samples), x_pp_plus=np.array(pertinent_positives))

        # Summarize results
        print(f"Num samples: {len(scores_pp['sparsity'])}")

        if model_desc == "logreg":
            print("\nPP-Strict")
            print(scores_pp_strict)
            print(f"=>Sparsity:\nMean: {np.mean(scores_pp_strict)}\nMedian: {np.median(scores_pp_strict)}\nVar: {np.var(scores_pp_strict)}\nStd: {np.std(scores_pp_strict)}")
        

        print("\nPP")
        print(scores_pp['sparsity'])
        print(scores_pp['closeness'])
        print(f"=>Sparsity:\nMean: {np.mean(scores_pp['sparsity'])}\nMedian: {np.median(scores_pp['sparsity'])}\nVar: {np.var(scores_pp['sparsity'])}\nStd: {np.std(scores_pp['sparsity'])}")
        print(f"=>Closeness:\nMean: {np.mean(scores_pp['closeness'])}\nMedian: {np.median(scores_pp['closeness'])}\nVar: {np.var(scores_pp['closeness'])}\nStd: {np.std(scores_pp['closeness'])}")

        print("\nPP+")
        print(scores_pp_plus['closeness'])
        print(f"=>Sparsity:\nMean: {np.mean(scores_pp_plus['sparsity'])}\nMedian: {np.median(scores_pp_plus['sparsity'])}\nVar: {np.var(scores_pp_plus['sparsity'])}\nStd: {np.std(scores_pp_plus['sparsity'])}")
        print(f"=>Closeness:\nMean: {np.mean(scores_pp_plus['closeness'])}\nMedian: {np.median(scores_pp_plus['closeness'])}\nVar: {np.var(scores_pp_plus['closeness'])}\nStd: {np.std(scores_pp_plus['closeness'])}")

        print("\nFeature overlap")
        print(scores_features_overlap)
        print(f"=>\nMean: {np.mean(scores_features_overlap)}\nMedian: {np.median(scores_features_overlap)}\nVar: {np.var(scores_features_overlap)}\nStd: {np.std(scores_features_overlap)}")
        
        print()
        print(arrange_results_in_latex_table(scores_pp, scores_pp_plus, scores_features_overlap))
