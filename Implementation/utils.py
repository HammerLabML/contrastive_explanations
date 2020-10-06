# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.utils import shuffle


def score(x_orig, x_pp):
    sparsity = np.linalg.norm(x_orig, ord=0) - np.linalg.norm(x_pp, ord=0)
    closeness = np.linalg.norm([x_orig[i] - x_pp[i] for i in [j for j in range(x_pp.shape[0]) if x_pp[j] != 0 ]], ord=1)  # TODO: Consider base values!

    return (sparsity, closeness)


def get_turned_on_features(a, basis_values):
    result = []
    
    epsilon = 0
    z = np.abs(a - basis_values) <= epsilon
    
    for zi, i in zip(z, range(len(z))):
        if zi == False:
            result.append(i)

    return set(result)


def load_data_iris():
    X, y = load_iris(return_X_y=True)
    base_values = np.zeros(X.shape[1])  # TODO: Come up with more meaningful basis values <- depends on the dataset and sample
    idx = y <= 1 # Convert data into a binary problem
    X, y = X[idx,:], y[idx]

    return X, y, base_values


def load_data_houseprices(file_path="res/data_houseprices.npz"):
    data = np.load(file_path)
    X, y = data["X"], data["y"]
    X = X[:, 1:]    # Remove LotArea
    y = y >= 160000  # boston: >=20000$
    y = y.astype(np.int).flatten()
    X, y = shuffle(X, y, random_state=42)
    base_values = np.zeros(X.shape[1]) 

    return X, y, base_values


def load_data_breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    base_values = np.zeros(X.shape[1])  # TODO: Come up with more meaningful basis values <- depends on the dataset and sample

    return X, y, base_values


def load_data_wine():
    X, y = load_wine(return_X_y=True)
    base_values = np.zeros(X.shape[1])  # TODO: Come up with more meaningful basis values <- depends on the dataset and sample

    idx = y <= 1 # Convert data into a binary problem
    X, y = X[idx,:], y[idx]

    return X, y, base_values


def arrange_results_in_latex_table(scores_pp, scores_pp_plus, scores_features_overlap):
    return f"${np.round(np.mean(scores_pp['sparsity']), 2)} (\pm {np.round(np.var(scores_pp['sparsity']), 2)})$\n${np.round(np.mean(scores_pp['closeness']), 2)} (\pm {np.round(np.var(scores_pp['closeness']), 2)})$\n${np.round(np.mean(scores_pp_plus['closeness']), 2)} (\pm {np.round(np.var(scores_pp_plus['closeness']), 2)})$\n${np.round(np.mean(scores_features_overlap), 2)} (\pm {np.round(np.var(scores_features_overlap), 2)})$"
