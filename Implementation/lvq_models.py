# -*- coding: utf-8 -*-
import numpy as np
import cvxpy as cp
import sklearn_lvq


def compute_pertinent_positive(x_orig, y_orig, base_values, model):
    dim = x_orig.shape[0]
    
    # Variables
    x = cp.Variable(dim)

    # Constants
    epsilon = 1e-3

    distmat = None
    if isinstance(model, sklearn_lvq.GlvqModel):
        distmat = np.eye(dim)
    elif isinstance(model, sklearn_lvq.GmlvqModel):
        distmat = np.dot(model.omega_.T, model.omega_)
    else:
        raise TypeError("model must be an instance of sklearn_lvq.GlvqModel or sklearn_lvq.GmlvqModel")
    
    x_orig_prime = x_orig - base_values
    
    prototypes = [] # Split prototypes into two groups (correct vs. incorrect target label)
    other_prototypes = []
    for y, p in zip(model.c_w_, model.w_):
        if y != y_orig:
            other_prototypes.append(p)
        else:
            prototypes.append(p)

    results_scores = []
    results_pps = []

    for i in range(len(prototypes)):   # Repeat for all prototypes with the correct target label - divide and conquer strategy -> selected the one with the smallest objective
        # Constraints
        constraints = []

        p_i = prototypes[i]
        z_i = x_orig_prime - p_i

        for p_j in other_prototypes:
            z_j = x_orig_prime - p_j
            q = -2.*np.dot(z_i.T, distmat) + 2.*np.dot(z_j.T, distmat)
            c = epsilon + np.dot(z_i.T, np.dot(distmat, z_i)) - np.dot(z_j.T, np.dot(distmat, z_j))

            constraints.append(q.T @ x + c <= 0)

        # Objective
        f = cp.Minimize(cp.norm((x_orig - base_values) - x, 1))    # Minimize Manhattan distance (l1 norm)

        # Solve it!
        prob = cp.Problem(f, constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        results_pps.append(np.round((x_orig - base_values) - x.value, 2))
        results_scores.append(f.value)

    return results_pps[np.argmin(results_scores)]


def improve_pertinent_positive(x_orig, y_orig, base_values, pp_orig, model):
    dim = x_orig.shape[0]
    
    fixed_indices = []
    for i in range(dim):
        if np.abs(pp_orig[i] - base_values[i]) <= 0.1:   # Fix features that are already close to zero (meaning that they might not 'strongly influence' the prediction)
            fixed_indices.append(i)

    # Variables
    x = cp.Variable(dim)

    # Constants
    epsilon = 1e-3

    distmat = None
    if isinstance(model, sklearn_lvq.GlvqModel):
        distmat = np.eye(dim)
    elif isinstance(model, sklearn_lvq.GmlvqModel):
        distmat = np.dot(model.omega_.T, model.omega_)
    else:
        raise TypeError("model must be an instance of sklearn_lvq.GlvqModel or sklearn_lvq.GmlvqModel")
    
    prototypes = [] # Split prototypes into two groups (correct vs. incorrect target label)
    other_prototypes = []
    for y, p in zip(model.c_w_, model.w_):
        if y != y_orig:
            other_prototypes.append(p)
        else:
            prototypes.append(p)

    results_scores = []
    results_pps = []

    for i in range(len(prototypes)):   # Repeat for all prototypes with the correct target label - divide and conquer strategy -> selected the one with the smallest objective
        # Constraints
        constraints = [x[i] == pp_orig[i] for i in fixed_indices]  # Do not change fixed features

        p_i = prototypes[i]
        z_i = p_i

        for p_j in other_prototypes:
            z_j = p_j
            q = -2.*np.dot(z_i.T, distmat) + 2.*np.dot(z_j.T, distmat)
            c = epsilon + np.dot(z_i.T, np.dot(distmat, z_i)) - np.dot(z_j.T, np.dot(distmat, z_j))

            constraints.append(q.T @ x + c <= 0)

        # Objective
        f = cp.Minimize(cp.norm(x_orig - x, 1))    # Minimize Manhattan distance (l1 norm)

        # Solve it!
        prob = cp.Problem(f, constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        if x.value is not None:
            results_pps.append(np.round(x.value, 2))
            results_scores.append(f.value)
        else:
            results_pps.append(np.round(pp_orig, 2))
            results_scores.append(np.linalg.norm(x_orig - pp_orig, ord=1))

    return results_pps[np.argmin(results_scores)]
