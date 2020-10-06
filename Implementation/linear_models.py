# -*- coding: utf-8 -*-
import numpy as np
import cvxpy as cp
from utils import score


def compute_pertinent_positive(x_orig, y_orig, base_values, model):
    w, b = model.coef_.reshape(-1), model.intercept_[0] # Extract model parameters

    dim = x_orig.shape[0]
    y_target = -1 if y_orig == 0 else 1
    
    # Variables
    x = cp.Variable(dim)

    # Constants
    epsilon = 1e-3

    r = y_target*(np.dot(w, (x_orig - base_values)) + b)
    w_prime = y_target * w

    # Constraints
    constraints = [w_prime.T @ x - r + epsilon <= 0]

    # Objective
    f = cp.Minimize(cp.norm((x_orig - base_values) - x, 1))    # Minimize Manhattan distance (l1 norm)

    # Solve it!
    prob = cp.Problem(f, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    return np.round((x_orig - base_values) - x.value, 2)


def improve_pertinent_positive(x_orig, y_orig, base_values, pp_orig, model):
    w, b = model.coef_.reshape(-1), model.intercept_[0] # Extract model parameters

    dim = x_orig.shape[0]
    y_target = -1 if y_orig == 0 else 1
    
    fixed_indices = []
    for i in range(dim):
        if np.abs(pp_orig[i] - base_values[i]) <= + 0.1:   # Fix features that are already close to zero (meaning that they might not 'strongly influence' the prediction)
            fixed_indices.append(i)

    # Variables
    x = cp.Variable(dim)

    # Constants
    epsilon = 1e-3

    # Constraints
    constraints = [y_target*(w.T @ x + b) + epsilon >= 0]
    constraints += [x[i] == pp_orig[i] for i in fixed_indices]

    # Objective
    f = cp.Minimize(cp.norm(x_orig - x, 1))    # Minimize Manhattan distance (l1 norm)

    # Solve it!
    prob = cp.Problem(f, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    return np.round(x.value, 2)
