# -*- coding: utf-8 -*-
import numpy as np
import cvxpy as cp
import sklearn


def compute_approx_pertinent_positive(x_orig, y_orig, base_values, model, n_max_iter=10):
    try:
        # Constants    
        dim = x_orig.shape[0]
        epsilon = 1e-2

        if isinstance(model, sklearn.naive_bayes.GaussianNB):
            means = model.theta_
            class_priors = model.class_prior_
            sigma_inv = [np.linalg.inv(np.diag(model.sigma_[i,:])) for i in range(model.class_count_.shape[0])]
        else:
            means = model.means_
            sigma_inv = [np.linalg.inv(cov) for cov in model.covariance_]
            class_priors = model.priors_

        x_orig_prime = x_orig - base_values
        
        # Constants for constraints
        i = int(y_orig)
        j = 0 if y_orig == 1 else 1

        q = np.dot(sigma_inv[j], means[j]) - np.dot(sigma_inv[i], means[i])
        c = np.log(class_priors[j] / class_priors[i]) + 0.5 * np.log(np.linalg.det(sigma_inv[j]) / np.linalg.det(sigma_inv[i])) + 0.5 * (means[i].T.dot(sigma_inv[i]).dot(means[i]) - means[j].T.dot(sigma_inv[j]).dot(means[j]))
        
        # CCP: Repeat the following loop
        x_cur = np.zeros(dim)   # Delta=0 => We take the original sample as a starting point

        for _ in range(n_max_iter):
            # Variables
            x = cp.Variable(dim)

            # Build constraints
            constraints = []

            c_prime = c + 0.5 * x_cur.T.dot(sigma_inv[j]).dot(x_cur) + x_orig_prime.T @ q - 0.5 * x_orig_prime.T.dot(sigma_inv[j]).dot(x_orig_prime) + 0.5 * x_orig_prime.T.dot(sigma_inv[i]).dot(x_orig_prime) + epsilon
            constraints += [cp.quad_form(x, sigma_inv[i]) - x.T @ q - np.dot(x_orig_prime, sigma_inv[i]).T @ x + np.dot(x_orig_prime, sigma_inv[j]).T @ x -\
                            x.T @ np.dot(sigma_inv[j], x_cur) + c_prime <= 0]

            # Objective
            f = cp.Minimize(cp.norm(x_orig_prime - x, 1))

            # Solve it!
            prob = cp.Problem(f, constraints)
            prob.solve(solver=cp.SCS, verbose=False)

            x_cur = x.value
        
        return np.round(x_orig_prime - x_cur, 2)
    except Exception as ex:
        #print(ex)
        return None


def compute_pertinent_positive(x_orig, y_orig, base_values, model):
    return compute_approx_pertinent_positive(x_orig, y_orig, base_values, model)

def improve_pertinent_positive(x_orig, y_orig, base_values, pp_orig, model, n_max_iter=10):
    # Constants    
    dim = x_orig.shape[0]
    epsilon = 1e-2
    
    # Enumerate all "turned off" features
    fixed_indices = []
    for i in range(dim):
        if np.abs(pp_orig[i] - base_values[i]) <= 0.1:   # Fix features that are already close to zero (meaning that they might not 'strongly influence' the prediction)
            fixed_indices.append(i)

    # Build QP (basically the same as the other method but this time we use a different objective and additional constraints for fixing some features)
    if isinstance(model, sklearn.naive_bayes.GaussianNB):
        means = model.theta_
        class_priors = model.class_prior_
        sigma_inv = [np.linalg.inv(np.diag(model.sigma_[i,:])) for i in range(model.class_count_.shape[0])]
    else:
        means = model.means_
        sigma_inv = [np.linalg.inv(cov) for cov in model.covariance_]
        class_priors = model.priors_
    
    # Constants for constraints
    i = int(y_orig)
    j = 0 if y_orig == 1 else 1

    q = np.dot(sigma_inv[j], means[j]) - np.dot(sigma_inv[i], means[i])
    c = np.log(class_priors[j] / class_priors[i]) + 0.5 * np.log(np.linalg.det(sigma_inv[j]) / np.linalg.det(sigma_inv[i])) + 0.5 * (means[i].T.dot(sigma_inv[i]).dot(means[i]) - means[j].T.dot(sigma_inv[j]).dot(means[j]))
    
    # CCP: Repeat the following loop
    x_cur = pp_orig
    for _ in range(n_max_iter):
        # Variables
        x = cp.Variable(dim)

        # Build constraints
        constraints = []
        constraints += [x[i] == pp_orig[i] for i in fixed_indices]  # Fix "turned off" features

        # Constraints without any basis values! x is the x no, delta, no b!
        c_prime = c + 0.5*x_cur.T.dot(sigma_inv[j]).dot(x_cur) + epsilon
        constraints += [0.5*cp.quad_form(x, sigma_inv[i]) + x.T @ q - x.T @ np.dot(sigma_inv[j], x_cur) + c_prime <= 0]

        # Objective
        f = cp.Minimize(cp.norm(x_orig - x, 1))

        # Solve it!
        prob = cp.Problem(f, constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        if x.value is None:
            print("Failed to find a solution")
            break
        else:
            x_cur = x.value
    
    x_final = np.round(x_cur, 2)
    if np.linalg.norm(x_final - x_orig, 2) > np.linalg.norm(pp_orig - x_orig, 2):
        return pp_orig
    else:
        return np.round(x_cur, 2)
