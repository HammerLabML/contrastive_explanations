# -*- coding: utf-8 -*-
import pickle
import numpy as np
import os
np.random.seed(42)
import cvxpy as cp
import seaborn as sns
import random
random.seed(42)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from pertinent_negative import FeasibleCounterfactual


class FeasiblePertinentPositive:
    def __init__(self, w, b, X, ellipsoids_r, gmm_weights, gmm_means, gmm_covariances, projection_matrix=None, projection_mean_sub=None, density_constraint=True, density_threshold=-85):
        self.w = w
        self.b = b

        self.X = X
        self.gmm_weights = gmm_weights
        self.gmm_means = gmm_means
        self.gmm_covariances = gmm_covariances
        self.ellipsoids_r = ellipsoids_r
        self.projection_matrix = np.eye(self.X.shape[1]) if projection_matrix is None else projection_matrix
        self.projection_mean_sub = np.zeros(self.X.shape[1]) if projection_mean_sub is None else projection_mean_sub
        self.density_constraint = density_constraint

        self.gmm_cluster_index = 0
        self.min_density = density_threshold
        self.epsilon = 1e-3#1e-5

    def _build_constraints(self, var_x, y):
        constraints = []
        if self.w.shape[0] > 1:
            for i in range(self.w.shape[0]):
                if i != y:
                    constraints += [(self.projection_matrix @ (var_x - self.projection_mean_sub)).T @ (self.w[i,:] - self.w[y,:]) + (self.b[i] - self.b[y]) + self.epsilon <= 0]
        else:
            if y == 0:  # TODO: Supports binary classification only!
                return [(self.projection_matrix @ (var_x - self.projection_mean_sub)).T @ self.w.reshape(-1, 1) + self.b + self.epsilon <= 0]
            else:
                return [(self.projection_matrix @ (var_x - self.projection_mean_sub)).T @ self.w.reshape(-1, 1) + self.b - self.epsilon >= 0]

        return constraints

    def compute_counterfactual(self, x, y, regularizer="l1"):        
        xcf = None
        s = float("inf")
        for i in range(self.gmm_weights.shape[0]):
            try:
                # Compute a sparse pertinent positive
                self.gmm_cluster_index = i
                xcf_ = self.build_solve_opt(x, y)
                if xcf_ is None:
                    continue
                
                # Improve pertinent positive
                xcf_ = self.build_solve_opt(xcf_, y, True)
                if xcf_ is None:
                    continue

                # Better than all the previous ones?
                s_ = None
                s_ = np.sum(np.abs(x - xcf_))

                if s_ <= s:
                    s = s_
                    xcf = xcf_
            except Exception as ex:
                print(ex)
        return xcf
    
    def _solve(self, prob):
        prob.solve(solver=cp.SCS, verbose=False)

    def build_solve_opt(self, x_orig, y, improve_pp=False):
        dim = x_orig.shape[0]

        # Variables
        x = cp.Variable(dim)
        x_var = x_orig - x if improve_pp == False else x

        fixed_indices = []
        if improve_pp == True:  # Fix "turned off" features
            for i in range(dim):
                if np.abs(x_orig[i]) <= 0.1:   # Fix features that are already close to zero (meaning that they might not 'strongly influence' the prediction)
                    fixed_indices.append(i)

        # Construct constraints
        constraints = self._build_constraints(x_var, y)
        constraints += [x[i] == x_orig[i] for i in fixed_indices]

        if self.density_constraint is True:
            i = self.gmm_cluster_index
            x_i = self.gmm_means[i]
            cov = np.linalg.inv(self.gmm_covariances[i])

            constraints += [cp.quad_form(self.projection_matrix @ (x_var - self.projection_mean_sub) - x_i, cov) - self.ellipsoids_r[i] <= 0] # Numerically much more stable than the explicit density omponent constraint

        # Build the final program
        f = None
        if improve_pp == True:
            f = cp.Minimize(cp.norm(x_orig - x, 1))    # Minimize Manhattan distance (l1 norm)
        else:
            f = cp.Minimize(cp.norm(x_var, 1))

        prob = cp.Problem(f, constraints)

        # Solve it!
        self._solve(prob)

        if improve_pp == False:
            return np.round(x_orig - x.value, 2)
        else:
            return np.round(x.value, 2)


class HighDensityEllipsoids:
    def __init__(self, X, X_densities, cluster_probs, means, covariances, density_threshold=None):
        self.X = X
        self.X_densities = X_densities
        self.density_threshold = density_threshold if density_threshold is not None else float("-inf")
        self.cluster_probs = cluster_probs
        self.means = means
        self.covariances = covariances
        self.t = 0.9
        self.epsilon = 0#1e-5

    def compute_ellipsoids(self):        
        return self.build_solve_opt()
    
    def _solve(self, prob):
        prob.solve(solver=cp.SCS, verbose=False)

    def build_solve_opt(self):
        n_ellipsoids = self.cluster_probs.shape[1]
        n_samples = self.X.shape[0]
        
        # Variables
        r = cp.Variable(n_ellipsoids, pos=True)

        # Construct constraints
        constraints = []
        for i in range(n_ellipsoids):
            mu_i = self.means[i]
            cov_i = np.linalg.inv(self.covariances[i])

            for j in range(n_samples):
                if self.X_densities[j][i] <= self.density_threshold:  # At least as good as a requested NLL
                    x_j = X_[j,:]
                    
                    a = (x_j - mu_i)
                    b = np.dot(a, np.dot(cov_i, a))
                    constraints.append(b <= r[i])

        # Build the final program
        f = cp.Minimize(cp.sum(r))
        prob = cp.Problem(f, constraints)

        # Solve it!
        self._solve(prob)

        return r.value


class Ellipsoids:
    def __init__(self, means, covariances, r, density_estimator):
        self.means = means
        self.covariances = [np.linalg.inv(cov) for cov in covariances]
        self.r = r
        self.n_ellipsoids = self.r.shape[0]
        self.density_estimator = density_estimator
    
    def score_samples(self, X):
        print(X.shape)
        pred = []

        pred = self.density_estimator.score_samples(X) >= -10

        return np.array(pred).astype(np.int)


if __name__ == "__main__":
    # Load/Create data
    pca_dim = 40

    X, y = load_digits(return_X_y=True) 
    X, y = shuffle(X, y, random_state=42)

    # k-fold cross validation
    scores_with_density_constraint = []
    scores_without_density_constraint = []

    original_data = []
    original_data_labels = []
    cfs_with_density_constraint = []
    cfs_without_density_constraint = []
    cfs_target_label = []
    closest_pn = []
    closest_pn_label = []

    kf = KFold(n_splits=4)
    for train_index, test_index in kf.split(X):
        # Split data into training and test set
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # If requested: Reduce dimensionality
        X_train_orig = np.copy(X_train)
        X_test_orig = np.copy(X_test)
        projection_matrix = None
        projection_mean_sub = None
        pca = None
        if pca_dim is not None:
            pca = PCA(n_components=pca_dim)
            pca.fit(X_train)

            projection_matrix = pca.components_ # Projection matrix
            projection_mean_sub = pca.mean_

            X_train = np.dot(X_train - projection_mean_sub, projection_matrix.T)
            X_test = np.dot(X_test - projection_mean_sub, projection_matrix.T)


        # Fit classifier
        model = LogisticRegression(multi_class="multinomial", solver="lbfgs")
        model.fit(X_train, y_train)

        # Compute accuracy on test set
        print("Accuracy: {0}".format(accuracy_score(y_test, model.predict(X_test))))

        # For each class, fit density estimators
        density_estimators = {}
        kernel_density_estimators = {}
        labels = np.unique(y)
        for label in labels:
            # Get all samples with the 'correct' label
            idx = y_train == label
            X_ = X_train[idx, :]

            # Optimize hyperparameters
            cv = GridSearchCV(estimator=KernelDensity(), iid=False, param_grid={'bandwidth': np.arange(0.1, 10.0, 0.05)}, n_jobs=-1, cv=5)
            cv.fit(X_)
            bandwidth = cv.best_params_["bandwidth"]
            print("bandwidth: {0}".format(bandwidth))

            cv = GridSearchCV(estimator=GaussianMixture(covariance_type='full'), iid=False, param_grid={'n_components': range(2, 10)}, n_jobs=-1, cv=5)
            cv.fit(X_)
            n_components = cv.best_params_["n_components"]
            print("n_components: {0}".format(n_components))

            # Build density estimators
            kde = KernelDensity(bandwidth=bandwidth)
            kde.fit(X_)

            de = GaussianMixture(n_components=n_components, covariance_type='full')
            de.fit(X_)

            density_estimators[label] = de
            kernel_density_estimators[label] = kde

        # For each point in the test set
        # Compute and plot without density constraints
        print("n_test_samples: {0}".format(X_test.shape[0]))
        for i in range(X_test.shape[0]):
            x_orig = X_test[i,:]
            x_orig_orig = X_test_orig[i,:]
            y_orig = y_test[i]
            y_target = y_orig

            if(model.predict([x_orig]) != y_target):  # Misslcassification of the original sample!
                continue

            # Compute and plot WITH kernel density constraints
            idx = y_train == y_target   # NOTE: -1 encodes class 0
            X_ = X_train[idx, :]

            # Build density estimator
            de = density_estimators[y_target]
            kde = kernel_density_estimators[y_target]

            from scipy.stats import multivariate_normal # TODO: Move this to the outer loop where the density estimators are fitted!
            densities_training_samples = []
            densities_training_samples_ex = []
            for j in range(X_.shape[0]):
                x = X_[j,:]
                z = []
                dim = x.shape[0]
                for i in range(de.weights_.shape[0]):
                    x_i = de.means_[i]
                    w_i = de.weights_[i]
                    cov = de.covariances_[i]
                    cov = np.linalg.inv(cov)

                    b = -2.*np.log(w_i) + dim*np.log(2.*np.pi) - np.log(np.linalg.det(cov))
                    z.append(np.dot(x - x_i, np.dot(cov, x - x_i)) + b) # NLL
                densities_training_samples.append(np.min(z))
                densities_training_samples_ex.append(z)
            densities_training_samples = np.array(densities_training_samples)
            densities_training_samples_ex = np.array(densities_training_samples_ex)

            # Compute soft cluster assignments
            cluster_prob_ = de.predict_proba(X_)
            X_densities = de.score_samples(X_)
            density_threshold = np.median(densities_training_samples)
            r = HighDensityEllipsoids(X_, densities_training_samples_ex, cluster_prob_, de.means_, de.covariances_, density_threshold).compute_ellipsoids()

            cf = None
            cf = FeasiblePertinentPositive(model.coef_, model.intercept_, X=X_, density_constraint=False, ellipsoids_r=r, gmm_weights=de.weights_, gmm_means=de.means_, gmm_covariances=de.covariances_, projection_matrix=projection_matrix, projection_mean_sub=projection_mean_sub)
            xcf = cf.compute_counterfactual(x_orig_orig, y=y_target)

            if xcf is None:
                print("No pertinent positive found!")
                continue
            xcf_transformed = [xcf] if pca_dim is None else pca.transform([xcf])
            if model.predict(xcf_transformed) != y_target:
                print("Wrong prediction on pertinent positive")
                continue
            else:
                print("Correct prediction on pertinent positive")

            cf2 = None
            cf2 = FeasiblePertinentPositive(model.coef_, model.intercept_, X_, ellipsoids_r=r, gmm_weights=de.weights_, gmm_means=de.means_, gmm_covariances=de.covariances_, projection_matrix=projection_matrix, projection_mean_sub=projection_mean_sub, density_threshold=density_threshold)
            xcf2 = cf2.compute_counterfactual(x_orig_orig, y=y_target)
            if xcf2 is None:
                print("No pertinent positive found!")
                continue

            # Compute closest pertinent negative
            y_pn_target = 0 if y_orig != 0 else 1
            pn = FeasibleCounterfactual(model.coef_, model.intercept_, X=X_, density_constraint=False, ellipsoids_r=r, gmm_weights=de.weights_, gmm_means=de.means_, gmm_covariances=de.covariances_, projection_matrix=projection_matrix, projection_mean_sub=projection_mean_sub)
            xpn = pn.compute_counterfactual(x_orig_orig, y=y_pn_target)

            # Save
            original_data.append(x_orig_orig)
            original_data_labels.append(y_orig)
            cfs_with_density_constraint.append(xcf2)
            cfs_without_density_constraint.append(xcf)
            cfs_target_label.append(y_target)
            closest_pn.append(xpn)
            closest_pn_label.append(y_pn_target)

            if pca is not None:
                xcf = pca.transform([xcf])
                xcf2 = pca.transform([xcf2])

            # Evaluate
            scores_without_density_constraint.append(kde.score_samples(xcf.reshape(1, -1)))
            scores_with_density_constraint.append(kde.score_samples(xcf2.reshape(1, -1)))
    
        print("Without density constrain: Median: {0} Mean: {1} Var: {2}".format(np.median(scores_without_density_constraint), np.mean(scores_without_density_constraint), np.var(scores_without_density_constraint)))
        print("With density constrain: Median: {0} Mean: {1} Var: {2}".format(np.median(scores_with_density_constraint), np.mean(scores_with_density_constraint), np.var(scores_with_density_constraint)))
