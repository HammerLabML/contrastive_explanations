import numpy as np
import cvxpy as cp


class FeasibleCounterfactual:
    def __init__(self, w, b, X, ellipsoids_r, gmm_weights, gmm_means, gmm_covariances, projection_matrix=None, projection_mean_sub=None, density_constraint=True, density_threshold=-85):
        self.w = w
        self.b = b

        self.kernel_var = 0.2#0.5   # Kernel density estimator
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
        mad = None
        if regularizer == "l1":
            mad = np.ones(x.shape[0])
        
        xcf = None
        s = float("inf")
        for i in range(self.gmm_weights.shape[0]):
            try:
                self.gmm_cluster_index = i
                xcf_ = self.build_solve_opt(x, y, mad)
                if xcf_ is None:
                    continue

                s_ = None
                if regularizer == "l1":
                    s_ = np.sum(np.abs(xcf_ - x))
                else:
                    s_ = np.linalg.norm(xcf_ - x, ord=2)

                if s_ <= s:
                    s = s_
                    xcf = xcf_
            except Exception as ex:
                print(ex)
        return xcf
    
    def _solve(self, prob):
        prob.solve(solver=cp.SCS, verbose=False)

    def build_solve_opt(self, x_orig, y, mad=None):
        dim = x_orig.shape[0]
     
        # Variables
        x = cp.Variable(dim)
        beta = cp.Variable(dim)

        # Constants
        c = np.ones(dim)
        z = np.zeros(dim)
        I = np.eye(dim)

        # Construct constraints
        constraints = self._build_constraints(x, y)

        if self.density_constraint is True:
            i = self.gmm_cluster_index
            x_i = self.gmm_means[i]
            cov = self.gmm_covariances[i]
            cov = np.linalg.inv(cov)

            constraints += [cp.quad_form(self.projection_matrix @ (x - self.projection_mean_sub) - x_i, cov) - self.ellipsoids_r[i] <= 0] # Numerically much more stable than the explicit density omponent constraint

        # If necessary, construct the weight matrix for the weighted Manhattan distance
        Upsilon = None
        if mad is not None:
            alpha = 1. / mad
            Upsilon = np.diag(alpha)

        # Build the final program
        f = None
        if mad is not None:
            f = cp.Minimize(c.T @ beta)    # Minimize (weighted) Manhattan distance
            constraints += [Upsilon @ (x - x_orig) <= beta, (-1. * Upsilon) @ (x - x_orig) <= beta, I @ beta >= z]
        else:
            f = cp.Minimize((1/2)*cp.quad_form(x, I) - x_orig.T@x)  # Minimize L2 distance
        
        prob = cp.Problem(f, constraints)

        # Solve it!
        self._solve(prob)

        return x.value
