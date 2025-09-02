import os
import numpy as np
from pathlib import Path
from itertools import product
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


# dev
from types import SimpleNamespace
self = SimpleNamespace()

class BayesCV:

    """
    Bayesian cross-validation wrapper for hyperparameter optimization.

    This class performs Bayesian optimization over a hyperparameter grid
    for a given estimator using Gaussian Process regression to model the
    cross-validation loss. It is designed to work with any scikit-learn–
    style estimator and supports custom scoring functions.

    Parameters
    ----------
    estimator : callable
        The estimator class (not an instance) to optimize. Must support
        `.fit()` and `.predict()` methods with keyword hyperparameters.

    param_grid : dict
        Dictionary with hyperparameter names as keys and iterables of
        values to explore as values.

    limits : dict
        Dictionary with the same keys as `param_grid` specifying the
        minimum and maximum values and search space type for each parameter:
        {'min': float, 'max': float, 'space': 'linear' or 'log'}.

    cv_splits : list of tuple
        List of `(train_idx, test_idx)` tuples specifying cross-validation
        splits.

    fixed_params : dict, optional (default=None)
        Hyperparameters to pass to the estimator that are fixed and not
        subject to optimization.

    scoring : callable, optional (default=mean_squared_error)
        Function with signature `scoring(y_true, y_pred)` returning a
        scalar loss to minimize.

    verbose : int, default=0
        Verbosity level. Higher values produce more output during fitting.

    Attributes
    ----------
    best_params_ : dict
        Best hyperparameters found during optimization.

    best_score_ : float
        Cross-validation score of the best hyperparameters.

    best_estimator_ : estimator
        Unfitted estimator initialized with best hyperparameters and
        fixed parameters.

    fitted_estimator_ : estimator
        Estimator fitted on the full dataset using best hyperparameters.

    results_ : dict
        Dictionary storing parameter combinations and corresponding CV scores.

    time : str
        Timestamp of instantiation for logging purposes.

    log_path : Path
        Directory path where logs are stored.

    param_names : list
        List of hyperparameter names in order.

    X_hyper_ : np.ndarray
        Hyperparameter combinations evaluated by the Gaussian Process.

    y_loss_ : list
        Corresponding cross-validation losses for evaluated hyperparameters.

    Notes
    -----
    - Attributes ending with an underscore (_) are only available after
      calling `fit`.
    - This class assumes that the estimator can accept hyperparameters
      as keyword arguments.
    - The `_fit_gp` method uses a Gaussian Process with RBF + WhiteKernel
      and iteratively proposes new hyperparameters using an upper
      confidence bound criterion.
    """

    def __init__(self, estimator, param_grid, limits, cv_splits,
                 fixed_params=None,
                 scoring=None,
                 log_path=None,
                 verbose=0):

        if param_grid.keys() != limits.keys():
            raise ValueError('param_grid and limits need to have same keys in same order')

        self.estimator = estimator
        self.param_grid = param_grid
        # Min, max, space-type limits for param grid (dict)
        # Order of key definition needs to be same as param_grid!
        self.limits = limits
        self.cv_splits = cv_splits
        self.fixed_params = fixed_params or {}
        if scoring is None:
            self.scoring = mean_squared_error
        else:
            self.scoring = scoring
        self.verbose = verbose
        self.results_ = {
                'params': [],
                'mean_test_score': [],
                'std_test_score': [],
                'split_test_score': []
                }
        now = datetime.now()
        self.time = now.strftime('%Y%m%d%H%M%S')
        self.log_path = Path(log_path or 'scripts/logs/BayesCV')
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.param_names = list(param_grid.keys())
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.log = []

    def fit(self, X, y):

        self.X = X
        self.y = y

        if self.param_grid is None:
            raise ValueError('Parameter grid is needed for BayesCV')

        param_names = self.param_names
        X_hyper = np.array(list(product(*self.param_grid.values())))
        y_loss = []

        # Get initiaul observations
        print(f'Gathering initial observations')
        for i, x_hyper in enumerate(X_hyper, start=1):

            if self.verbose:
                print(f'Parameter combination {i} of {len(X_hyper)}\n')
            y_loss.append(self._cross_val(x_hyper))

        X_final = self._fit_gp(X_hyper, y_loss)
        self.best_params_ = dict(zip(param_names, X_final))
        self.best_estimator_ = self.estimator(**self.best_params_, **self.fixed_params)
        self.fitted_estimator_ = self.best_estimator_.fit(X, y)
        self.nonzero_coef_ = None
        if hasattr(self.fitted_estimator_, 'coef_'):
            coefs = self.fitted_estimator_.coef_
            self.nonzero_coef_ = sum(coefs != 0)

    def predict(self, X):

        out = self.fitted_estimator_.predict(X)
        return out
        

    def _cross_val(self, x_hyper):
        mean_loss = 0
        params = dict(zip(self.param_names, x_hyper))
        X = self.X
        y = self.y

        for run, (train_idx, test_idx) in enumerate(self.cv_splits, start=1):

            # Train test split
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Scale
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Fit predict
            model = self.estimator(**params, **self.fixed_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            loss = self.scoring(y_test, y_pred)
            mean_loss += loss * (1/len(self.cv_splits))

        return mean_loss


    def _fit_gp(self, X_hyper, y_loss, k=.01, max_iter=5,
                optimum_type='empirical'):

        # Init GP
        kernel = ConstantKernel(1) * RBF(1) + WhiteKernel(1)
        gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                normalize_y=True)

        # Optimize
        if self.verbose:
            print('Performing Bayesian optimization')
        for iteration in range(max_iter):
            if self.verbose:
                print(f'Iteration {iteration+1}/{max_iter}\n')

            # Fit GP to observations
            gp.fit(X_hyper, y_loss)

            # Specify domain
            domain = {}
            for param in self.limits:
                domain[param] = self._get_domain(self.limits[param])
            X_domain = np.column_stack(list(product(*domain.values()))).transpose()

            # Predict from multivariate normal
            mu, sigma = gp.predict(X_domain, return_std = True)

            # Find best next hyper parameters
            ucb = mu - sigma * k
            idx = np.where(ucb == np.min(ucb))[0][0]
            x_next = X_domain[idx]

            # Obtain the observation
            loss = self._cross_val(x_next)

            # Update
            X_hyper = np.concatenate([X_hyper, x_next.reshape(-1, 2)], axis=0)
            y_loss.append(loss)

        self.X_hyper_ = X_hyper
        self.y_loss_ = y_loss

        if optimum_type == 'empirical':
            return X_hyper[np.where(y_loss == np.min(y_loss))[0][0], :]
        elif optimum_type == 'theoretical':
            return X_domain[np.where(mu == np.min(mu))[0][0], :]


    def _get_domain(self, limit, steps=100):
        # Takes in limit dict for single param
        # Returns 1D array

        if limit['space'] not in ['linear', 'log']:
            raise ValueError('space arg of limits must be "linear" or "log"')

        if limit['space'] == 'linear':
            return np.linspace(limit['min'], limit['max'], steps)

        return np.logspace(limit['min'], limit['max'], steps)



