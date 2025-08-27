import os
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

# Dev
import pickle
from types import SimpleNamespace
from scripts.modules.modeling.grid_search import get_cv_splits
from groupyr import SGL
from scripts.modules.modeling.groups import get_groups
from pyprojroot import here
os.chdir(here() / Path('analysis'))


dpath = Path('data/formatted/sub-002.pkl')

with open(dpath, 'rb') as file:
    d = pickle.load(file)

cv_splits = get_cv_splits(d['ses-001'])

runs = list(d['ses-001'].keys())
X = np.concatenate([d['ses-001'][x]['X'] for x in runs])
y = np.concatenate([d['ses-001'][x]['y']['DNa'] for x in runs])
fixed_params = {'groups': get_groups()}
param_grid = {'l1_ratio': [.2, .5, .8],
              'alpha': [1e-4, 1e-2, 1]}
limits = {'l1_ratio': {'min': 0, 'max': 1, 'space': 'linear'},
          'alpha': {'min': -4, 'max': 1, 'space': 'log'}}


class BayesCV:

    def __init__(self, estimator, param_grid, limits, cv_splits,
                 fixed_params=None,
                 scoring=None,
                 verbose=0):

        if param_grid.keys() != limits.keys():
            raise ValueError('param_grid and limits need to have same keys in same order')

        self.estimator = estimator
        self.param_grid = param_grid
        # Min, max, space-type limits for param grid (dict)
        # Order of key definition needs to be same as param_grid!
        self.limits = limits
        self.cv_splits = cv_splits
        self.fixed_params = fixed_params
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
        self.best_estimator_ None
        self.log = []

    def fit(self, X, y):

        self.X = X
        self.y = y

        if self.param_grid is None:
            raise ValueError('Parameter grid is needed for BayesCV')

        param_names = self.param_names
        X_hyper = np.array(list(product(*self.param_grid.values())))

        # Get initiaul observations
        for x_hyper in X_hyper:

            y_loss = self._cross_val(x_hyper)

        X_final = self._fit_gp(X_hyper, y_loss)


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


        def _fit_gp(self, X_hyper, y_loss, k=2, max_iter=5,
                    optimum_type='empirical'):

            # Init GP
            kernel = ConstantKernel(1) * RBF(1) + WhiteKernel(1)
            gp = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=5,
                    normalize_y=True)

            # Optimize
            for iteration in range(max_iter):
                print(f'Iteration {iteration+1}\n')

                # Fit GP to observations
                gp.fit(X_hyper, y_loss)

                # Specify domain
                domain = {}
                for param in self.limits:
                    domain[param] = self._get_domain(self.limits[param])
                X_domain = np.column_stack(list(product(*domain.values()))).transpose()

                k = .01

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




