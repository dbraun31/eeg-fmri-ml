import numpy as np
from itertools import product
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
import os
from pathlib import Path

def get_cv_splits(session):
    '''
     Takes in session data {'run1': {'X': ..., 'y': ...}, ...}
     Returns cv_splits
       [(train_idx1, test_idx1), (train_idx2, test_idx2), ...]
    '''

    # Concatenate data
    runs = list(session.keys())
    X = np.concatenate([session[x]['X'] for x in runs])
    y = np.concatenate([session[x]['y'] for x in runs])

    # Get all run indices
    run_idxs = {}
    start = 0
    for run in runs:
        stop = session[run]['X'].shape[0] 
        run_idxs[run] = np.array(range(start, start+stop))
        start += stop

    cv_splits = []
    for run in runs:
        test_idx = run_idxs[run]
        train_idx = np.setdiff1d(np.arange(X.shape[0]), test_idx)
        cv_splits.append((train_idx, test_idx))

    return cv_splits


class GridSearchCV(BaseEstimator):

    def __init__(self, estimator, param_grid, cv_splits, 
                 fixed_params=None,
                 scoring=None,
                 log_path=None,
                 verbose=0):

        self.estimator = estimator
        self.param_grid = param_grid
        self.cv_splits = cv_splits
        self.fixed_params = fixed_params or {}
        # (y_test, y_preds)
        if scoring is None:
            self.scoring = mean_squared_error
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
        self.log_path = Path(log_path or 'scripts/logs/GridSearchCV')
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.log = []


    def fit(self, X, y):

        # Do single pass CV if no grid
        if self.param_grid is None:
            all_params = [None]
        else:
            all_params = list(product(*self.param_grid.values()))
            param_names = list(self.param_grid.keys())

        best_params = None
        best_score = np.inf
        best_estimator = None

        for param_set in all_params:
            # {'param_name', param_value}
            if param_set is not None:
                params = dict(zip(param_names, param_set))
            else:
                params = {}
            fold_scores = []

            print(f'Param set {all_params.index(param_set)+1} / {len(all_params)}')

            for run, (train_idx, test_idx) in enumerate(self.cv_splits, start=1):

                # Train test split
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]

                # Rescale
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Fit predict
                model = self.estimator(**params, **self.fixed_params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Score fold
                score = self.scoring(y_test, y_pred)
                fold_scores.append(self.scoring(y_test, y_pred))

                if self.verbose:
                    status0 = f'Held out run: {run}'
                    status1 = f'Params: {params}\nFold score: {score:.7f}'
                    status2 = ''
                    if hasattr(model, 'coef_'):
                        status2 = f'Nonzero coefficients: {np.sum(model.coef_ != 0)}'
                    status = status0 + '\n' + status1 + '\n' + status2 + '\n' 
                    print(status)
                    self.log.append(status)



            # Param results
            avg_score = np.mean(fold_scores)
            self.results_['params'].append(params) 
            self.results_['mean_test_score'].append(avg_score) 
            self.results_['std_test_score'].append(np.std(fold_scores))
            self.results_['split_test_score'].append(fold_scores)


            if avg_score < best_score:
                best_score = avg_score
                best_estimator = model
                best_params = params

        self.best_score_ = best_score
        # Fit best estimator on full data
        best_estimator = self.estimator(**best_params)
        best_estimator.fit(X, y)
        self.best_estimator_ = best_estimator
        self.best_params_ = best_params

        if self.verbose:
            print(f'Best params: {best_params} with score {best_score: .4f}')


        # Write log
        log_file = self.log_path / Path(f'GSCV_log_{self.time}.txt')
        with open(log_file, 'w') as file:
            for item in self.log:
                file.write(item + '\n')

        return self

    def predict(self, X):
        if self.best_estimator_ is None:
            raise RuntimeError('You must first call fit() before predict()')
        preds = self.best_estimator_.predict(X)
        return preds



def get_final_score(estimator, params, data, scoring):
    # Get averaged session score

    scores = []

    sessions = list(data.keys())

    for train_session in sessions:
        # Get data
        test_session = [s for s in sessions if s != train_session][0]
        runs = list(data[train_session].keys())
        X_train = np.concatenate([data[train_session][x]['X'] for x in runs])
        y_train = np.concatenate([data[train_session][x]['y'] for x in runs])
        X_test = np.concatenate([data[test_session][x]['X'] for x in runs])
        y_test = np.concatenate([data[test_session][x]['y'] for x in runs])

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Fit predict score
        model = estimator(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = scoring(y_test, y_pred)
        scores.append(score)

    return np.mean(scores)


