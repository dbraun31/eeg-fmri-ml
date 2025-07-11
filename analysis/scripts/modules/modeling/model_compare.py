import numpy as np
import pickle
import re
import os
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from scripts.modules.modeling.grid_search import (
        get_cv_splits,
        GridSearchCV,
        get_final_score
)



class ModelCompare:
    """
    Compare multiple machine learning estimators by tuning their hyperparameters
    on a specified dataset and evaluating their performance.

    This class automates the process of hyperparameter tuning using cross-validation 
    on session 1 data, applies the tuned models to held-out data for final evaluation, 
    and logs detailed summaries of results for each estimator.

    Parameters
    ----------
    estimators : dict
        Dictionary of model names and their corresponding estimator classes or functions.
        Example: {'ridge': Ridge, 'lasso': Lasso}
    
    param_grids : dict
        Dictionary of hyperparameter grids for each estimator. Keys should match
        the estimator names. Each value is itself a dict mapping parameter names
        to lists of possible values.
        Example: {'ridge': {'alpha': [0.1, 1, 10]}, 'lasso': {'alpha': [0.01, 0.1]}}

    subject : str or int
        Identifier for the subject whose data will be loaded and used for model training 
        and evaluation.

    fixed_params_dict : dict, optional
        Optional dictionary mapping estimator names to fixed parameters that should be 
        passed to the estimator along with the hyperparameters being tuned. Defaults to None.

    data_path : str or pathlib.Path, optional
        Path to the directory containing formatted data files. If not provided, defaults 
        to 'data/formatted'.

    scoring : callable, optional
        Scoring function to evaluate model performance. Should accept true and predicted 
        values as arguments and return a scalar score. Defaults to RMSE (root mean squared error).

    log_dir : str or pathlib.Path, optional
        Directory path where log files will be saved. Defaults to 'scripts/logs/model_compare'.

    groups : str, optional
        Placeholder for future grouping strategies in cross-validation. Currently unused.

    verbose : int, optional
        Verbosity level. If > 0, progress and intermediate results will be printed.

    Attributes
    ----------
    best_model_ : estimator
        The best performing estimator found during tuning across all models.

    data : dict
        Loaded data for the specified subject.

    time : str
        Timestamp string generated at initialization, used to name log files.

    Methods
    -------
    run()
        Performs hyperparameter tuning for all estimators, evaluates their final performance,
        logs results, and returns a list of (estimator_name, final_score) tuples.
    """

    def __init__(self, estimators, param_grids, subject,
                 fixed_params_dict = None,
                 data_path=None,
                 scoring=None,
                 log_dir=None,
                 groups='channels', # For future use
                 verbose=0): 

        if len(estimators) != len(param_grids):
            raise RuntimeError('Number of estimators must equal number of '
                                'parameter grids')

        self.estimators = estimators
        self.param_grids = param_grids
        self.fixed_params_dict = fixed_params_dict
        self.scoring = scoring or (lambda x, y: np.sqrt(np.mean((x - y)**2)))
        self.subject = subject
        data_path = Path(data_path or 'data/formatted')
        self.data = self._import_data(data_path)
        self.verbose = verbose
        self.best_model_ = None
        now = datetime.now()
        self.time = now.strftime('%Y%m%d%H%M%S')
        self.log_dir = Path(log_dir or 'scripts/logs/model_compare')
        self.log_dir.mkdir(parents=True, exist_ok=True)



    def run(self):
        
        # List of (model name, score) tuples
        scores = []

        for estimator_name in self.estimators:

            # Get estimator and params
            estimator = self.estimators[estimator_name]
            param_grid = self.param_grids[estimator_name]
            fixed_params = None
            if type(self.fixed_params_dict) is dict:
                fixed_params = self.fixed_params_dict.get(estimator_name, None)

            # Format data
            runs = list(self.data['ses-001'].keys())
            X = np.concatenate([self.data['ses-001'][x]['X'] for x in runs])
            y = np.concatenate([self.data['ses-001'][x]['y'] for x in runs])
            cv_splits = get_cv_splits(self.data['ses-001'])

            # Grid search
            print('-' * 20)
            print(f'\nInitiating grid search for {estimator_name}.\n')
            cv = GridSearchCV(estimator=estimator,
                              param_grid=param_grid,
                              cv_splits=cv_splits,
                              fixed_params=fixed_params, 
                              scoring=self.scoring,
                              verbose=self.verbose)
            cv.fit(X, y)
            score =  get_final_score(estimator, cv.best_params_, self.data, self.scoring)
            scores.append((estimator_name, score))

            self._write_log(estimator_name, cv, score)

        # Sort and return
        scores.sort(key=lambda x: x[1])
        self.best_model_ = scores[0]

        return scores



    def _import_data(self, data_path):
        
        # Deal with many types of subject input
        subject = re.search(r'\d+', self.subject).group()
        if subject is None:
            raise ValueError('subject must contain a number')

        subject = Path('sub-' + str(int(subject)).zfill(3))

        data_file = data_path / subject.with_suffix('.pkl')
        with open(data_file, 'rb') as file:
            data = pickle.load(file)

        return data


    def _write_log(self, estimator_name, cv, score):
        summary = []
        summary.append('=' * 20)
        summary.append(f'Estimator: {estimator_name}')
        summary.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f'Best parameters: {cv.best_params_}')
        summary.append(f'Best CV score: {cv.best_score_:.4f}')
        summary.append(f'Final held-out score: {score:4f}')
        
        summary.append('\nAll CV results:')
        n_results = len(cv.results_['params'])
        for i in range(n_results):
            summary.append(
                f"- Params: {cv.results_['params'][i]}, "
                f"Mean CV score: {cv.results_['mean_test_score'][i]:.4f}, "
                f"Std CV score: {cv.results_['std_test_score'][i]:.4f}"
            )

        if hasattr(cv.best_estimator_, 'coef_'):
            n_nonzero = np.sum(cv.best_estimator_.coef_ != 0)
            summary.append(f'Nonzero coefficients (best model): {n_nonzero} / {cv.best_estimator_.coef_.size}')

        summary.append('=' * 20)

        # Append to log
        log_file = self.log_dir / Path(f'ModelCompare_{self.time}.txt')
        with open(log_file, 'a') as f:
            for line in summary:
                f.write(line + '\n')


