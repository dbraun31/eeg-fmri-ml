import numpy as np
import pickle
import re
import os
from glob import glob
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
    Compare multiple machine learning estimators across one or more subjects 
    by tuning their hyperparameters and evaluating their performance.

    This class automates hyperparameter tuning using cross-validation on 
    session 1 data, applies the tuned models to held-out data for final 
    evaluation, and logs detailed summaries of results for each estimator 
    and subject.

    Parameters
    ----------
    estimators : dict
        Dictionary mapping model names to their corresponding estimator 
        classes or functions.
        Example: {'ridge': Ridge, 'lasso': Lasso}

    param_grids : dict
        Dictionary of hyperparameter grids for each estimator. Keys must match 
        the estimator names. Each value is a dict mapping parameter names to 
        lists of possible values.
        Example: {'ridge': {'alpha': [0.1, 1, 10]}, 
                  'lasso': {'alpha': [0.01, 0.1]}}

    subject : str, int, or None, optional
        Identifier for the subject whose data will be used for training 
        and evaluation. If None, all subjects in `data_path` will be processed.

    fixed_params_dict : dict, optional
        Dictionary mapping estimator names to fixed parameters that should 
        be passed to the estimator along with the hyperparameters. Defaults to None.

    data_path : str or pathlib.Path, optional
        Path to the directory containing formatted data files. Defaults to 
        'data/formatted'.

    scoring : callable, optional
        Scoring function to evaluate model performance. Should accept true and 
        predicted values as arguments and return a scalar score. Defaults to RMSE 
        (root mean squared error).

    log_dir : str or pathlib.Path, optional
        Directory path where log files will be saved. Defaults to 
        'scripts/logs/model_compare'.

    groups : str, optional
        Placeholder for future grouping strategies in cross-validation. Currently unused.

    verbose : int, optional
        Verbosity level. If > 0, progress and intermediate results will be printed.

    Attributes
    ----------
    best_model_ : tuple
        The (estimator_name, score) of the best performing estimator found during 
        tuning for the current subject.

    data : dict
        Loaded data for the current subject.

    time : str
        Timestamp string generated at initialization, used to name log files.

    Methods
    -------
    run()
        Executes hyperparameter tuning for all estimators across one or all subjects, 
        evaluates their performance, logs results, and returns a dictionary 
        mapping subjects to lists of (estimator_name, final_score) tuples.
    """

    def __init__(self, estimators, param_grids, 
                 subject=None,
                 fixed_params_dict = None,
                 data_path=None,
                 scoring=None,
                 log_dir=None,
                 groups='channels', # For future use
                 bads=None,
                 verbose=0): 

        if len(estimators) != len(param_grids):
            raise RuntimeError('Number of estimators must equal number of '
                                'parameter grids')

        # Bads from preprocessing observation misalignment
        self.bads = [1, 11, 13, 19, 23, 25]

        self.estimators = estimators
        self.param_grids = param_grids
        self.fixed_params_dict = fixed_params_dict
        self.scoring = scoring or (lambda x, y: np.sqrt(np.mean((x - y)**2)))
        self.subject = None
        if subject is not None:
            self.subject = self._parse_subject(subject)
        self.subjects = None
        self.data_path = Path(data_path or 'data/formatted')
        self.verbose = verbose
        self.best_model_ = None
        self.data = None
        now = datetime.now()
        self.time = now.strftime('%Y%m%d%H%M%S')
        log_dir = Path(log_dir or 'scripts/logs/model_compare')
        # Set up log directories
        self.log_dir_txt = log_dir / Path('txt')
        self.log_dir_txt.mkdir(parents=True, exist_ok=True)
        self.log_dir_pkl = log_dir / Path('pkl')
        self.log_dir_pkl.mkdir(parents=True, exist_ok=True)

        # Process one subject or all subjects
        if self.subject is not None:
            self.data = self._import_data(self.data_path, self.subject)
        else:
            self.subjects = [Path(x).stem for x in glob(str(self.data_path) + '/*')]
            if all([type(x) is int for x in bads]):
                bads = ['sub-' + str(x).zfill(3) for x in bads]
            elif any([not bool(re.fullmatch(r'sub-\d{3}', x)) for x in bads]):
                raise ValueError('subjects must all be either integers or in form (eg) sub-001')

            self.subjects = [x for x in self.subjects if x not in bads]
            self.subjects = sorted(self.subjects, key=lambda x: int(re.search(r'\d+', x).group()))


    def run(self):
        
        if self.subject is not None:
            result = {self.subject: self._eval_subject(self.subject)}
        else:
            result = {}

            for i, subject_raw in enumerate(self.subjects, start=1):
                print('\n' + '/' * 2 + ' ' * 3 + 
                      f'PROCESSING SUBJECT {subject_raw}: {i}/{len(self.subjects)}' + 
                      ' ' * 3 + '/' * 2 + '\n')
                subject = self._parse_subject(subject_raw)
                self.data = self._import_data(self.data_path, subject)
                result[subject] = self._eval_subject(subject)

        # Write result
        try: 
            subject_key = 'AllSubjects' if self.subject is None else self.subject
            log_file = Path(f'{self.time}_{subject_key}_ModelCompare.pkl')
            with open(self.log_dir_pkl / log_file, 'wb') as file:
                pickle.dump(result, file)
        except Exception as exception:
            print('Writing final result failed!')
            print(f'With error: {exception}')

        return result


    def _eval_subject(self, subject):
        
        # List of (model name, score) tuples
        scores = []

        self._write_log(subject=subject)

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
            y = np.concatenate([self.data['ses-001'][x]['y']['dmn_a'] for x in runs])
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
            # Combine params
            params = cv.best_params_
            if fixed_params is not None:
                params = cv.best_params_ | fixed_params
            print(f'PARAMS: {params}')
            score =  get_final_score(estimator, params, self.data, self.scoring)
            scores.append((estimator_name, score))

            self._write_log(estimator_name, cv, score)

        # Sort and return
        scores.sort(key=lambda x: x[1])
        self.best_model_ = scores[0]

        return scores


    def _parse_subject(self, subject_input):
        # Deal with many types of subject input
        subject_match = re.search(r'\d+', subject_input)
        if subject_match is None:
            raise ValueError('subject must contain a number')

        subject = 'sub-' + str(int(subject_match.group())).zfill(3)
        return subject

    def _import_data(self, data_path, subject):

        data_file = data_path / Path(subject).with_suffix('.pkl')
        with open(data_file, 'rb') as file:
            data = pickle.load(file)

        return data


    def _write_log(self, estimator_name=None, cv=None, score=None,
                   subject=None):


        def writer(summary):
            # Append to log
            log_file = self.log_dir_txt / Path(f'ModelCompare_{self.time}.txt')
            with open(log_file, 'a') as f:
                for line in summary:
                    f.write(line + '\n')

        summary = []

        if subject is not None:
            summary.append('-' * 20 + '\n')
            summary.append(' ' * 10 + f'SUBJECT: {subject}\n')
            summary.append('-' * 20 + '\n')
            writer(summary)
            return None


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

        writer(summary)

