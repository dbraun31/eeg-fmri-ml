import mne
import os
from pyprojroot import here
from pathlib import Path
import pickle
import numpy as np
np.int = int
import pandas as pd
from glob import glob
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, get_scorer
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
os.chdir(here() / Path('analysis'))
from scripts.modules.modeling.model_selection import train_test_split
from scripts.modules.modeling.model_compare import ModelCompare, get_final_score
from scripts.modules.modeling.baseline import BaselineModel
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scripts.modules.preprocessing.cv_prep import get_cv_splits_ar


root = Path('scripts/sandbox/run_ml')
droot = Path('data/formatted')

subjects = [Path(x).stem for x in glob(str(droot / Path('*')))]
subjects = sorted(subjects, key=lambda x: int(x.split('-')[1]))
subjects = [x for x in subjects if x != 'sub-023']

result = pd.DataFrame()

for subject in subjects:
    print(f'\n---PROCESSING SUBJECT {subject} of {subjects[-1]}---\n')

    dpath = droot / Path(f'{subject}.pkl')

    with open(dpath, 'rb') as file:
        d = pickle.load(file)

    cv_splits = get_cv_splits_ar(d['ses-001'])
    X_train, y_train, X_test, y_test = train_test_split(d)



    def rmse(y_test, y_pred):
        return np.sqrt(sum((y_pred - y_test)**2))

    scoring = make_scorer(rmse, greater_is_better=False)

# --- BASELINE --- #
    bm = BaselineModel()
    bm.fit(X_train, y_train)
    y_pred = bm.predict(X_test)
    loss_baseline = get_final_score(bm, d, scoring)


# --- RIDGE --- #


    pipe = Pipeline([
        ('scaler', StandardScaler()),
         ('ridge', Ridge())])


    param_grid = {'ridge__alpha': np.logspace(np.log10(.1), np.log10(10), 7)}

    grid = GridSearchCV(estimator=pipe,
                        param_grid=param_grid,
                        cv=cv_splits,
                        scoring = 'neg_root_mean_squared_error')

    grid.fit(X_train, y_train)
    loss_ridge = get_final_score(grid.best_estimator_, d, scoring)

# --- PLS --- #

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pls', PLSRegression(scale=False))])

    param_grid = {'pls__n_components': [2, 5, 10, 15, 20]}

    grid = GridSearchCV(estimator=pipe,
                        param_grid=param_grid,
                        cv=cv_splits,
                        scoring = 'neg_root_mean_squared_error',
                        n_jobs=-1)

    grid.fit(X_train, y_train)
    loss_pls = get_final_score(grid.best_estimator_, d, scoring)

# --- RANDOM FOREST --- #

    rf =  RandomForestRegressor(
            max_depth=5,
            min_samples_leaf=8,
            n_estimators=100,
            bootstrap=True,
            n_jobs=-1)

    loss_rf = get_final_score(rf, d, scoring)

    obs = np.column_stack([subject, loss_baseline, loss_ridge, loss_rf])
    result = pd.concat([result, pd.DataFrame(obs)], axis=0)

# --- XGBoost --- #
    model = XGBRegressor(
            tree_method='gpu_hist',
            max_bin=64,
            gpu_id=0)



    search_spaces = {
		'n_estimators': Integer(50, 500),
		'max_depth': Integer(3, 10),
		'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
		'subsample': Real(0.5, 1.0),
		'colsample_bytree': Real(0.5, 1.0)
	}

    param_grid = {
		"n_estimators": [100, 300, 500, 1000],        # more trees for stability
		"max_depth": [2, 3, 5, 7, 9],                 # shallow to moderately deep
		"learning_rate": [0.001, 0.01, 0.05, 0.1],    # smaller rates + more trees
		"subsample": [0.5, 0.7, 0.8, 1.0],            # more aggressive row sampling
		"colsample_bytree": [0.3, 0.5, 0.7, 0.9, 1.0],# column sampling (important for collinear features!)
		"min_child_weight": [1, 5, 10],               # larger = more conservative splits
		"gamma": [0, 0.1, 0.5, 1.0],                  # min loss reduction for split
		"reg_alpha": [0, 0.01, 0.1, 1.0],             # L1 regularization (feature selection)
		"reg_lambda": [0.1, 1.0, 10.0]                # L2 regularization (shrinkage)
	}


    grid = RandomizedSearchCV(estimator=model,
						param_distributions=param_grid,
                        cv=cv_splits,
                        scoring = 'neg_root_mean_squared_error',
                        n_jobs=-1)

    bayes = BayesSearchCV(estimator=model,
                          search_spaces=search_spaces,
                          n_iter=30,
                          cv=cv_splits,
                          scoring='neg_root_mean_squared_error',
                          n_jobs=-1)

    grid.fit(X_train, y_train)

    bayes.fit(X_train, y_train)

    loss_xgb = get_final_score(grid.best_estimator_, d, scoring)





result.columns = ['subject', 'baseline', 'ridge', 'random_forest']

result.to_csv(root / Path('within_subject.csv'), index=False)
