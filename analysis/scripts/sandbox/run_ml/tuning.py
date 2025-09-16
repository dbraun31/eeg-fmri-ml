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
import sys
os.chdir(here() / Path('analysis'))
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


root = Path('scripts/sandbox/run_ml')
droot = Path('data')

subjects = [Path(x).stem for x in glob(str(droot / Path('formatted/*')))]
subjects = sorted(subjects, key=lambda x: int(x.split('-')[1]))


class BaselineModel:
    '''
    Fit and predict methods for a unconditional mean model
    '''

    def __init__(self, **kwargs):

        self.mean_ = None

    def fit(self, X, y):

        self.mean_ = np.mean(y)

    def predict(self, X):
        
        return np.full(shape=(X.shape[0],), fill_value=self.mean_)

def drop_lags(X, seconds_back=10):
    # X is (obs, channels, freqs, lags)
    ar = X.reshape(X.shape[0], 31, 40, -1)
    n_lags = (seconds_back // 2) + 1

    ar_trim = ar[:, :, :, :n_lags]
    out = ar_trim.reshape(X.shape[0], -1)

    return out

# --- HYPERPARAM TUNING --- #
leftout = 'sub-025'
subjects_tune = [x for x in subjects if x != leftout]


dpath = droot / Path(f'subject_X_y.pkl')
mpath = root / Path('models/models.pkl')

with open(dpath, 'rb') as file:
    d = pickle.load(file)


X_train = drop_lags(np.concatenate([d[x]['X'] for x in subjects_tune], axis=0))
y_train = np.concatenate([d[x]['y'] for x in subjects_tune], axis=0)
X_test = drop_lags(d[leftout]['X'])
y_test = d[leftout]['y']

# Float 32 to save memory
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)

def rmse(y_test, y_pred):
    return np.sqrt(sum((y_pred - y_test)**2))

def save_model(model, model_name, mpath):

    if not os.path.exists(mpath):
        models = {model_name: model}
        with open(mpath, 'wb') as file:
            pickle.dump(models, file)
    else:
        with open(mpath, 'rb') as file:
            models = pickle.load(file)
        if model_name not in models:
            models[model_name] = model
        with open(mpath, 'wb') as file:
            pickle.dump(models, file)

scoring = make_scorer(rmse, greater_is_better=False)

obs = {}
models = {}


# What to run
ridge = False
pls = True
rf = False
xgboost = True

# --- BASELINE --- #
bm = BaselineModel()
bm.fit(X_train, y_train)
y_pred = bm.predict(X_test)
obs['loss_baseline'] = scoring(bm, X_test, y_test) * scoring._sign



# --- RIDGE --- #
if ridge:
    print('--- Fitting ridge ---')

    pipe = Pipeline([
        ('scaler', StandardScaler()),
         ('ridge', Ridge())])


    param_grid = {'ridge__alpha': np.logspace(np.log10(10000000), np.log10(100), 10)}

    grid = GridSearchCV(estimator=pipe,
                        param_grid=param_grid,
                        cv = 4,
                        scoring = 'neg_root_mean_squared_error',
                        n_jobs=1)

    grid.fit(X_train, y_train)
    obs['loss_ridge'] = scoring(grid.best_estimator_, X_test, y_test) * scoring._sign

    save_model(grid.best_estimator_, 'ridge', mpath)

    print(obs)


# --- PLS --- #
if pls:
    print('--- Fitting PLS --- ')

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pls', PLSRegression(scale=False))])

    param_grid = {'pls__n_components': [2, 3, 4]}

    grid = GridSearchCV(estimator=pipe,
                        param_grid=param_grid,
                        cv=4,
                        scoring = 'neg_root_mean_squared_error',
                        n_jobs=1)

    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test).reshape(-1,)
    obs['loss_pls'] = rmse(y_test, y_pred)

    save_model(grid.best_estimator_, 'pls', mpath)

# --- RANDOM FOREST --- #
if rf:
    print('--- Fitting random forest ---')

    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [4, 8, 16],
        "max_features": ["sqrt", 0.3, 0.5],
        "bootstrap": [True]   # keep True to match your prior setting
    }

    rf =  RandomForestRegressor(n_jobs=os.cpu_count()-1)

    grid = RandomizedSearchCV(estimator=rf,
                              param_distributions=param_grid,
                              cv=4,
                              scoring='neg_root_mean_squared_error',
                              n_jobs=os.cpu_count()-1)

    grid.fit(X_train, y_train)

    obs['loss_rf'] = scoring(grid.best_estimator_, X_test, y_test) * scoring._sign

    save_model(grid.best_estimator_, 'rf', mpath)


# --- XGBoost --- #
if xgboost:
    print('--- Fitting XGBoost ---')
    model = XGBRegressor(
            tree_method='gpu_hist',
            gpu_id=0)

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
                        cv=4,
                        scoring = 'neg_root_mean_squared_error',
                        n_jobs=1)


    grid.fit(X_train, y_train)
    obs['loss_xgb'] = scoring(grid.best_estimator_, X_test, y_test) * scoring._sign
    save_model(grid.best_estimator_, 'xgboost', mpath)


