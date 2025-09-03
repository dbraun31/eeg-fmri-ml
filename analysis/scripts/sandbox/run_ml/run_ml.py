from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
import os
import numpy as np
from glob import glob
import pickle
from pyprojroot import here
from pathlib import Path
os.chdir(here() / Path('analysis'))
from scripts.modules.modeling.model_selection import train_test_split, get_cv_splits
import optuna


# Test subject
subjects = glob('data/formatted/*')
subjects = sorted([Path(x).stem for x in subjects], key=lambda x: int(x.split('-')[1]))

subject = subjects[1]
dpath = Path(f'data/formatted/{subject}.pkl')

with open(dpath, 'rb') as file:
    d = pickle.load(file)


# Partition data
X_train, y_train, X_test, y_test = train_test_split(d)
splits = get_cv_splits(d['ses-001'])
logo = LeaveOneGroupOut()
cv = list(logo.split(X_train, y_train, groups=splits))


# Loss
def rmse(y_pred, y_test):
    return np.sqrt(sum((y_pred - y_test)**2))


# Baseline
y_pred = np.full(y_test.shape[0], y_train.mean())
loss_baseline = rmse(y_pred, y_test)

# Elastic net
enet = ElasticNetCV()
enet.fit(X_train, y_train)
y_pred = enet.predict(X_test)
loss_enet = rmse(y_pred, y_test)

def objective(trial):
    alpha = trial.suggest_float('alpha', .001, 10, log=True)
    l1_ratio = trial.suggest_float('l1_ratio', .1, .9)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('enet', ElasticNet(alpha=alpha, l1_ratio=l1_ratio))])

    scores = cross_val_score(pipe, X_train, y_train, cv=cv, 
                             scoring='neg_root_mean_squared_error')

    return -scores.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, n_jobs=os.cpu_count()-1)

enet = ElasticNet(**study.best_params)
enet.fit(X_train, y_train)
y_pred = enet.predict(X_test)
loss_enet = rmse(y_pred, y_test)


# Random forest

def objective(trial):
    n_components = trial.suggest_int('n_components', 50, 500)
    n_estimators = trial.suggest_int('n_estimators', 100, 300, step=50)
    max_depth = trial.suggest_int('max_depth', 5, 60)
    max_features_frac = trial.suggest_float('max_features_frac', 0.05, 1.0)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features_frac,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            random_state=42,
            n_jobs=-1
        )

    scores = cross_val_score(rf, X_train, y_train, cv=cv, groups=splits,
                             scoring='neg_root_mean_squared_error')

    return -scores.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=80)





print(f'Baseline: {loss_baseline}\nENet: {loss_enet}')
