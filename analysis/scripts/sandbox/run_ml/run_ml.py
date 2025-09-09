import mne
import os
from pyprojroot import here
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from glob import glob
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, get_scorer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
os.chdir(here() / Path('analysis'))
from scripts.modules.modeling.model_selection import train_test_split
from scripts.modules.modeling.model_compare import ModelCompare, get_final_score
from scripts.modules.modeling.baseline import BaselineModel
from sklearn.model_selection import GridSearchCV
from scripts.modules.preprocessing.cv_prep import get_cv_splits_ar


root = Path('scripts/sandbox/run_ml')
droot = Path('data/formatted')

subjects = [Path(x).stem for x in glob(str(droot / Path('*')))]
subjects = sorted(subjects, key=lambda x: int(x.split('-')[1]))

result = pd.DataFrame()

for subject in subjects:
    print(f'\n---PROCESSING SUBJECT {subject} of {subjects[-1]}---\n')

    subject = subjects[12]
    dpath = droot / Path(f'{subject}.pkl')

    with open(dpath, 'rb') as file:
        d = pickle.load(file)

    X_train, y_train, X_test, y_test = train_test_split(d)

    estimators = {'ridge': Ridge, 'rf': RandomForestRegressor}
    param_grids = {'ridge': {'alpha': np.logspace(np.log10(.1), np.log10(10), 7)},
                   'rf': {'max_samples_leaf': [5, 10, 15]}}
    fixed_params_dict = {'rf': {'n_estimators': 100}}



    def rmse(y_test, y_pred):
        return np.sqrt(sum((y_pred - y_test)**2))

    scoring = make_scorer(rmse, greater_is_better=False)

# --- BASELINE --- #
    bm = BaselineModel()
    bm.fit(X_train, y_train)
    y_pred = bm.predict(X_test)
    loss_baseline = get_final_score(bm, d, scoring)


# --- RIDGE --- #

    cv_splits = get_cv_splits_ar(d['ses-001'])

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


result.columns = ['subject', 'baseline', 'ridge', 'random_forest']

result.to_csv(root / Path('within_subject.csv'), index=False)
