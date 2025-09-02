import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from glob import glob
import os
from groupyr import SGL
from pyprojroot import here
os.chdir(here() / Path('analysis'))
from scripts.modules.modeling.bayes_opt import BayesCV
from scripts.modules.preprocessing.cv_prep import get_cv_splits
from sklearn.metrics import mean_squared_error as mse
root = Path('scripts/sandbox/run_ml')


dpath = Path('data/formatted')
subjects = glob(str(dpath / Path('*')))
subjects = [Path(x).stem for x in subjects ]
subjects = sorted(subjects, key=lambda x: int(x.split('-')[1]))
bads = ['sub-023']
subjects = [x for x in subjects if x not in bads]

out = pd.DataFrame()

for subject in subjects:
    print(f'Subject {subject} of {subjects[-1]}')

    dpath = Path(f'data/formatted/{subject}.pkl')
    with open(dpath, 'rb') as file:
        d = pickle.load(file)


    splits = get_cv_splits(d['ses-001'])

    train_runs = list(d['ses-001'].keys())
    test_runs = list(d['ses-002'].keys())

# Train test split
    X_train = np.concatenate([d['ses-001'][x]['X'] for x in train_runs], axis=0)
    y_train = np.concatenate([d['ses-001'][x]['y']['DNa'] for x in train_runs], axis=0)
    X_test = np.concatenate([d['ses-002'][x]['X'] for x in test_runs], axis=0)
    y_test = np.concatenate([d['ses-002'][x]['y']['DNa'] for x in test_runs], axis=0)

    param_grid = {'l1_ratio': [.2, .8],
                  'alpha': np.logspace(-4, 1, 3)}

    limits = {'l1_ratio': {'min': 0, 'max': 1, 'space': 'linear'},
              'alpha': {'min': 1e-4, 'max': 10, 'space': 'log'}}

    bcv = BayesCV(estimator=SGL,
                  param_grid=param_grid,
                  cv_splits=splits,
                  limits=limits,
                  verbose=True)


    bcv.fit(X_train, y_train)
    nonzero_coef = bcv.nonzero_coef_
    y_pred = bcv.predict(X_test)
    ses2_loss = mse(y_test, y_pred)

    model = bcv.best_estimator_
    model.fit(X_test, y_test)
    y_pred = model.predict(X_train)
    ses1_loss = mse(y_train, y_pred)

    sgl_score = np.mean([ses1_loss, ses2_loss])
    baseline_score = np.mean([mse(np.full(len(y_test), np.mean(y_train)), y_test),
                              mse(np.full(len(y_train), np.mean(y_test)), y_train)])


    row = pd.DataFrame(np.column_stack([subject, nonzero_coef,
                                        baseline_score, sgl_score]))
    out = pd.concat([out, row], axis=0)


out.columns = ['subject', 'nonzero_coef', 'baseline', 'sgl']
out.to_csv(root / Path('sgl_subjects.csv'), index=False)

