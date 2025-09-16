from pyprojroot import here
from pathlib import Path
import pandas as pd
from glob import glob
import numpy as np
import os
import pickle
os.chdir(here() / Path('analysis'))
root = Path('scripts/sandbox/run_ml')
from scripts.modules.modeling.baseline import BaselineModel
from scripts.modules.modeling.model_selection import drop_lags

subjects = [Path(x).stem for x in glob('data/formatted/*')]
subjects = sorted(subjects, key=lambda x: int(x.split('-')[1]))

with open('data/subject_X_y.pkl', 'rb') as file:
    d = pickle.load(file)

with open(root / Path('models/models.pkl'), 'rb') as file:
    models = pickle.load(file)

models['baseline'] = BaselineModel()
loss = {x: {'rmse': [], 'r2': []} for x  in models}


def r2(y_test, y_pred):
    y_test_bar = y_test.mean()
    term1 = sum((y_test - y_pred)**2)
    term2 = sum((y_test - y_test_bar)**2)

    return 1 - (term1 / term2)

def rmse(y_test, y_pred):
    return np.sqrt(sum((y_test - y_pred)**2))


for subject in subjects:
    print(f'---Subject {subject} of {subjects[-1]}---')

    X_train = drop_lags(np.concatenate([d[x]['X'] for x in subjects if x != subject], axis=0))
    y_train = np.concatenate([d[x]['y'] for x in subjects if x != subject], axis=0)
    X_test = drop_lags(d[subject]['X'])
    y_test = d[subject]['y']


    for i, model in enumerate(models, start=1):

        print(f'\nModel {i} ({model}) of {len(models)}\n')

        estimator = models[model]
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        loss_r2 = r2(y_test, y_pred)
        loss_rmse = rmse(y_test, y_pred)

        loss[model]['r2'].append(loss_r2)
        loss[model]['rmse'].append(loss_rmse)


out = {}

for model in loss:
    for metric in ['r2', 'rmse']:
        out[f'{model}_{metric}'] = [loss[model][metric]]


pd.DataFrame(out).to_csv(root / Path('across_subjects.csv'), index=False)

