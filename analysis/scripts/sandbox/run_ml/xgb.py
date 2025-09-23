from pyprojroot import here
import os
from pathlib import Path
from tqdm import tqdm
import sys
modules_root = here() / Path('analysis/scripts/modules')
os.chdir(here() / Path('analysis'))
sys.path.append(str(modules_root))
from modeling.model_selection import train_test_across
from modeling.dummy import BaselineModel
from utility import get_subjects
root = Path(modules_root / Path('../sandbox/run_ml'))
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle
from glob import glob



def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

dpath = Path('data/formatted/reduced')
subjects = get_subjects(dpath)

def get_p(surrogates, true_loss):
    return (np.sum(surrogates >= true_loss) + 1) / (len(surrogates) + 1)

def get_loss(subjects, surrogate=False, nsims=500):
	
    if not surrogate:
        nsims=1
	
    out = []

    for sim in range(1, nsims+1):

        print(f'\n\n--- SIMULATION {sim} OF {nsims}---\n\n')

        loss = []

        for subject in tqdm(subjects, desc='Processing subjects'):

            # Import data
            X_train, y_train, X_test, y_test = train_test_across(subject)

            if surrogate:
                np.random.shuffle(y_train)

            # XGBoost
            xgb = XGBRegressor(n_estimators=300,
                               max_depth=3,
                               learning_rate=.05,
                               subsample=.8,
                               colsample_bytree=.8,
                               tree_method='gpu_hist',
                               preditor='gpu_predictor',
                               verbosity=0)


            xgb.fit(X_train, y_train)
            y_pred = xgb.predict(X_test)
            loss.append(r2_score(y_test, y_pred))

        out.append(np.array(loss).mean())

    if not surrogate:
        return np.array(loss)

    return np.array(out)



surrogates = get_loss(subjects, surrogate=True, nsims=50)

# Combine with existing if they exist
if os.path.exists(root / Path('surrogates.npy')):
    surrogates_old = np.load(root / Path('surrogates.npy'))
    surrogates = np.concatenate([surrogates, surrogates_old], axis=0)

# Write surrogates
np.save(root / Path('surrogates.npy'), surrogates)

# Get and write truth
truth = get_loss(subjects)
p = get_p(surrogates, truth.mean())
np.save(root / Path('truth.npy'), truth)

