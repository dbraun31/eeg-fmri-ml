from groupyr import SGL
from pyprojroot import here 
from itertools import product
from datetime import datetime
import numpy as np
import os
from glob import glob
from pathlib import Path
import pickle
os.chdir(here() / Path('analysis'))
from scripts.modules.modeling.baseline import BaselineModel
from scripts.modules.modeling.model_compare import ModelCompare
from scripts.modules.modeling.groups import get_groups



# Pick a subject
subjects = [Path(x).stem for x in glob('data/formatted/*')]
bads = [1, 11, 13, 19, 23, 25]
bads = ['sub-' + str(x).zfill(3) for x in bads]
subjects = [x for x in subjects if x not in bads]
# Random one for now
subject = subjects[-1]




# --- COMPARE MODELS --- #
estimators = {'baseline': BaselineModel,
              'SGL': SGL}
param_grids = {'baseline': None,
               'SGL': {'alpha': np.logspace(-7, -3, 3),
                       'l1_ratio': [.1, .5, .9]}
               }
fixed_params_dict = {'SGL': {'groups': get_groups()}}

mc = ModelCompare(estimators=estimators,
                  param_grids=param_grids,
                  fixed_params_dict=fixed_params_dict,
                  subject=subject)

result = mc.run()




# --- RUN SINGLE --- #

# Tune hyperparams
param_grid = {'alpha': np.logspace(-7, -3, 10),
              'l1_ratio': [.1, .3, .5, .7, .9]}
groups = get_groups(X_train.shape[1])
fixed_params = {'groups': groups}
cv_splits = get_cv_splits(data['ses-001'])
cv = GridSearchCV(estimator=SGL, 
                  param_grid=param_grid,
                  cv_splits=cv_splits,
                  fixed_params=fixed_params,
                  scoring=rmse,
                  verbose=1)
runs = list(data['ses-001'].keys())
X = np.concatenate([data['ses-001'][x]['X'] for x in runs])
y = np.concatenate([data['ses-001'][x]['y'] for x in runs])

cv.fit(X, y)


get_final_score(SGL, cv.best_params_, data, rmse)


