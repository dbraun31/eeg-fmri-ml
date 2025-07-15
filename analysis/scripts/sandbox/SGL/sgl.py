from groupyr import SGL
from pyprojroot import here 
from itertools import product
from datetime import datetime
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
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
# Full
estimators = {'baseline': BaselineModel,
              'SGL': SGL,
              'RandomForest': RandomForestRegressor}
param_grids = {'baseline': None,
               'SGL': {'alpha': np.logspace(-7, -3, 3),
                       'l1_ratio': [.1, .5, .9]},
               'RandomForest': {
                   'n_estimators': [50, 100],
                   'max_depth': [None, 10],
                   'min_samples_split': [2, 5],
                   'min_samples_leaf': [1, 2]}
               }
fixed_params_dict = {'SGL': {'groups': get_groups()},
                     'RandomForest': {'max_features': 'sqrt', 
                                      'n_jobs': os.cpu_count() - 1}
                    }

# Simple
estimators = {'RandomForest': RandomForestRegressor}
param_grids = {'RandomForest': {'n_estimators': [50, 100]}}
fixed_params_dict = {'RandomForest': {'max_features': 'sqrt',
                                      'n_jobs': os.cpu_count() - 1}}

mc = ModelCompare(estimators=estimators,
                  param_grids=param_grids,
                  fixed_params_dict=fixed_params_dict,
                  verbose=1)

result = mc.run()




