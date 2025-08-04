from pyprojroot import here
from pathlib import Path
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
from groupyr import SGL
import numpy as np
from sklearn.ensemble import RandomForestRegressor
os.chdir(here() / Path('analysis'))
from scripts.modules.modeling.baseline import BaselineModel
from scripts.modules.modeling.groups import get_groups

data_path = Path('data/formatted')

with open(data_path / Path('sub-025.pkl'), 'rb') as file:
    d = pickle.load(file)


runs = list(d['ses-001'].keys())
X_train = np.concatenate([d['ses-001'][x]['X'] for x in runs if x != 'run-001'])
y_train = np.concatenate([d['ses-001'][x]['y']['dan'] for x in runs if x != 'run-001'])
X_test = np.concatenate([d['ses-002'][x]['X'] for x in runs if x != 'run-001'])
y_test = np.concatenate([d['ses-002'][x]['y']['dan'] for x in runs if x != 'run-001'])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def rmse(y_test, y_pred):
    return np.sqrt(np.mean((y_test - y_pred)**2))

# Baseline
baseline = BaselineModel()
baseline.fit(X_train, y_train)
y_pred = baseline.predict(X_test)
baseline_loss = rmse(y_test, y_pred)
print(f'Baseline loss: {baseline_loss}')

# SGL
l1_ratio = 1
alpha = 3.30001
sgl = SGL(l1_ratio=l1_ratio, alpha=alpha, groups=get_groups())
sgl.fit(X_train, y_train)
y_pred = sgl.predict(X_test)
sgl_loss = rmse(y_test, y_pred)
nonzero = np.sum(sgl.coef_ != 0)
print(f'SGL Loss: {sgl_loss}, Nonzero coefs: {nonzero}')


# RF
rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        n_jobs=os.cpu_count()-1)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rf_loss = rmse(y_test, y_pred)
print(f'Random forest loss: {rf_loss}')

x = y_pred
y = y_train
min_val = min(min(x), min(y))
max_val = max(max(x), max(y))
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.scatter(x, y)
plt.xlabel('Predicted')
plt.ylabel('Observed')
plt.show()


plt.hist(y_pred, edgecolor='black')
plt.show()
