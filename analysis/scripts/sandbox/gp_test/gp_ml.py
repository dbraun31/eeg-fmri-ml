from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from groupyr import SGL
import pickle
from pyprojroot import here
from pathlib import Path
import os
import numpy as np
from analysis.scripts.modules.modeling.groups import get_groups

os.chdir(here())
root = Path('analysis/scripts/sandbox/gp_test')


path = Path('analysis/data/formatted/sub-002.pkl')

# Get data
with open(path, 'rb') as file:
    d = pickle.load(file)

runs = list(d['ses-001'].keys())
X_train = np.concatenate([d['ses-001'][x]['X'] for x in runs])
y_train = np.concatenate([d['ses-001'][x]['y']['DAN'] for x in runs])
X_test = np.concatenate([d['ses-002'][x]['X'] for x in runs])
y_test = np.concatenate([d['ses-002'][x]['y']['DAN'] for x in runs])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def rmse(y_pred, y_test):
    return np.sqrt(np.mean((y_pred - y_test)**2))

def objective(hyper):
    groups = get_groups()
    sgl = SGL(alpha = hyper[0], l1_ratio = hyper[1],
              groups = groups)
    sgl.fit(X_train, y_train)
    y_pred = sgl.predict(X_test)
    loss = rmse(y_pred, y_test)
    return loss

# --- GET INITIAL OBSERVATION --- #

# Starting obs
# [l1_ratio, alpha]
alpha = np.logspace(-4, 0, 3)
l1_ratio = np.array([.3, .6])
P1, P2 = np.meshgrid(alpha, l1_ratio)
X_hyper = np.column_stack([P1.ravel(), P2.ravel()])
y_loss = []

for i, hyper in enumerate(X_hyper, start=1):

    print(f'Iteration {i}/{X_hyper.shape[0]}')

    y_loss.append(objective(hyper))


# --- INIT GP --- #

kernel = ConstantKernel(1) * RBF(1) + WhiteKernel(1)

gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True)


# --- OPTIMIZE --- #

max_iter = 5

for iteration in range(max_iter):

    print(f'Iteration {iteration+1}\n')
    gp.fit(X_hyper, y_loss)

    # Predict over grid

    l1_s = np.linspace(0, 1, 100)
    alpha_s = np.logspace(-4, 1, 100)
    P1, P2 = np.meshgrid(alpha_s, l1_s)
    X_s = np.column_stack([P1.ravel(), P2.ravel()])
    k = .01

    mu, sigma = gp.predict(X_s, return_std=True)

    ucb = mu - k * sigma
    idx = np.where(ucb == np.min(ucb))[0][0]
    X_next = X_s[idx]
    
    # Obtain the observation
    loss = objective(X_next)

    # Append observation
    X_hyper = np.concatenate([X_hyper, X_next.reshape(-1, 2)])
    y_loss.append(loss)



# Inspect
# Theoretical min
print('Theoretical min')
print('X: ', X_s[np.where(mu == np.min(mu))[0][0],:])
print('y: ', np.min(mu))
# Empirical min
print('Empirical min')
print('X: ', X_hyper[np.where(y_loss == np.min(y_loss))[0][0], :])
print('y: ', np.min(y_loss))
