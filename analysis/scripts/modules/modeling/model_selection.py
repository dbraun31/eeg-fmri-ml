import numpy as np


def train_test_split(d, train_session='ses-001', network='DNa'):
    '''
    Given data d, return X_train, y_train, X_test, y_test
    session controls whether data is from ses 1 or 2
    '''

    test_session = 'ses-002' if train_session == 'ses-001' else 'ses-001'
    train_runs = list(d[train_session].keys())
    test_runs = list(d[test_session].keys())

    # Train test split
    X_train = np.concatenate([drop_lags(d[train_session][x]['X']) for x in train_runs], axis=0)
    y_train = np.concatenate([d[train_session][x]['y'][network] for x in train_runs], axis=0)
    X_test = np.concatenate([drop_lags(d[test_session][x]['X']) for x in test_runs], axis=0)
    y_test = np.concatenate([d[test_session][x]['y'][network] for x in test_runs], axis=0)

    return X_train, y_train, X_test, y_test


    
def drop_lags(X, seconds_back=10):
    # X is (obs, channels, freqs, lags)
    ar = X.reshape(X.shape[0], 31, 40, -1)
    n_lags = (seconds_back // 2) + 1

    ar_trim = ar[:, :, :, :n_lags]
    out = ar_trim.reshape(X.shape[0], -1)

    return out



def get_cv_splits(session):
    '''
     Takes in session data {'run1': {'X': ..., 'y': ...}, ...}
     Returns cv_splits
       [(train_idx1, test_idx1), (train_idx2, test_idx2), ...]
    '''

    # Concatenate data
    runs = list(session.keys())
    X = np.concatenate([session[x]['X'] for x in runs])
    y = np.concatenate([session[x]['y']['DNa'] for x in runs])

    # Get all run indices
    run_idxs = {}
    start = 0
    for run in runs:
        stop = session[run]['X'].shape[0] 
        run_idxs[run] = np.array(range(start, start+stop))
        start += stop

    cv_splits = []
    for run in runs:
        test_idx = run_idxs[run]
        train_idx = np.setdiff1d(np.arange(X.shape[0]), test_idx)
        cv_splits.append((train_idx, test_idx))

    return cv_splits


