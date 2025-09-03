import numpy as np


def train_test_split(d, session='ses-001', network='DNa'):
    '''
    Given data d, return X_train, y_train, X_test, y_test
    session controls whether data is from ses 1 or 2
    '''

    session_test = 'ses-002' if session == 'ses-001' else 'ses-001'
    train_runs = list(d[session].keys())
    test_runs = list(d[session_test].keys())

    # Train test split
    X_train = np.concatenate([drop_lags(d[session][x]['X']) for x in train_runs], axis=0)
    y_train = np.concatenate([d[session][x]['y'][network] for x in train_runs], axis=0)
    X_test = np.concatenate([drop_lags(d[session][x]['X']) for x in test_runs], axis=0)
    y_test = np.concatenate([d[session][x]['y'][network] for x in test_runs], axis=0)

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
    *** this is for sklearn functions ***
    '''
    out = []

    for i, run in enumerate(session.keys(), start=1):

        X = session[run]['X']
        out += list(np.full(X.shape[0], i))

    return out




