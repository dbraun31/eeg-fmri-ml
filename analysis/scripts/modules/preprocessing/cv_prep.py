import numpy as np

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

