import numpy as np

def get_groups(n_channels=31, n_freqs=40, n_lags=11):
    # List of arrays needed for SGL input
    # Grouped only over channels for now
    # Eventually I'd like to vary what we group over

    n_features = n_channels * n_freqs * n_lags

    idxs = np.arange(n_features)

    groups = list(np.split(idxs, n_channels))

    return groups


