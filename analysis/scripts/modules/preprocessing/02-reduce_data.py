import mne
from pyprojroot import here
import pandas as pd
import os
from glob import glob
import numpy as np
import pickle
from pathlib import Path
from itertools import product
import sys
sys.path.append(str(here()))
os.chdir(here())



class Reduce:

    def __init__(self, n_freqs=40, n_chans=31, n_lags=11,
                 dpath=Path('analysis/data/formatted/full')):

        self.n_freqs = n_freqs
        self.n_chans = n_chans
        self.n_lags = n_lags
        self.dpath = dpath

        if not os.path.exists(dpath):
            raise RuntimeError('Must run reformat_data.py before this script')

        subjects = [Path(x).stem for x in glob(str(dpath / Path('*')))]
        self.subjects = sorted(subjects, key=lambda x: int(x.split('-')[1]))

        # Get channel names
        raw_path = Path('analysis/data/original/sub-001/ses-001/eeg')
        raw_name = Path('sub-001_ses-001_bld001_eeg_Bergen_CWreg_filt_ICA_rej.set')
        raw_file = raw_path / raw_name
        raw = mne.io.read_raw_eeglab(raw_file)
        self.ch_names = raw.info['ch_names']


    def run(self):

        for subject in self.subjects:
            print('\n')
            print(f'---Processing subject {subject} of {self.subjects[-1]}---\n')
            out = {}

            dfile = self.dpath / Path(f'{subject}.pkl')
            with open(dfile, 'rb') as file:
                d = pickle.load(file)

            for session in d:
                for run in d[session]:
                    X = self._reduce_eeg(d[session][run]['X'])
                    if session not in out:
                        out[session] = {run: {'X': X}}
                    else:
                        out[session][run] = {'X': X}
                    out[session][run]['y'] = d[session][run]['y']

            # Write per subject
            out_path = Path(f'analysis/data/formatted/reduced/{subject}.pkl')

            if not os.path.exists(out_path.parent):
                os.makedirs(out_path.parent)

            with open(out_path, 'wb') as file:
                pickle.dump(out, file)


    def _reduce_eeg(self, X):

        freqs = [f'freq{x}' for x in range(1, self.n_freqs+1)]
        lags = [f'lag{x}' for x in range(self.n_lags)]
        cols = list(product(self.ch_names, freqs, lags))
        cols = [f'{x[0]}_{x[1]}_{x[2]}' for x in cols]

        # --- Transform array to df and reshape ---
        X_df = pd.DataFrame(X, columns=cols)
        X_df['sample'] = range(1, X_df.shape[0]+1)
        X_df = X_df.melt(id_vars='sample', var_name='col', value_name='power')
        X_df[['channel', 'frequency', 'lag']] = X_df['col'].str.split('_', expand=True)
        X_df = X_df.drop(columns='col')
        X_df['frequency'] = X_df['frequency'].str.replace('freq', '').astype(int)
        X_df['lag'] = X_df['lag'].str.replace('lag', '').astype(int)

        # Band frequency
        bins = [0, 4, 8, 13, 30, 40]
        labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

        X_df['eeg_band'] = pd.cut(X_df['frequency'], bins=bins,
                                  labels=labels,
                                  right=True,
                                  include_lowest=True)

        # Reduce
        X_df = X_df[X_df['lag'] <= 5].groupby(['sample', 'channel', 'eeg_band',
                                        'lag'])['power']\
                                                .mean()\
                                                .reset_index()

        # Spread
        X_df['col'] = X_df[['channel', 'eeg_band', 'lag']].astype(str).agg('_'.join, axis=1)
        X_df = X_df.drop(columns=['channel', 'eeg_band', 'lag'])
        X_df = X_df.pivot(index='sample', columns='col', values='power').reset_index()
        X_df = X_df.drop(columns='sample')

        X = X_df.to_numpy()

        return X


if __name__ == '__main__':
    
    reduce = Reduce()
    reduce.run()
