import mne
import numpy as np
import pandas as pd
from pyprojroot import here
import os
from glob import glob
from pathlib import Path
import pickle
from itertools import product
os.chdir(here() / Path('analysis'))




def process_subject(subject, col_names, data_path=Path('data/formatted')):

    file = data_path / Path(f'{subject}.pkl')
    with open(file, 'rb') as file:
        data = pickle.load(file)

    subject_d = pd.DataFrame()

    for session in data:
        for run in data[session]:

            run_data = data[session][run]
            run_df = pd.DataFrame(run_data['X'], columns=col_names)
            run_df.insert(0, 'subject', subject)
            run_df.insert(1, 'session', session)
            run_df.insert(2, 'run', run)
            run_df.insert(3, 'tr', list(range(1, run_df.shape[0]+1)))
            run_df.insert(4, 'dmn', data[session][run]['y']['dmn'])
            run_df.insert(5, 'dan', data[session][run]['y']['dan'])
            run_df.insert(6, 'dmn_a', data[session][run]['y']['dmn_a'])
            run_df.insert(7, 'dmn_b', data[session][run]['y']['dmn_b'])
            subject_d = pd.concat([subject_d, run_df], axis=0)

    return subject_d


def make_column_names(channels):
    freqs = list(range(1, 41))
    lags = list(range(11))
    
    col_names_tuple = list(product(channels, freqs, lags))

    col_names = []
    for name_tuple in col_names_tuple:
        name = '_'.join([str(x) for x in name_tuple])
        col_names.append(name)

    return col_names


if __name__ == '__main__':


    raw = mne.io.read_raw_eeglab('data/original/sub-001/ses-001/eeg/sub-001_ses-001_bld001_eeg_Bergen_CWreg_filt_ICA_rej.set')
    channels = raw.info['ch_names']
    col_names = make_column_names(channels)

    subjects = [Path(x).stem for x in glob('data/formatted/*')]

    d = pd.DataFrame()

    for subject in subjects:
        d = pd.concat([d, process_subject(subject, col_names)], axis=0)

    d.reset_index(drop=True).to_feather('data/merged_data.feather')
