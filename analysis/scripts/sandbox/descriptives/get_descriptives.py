import mne
import math
import pandas as pd
from glob import glob
from pathlib import Path
import os
from pyprojroot import here
os.chdir(here())
root = Path('analysis/scripts/sandbox/descriptives')


# Get subjects
subjects = [Path(x).stem for x in glob('analysis/data/original/sub-*')]

N = len(subjects)


# Loop

out = {}
for subject in subjects:

    runs = glob(f'analysis/data/original/{subject}/**/*.set', recursive=True)

    N_runs = len(runs)

    mins = 0
    for run in runs:
        raw = mne.io.read_raw_eeglab(run)
        mins += len(raw) / raw.info['sfreq'] / 60

    hours = math.floor(mins / 60)
    mins = round(mins % 60, 2)

    time_report = f'{hours} hour(s) and {mins} minutes'

    out[subject] = {'N_runs': N_runs, 'time_report': time_report}


    

subjects = sorted(subjects, key = lambda x: int(x.split('-')[1]))

d = pd.DataFrame()

for subject in subjects:
    row = pd.DataFrame({'subject': [subject],
                        'N_runs': [out[subject]['N_runs']],
                        'runtime': [out[subject]['time_report']]})
    d = pd.concat([d, row], axis = 0)


d.to_csv(root / Path('subject_info.csv'), index=False)
