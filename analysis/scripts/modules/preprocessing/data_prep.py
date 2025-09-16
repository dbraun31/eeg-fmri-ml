import numpy as np
import os
import pickle
import joblib

# dev
from glob import glob
from pathlib import Path
from pyprojroot import here

os.chdir(here() / Path('analysis'))

subjects = [Path(x).stem for x in glob('data/formatted/*')]
subjects = sorted(subjects, key=lambda x: int(x.split('-')[1]))


def get_full_data(subjects, network='DNa'):
    # Create subjects_X_y.pkl of all the data
    
    out = {}

    for subject in subjects:

        dpath = Path(f'data/formatted/{subject}.pkl')
        with open(dpath, 'rb') as file:
            d = pickle.load(file)

        sessions = list(d.keys())
        X_ses = []
        y_ses = []
        for session in sessions:
            runs = list(d[session].keys())

            if runs:
                X_ses.append(np.concatenate([d[session][x]['X'] for x in runs], axis=0))
                y_ses.append(np.concatenate([d[session][x]['y'][network] for x in runs], axis=0))


        X = np.concatenate(X_ses, axis=0)
        y = np.concatenate(y_ses, axis=0)
        out[subject] = {'X': X, 'y': y}

    with open('data/subject_X_y.pkl', 'wb') as file:
        pickle.dump(out, file)





        
