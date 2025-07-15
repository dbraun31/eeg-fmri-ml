import mne
from mne.time_frequency import tfr_array_morlet
from glob import glob
from datetime import datetime
import pickle
import re
from sklearn.preprocessing import StandardScaler
from pyprojroot import here
from pathlib import Path
import numpy as np
import os
import numpy as numpy
from types import SimpleNamespace

class Reformat:

    def __init__(self, subjects):
        self.subjects = subjects
        now = datetime.now()
        self.time = now.strftime('%Y%m%d%H%M%S')
        self.data_path = Path('analysis/data/formatted')
        completed = glob(str(self.data_path / Path('sub')) + '*')
        self.completed = [Path(x).stem for x in completed]


    def run(self):

        subjects = self.subjects

        for subject in subjects:
            # Subject specific data only
            d = {}
            print('\n')
            print('-----------------------------')
            print(f'Subject: {subject}')
            print('\n')

            if subject in self.completed:
                continue

            for session in ['ses-001', 'ses-002']:
                d[session] = {}
                path_eeg = Path(f'analysis/data/original/{subject}/{session}/eeg')
                path_fmri= Path(f'analysis/data/original/{subject}/{session}/func')

                # Only experience sampling
                files_eeg = glob(str(path_eeg / Path('*.set')))
                files_eeg = [x for x in files_eeg if 'gradcpt' not in x.lower()]
                files_eeg = sorted(files_eeg, key=self._sort)
                files_fmri = glob(str(path_fmri / Path('DMN_*')))
                files_fmri = [x for x in files_fmri if 'gradcpt' not in x.lower()]
                files_fmri = sorted(files_fmri, key=self._sort)

                if not files_eeg or not files_fmri:
                    mi = self._get_metainfo(files_eeg)
                    message = (f'Missing data for subject {subject} '
                               f'session {session}. Skipping session.')
                    print(message + '\n')
                    self._update_log(message, mi)
                    continue
                    

                if not len(files_eeg) == len(files_fmri):
                    mi = self._get_metainfo(files_eeg)
                    message("Number of files detected for EEG not equal to "
                             "number detected for fMRI.\n" 
                            f"Subject: {subject}, Session: {session}, Run: {run}\n"
                            f"EEG files: {files_eeg}\n"
                            f"fMRI files: {files_fmri}\n"
                            "Skipping run.")
                    self._update_log(message, mi)
                    continue

                for run, (file_eeg, file_fmri) in enumerate(zip(files_eeg, files_fmri), start=1):
                    run = 'run-' + str(run).zfill(3)
                    print('\n')
                    print(f'Subject: {subject}, Session: {session}, Run: {run}')
                    print('\n')
                    X = self._process_eeg(file_eeg)

                    # If no EEG data
                    if X is None:
                        continue

                    y = self._process_fmri(file_fmri)

                    # Try chopping off last TR
                    if X[:-1, :].shape[0] == len(y):
                        X = X[:-1, :]

                    # Validate equal observations across X and y
                    if X.shape[0] != len(y):
                        mi = self._get_metainfo(file_eeg)
                        message = (f"Unequal observations for {mi['subject']} "
                                   f"{mi['session']} {mi['run']}, X: {X.shape[0]}, "
                                   f"y: {len(y)}. Skipping run.")
                        print(message + '\n')
                        self._update_log(message, mi)
                        continue

                    d[session][run] = {'X': X, 'y': y}

            self.subject = subject
            self._write_data(d)



    def _write_data(self, d):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        out_file = self.data_path / Path(self.subject + '.pkl')
        with open(out_file, 'wb') as file:
            pickle.dump(d, file)


    def _sort(self, run):
        '''
        Sort by bld\d\d\d 
        or run-\d\d\d
        '''
        pattern = r'(?:run-|bld)(\d+)_'
        m = int(re.search(pattern, run).group(1))
        return m


    def _process_eeg(self, file_eeg):
        '''
        Convert eeg file path to a (248, 31 * 40 * 9) array
        '''

        # Open EEGlab file
        # (some of these EEGlab data don't actually have the data in it)
        try:
            raw = mne.io.read_raw_eeglab(file_eeg)
        except FileNotFoundError as exception:

            mi = self._get_metainfo(file_eeg)
            message = (f"Error with {mi['subject']} {mi['session']}"
                       f" {mi['run']}. Message: {exception}. Skipping run.")
            self._update_log(message, mi)

            return None

        # Add a bogus epochs dimension
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        data = data[np.newaxis, ...]

        # Obtain time-frequency data
        freqs = np.array(range(1, 41))
        njobs = int(os.cpu_count() - 1)

        # Index out the bogus dimension
        tf_full = tfr_array_morlet(
                data,
                sfreq=sfreq,
                freqs = freqs,
                n_cycles=7,
                n_jobs=njobs,
                output='complex')[0]


        # Convert to power
        power = np.abs(tf_full.data) ** 2
        # Normalize within frequency band
        rel_power = power / power.sum(axis=2, keepdims=True)

        # Downsample to only TR observations
        events, event_id = mne.events_from_annotations(raw)
        tr_samples = events[events[:,2] == event_id['T  1'], 0]
        tf = rel_power[:, :, tr_samples]

        # Reshape (TRs, channels, freqs)
        tf = tf.transpose(2, 0, 1)

        # Get the lags
        num_lags = 9
        lag_ar = np.zeros((tf.shape[0], tf.shape[1], tf.shape[2], num_lags))

        # Only fill in starting at TR 9 so can fully backfill with lags
        for i in range(num_lags-1, tf.shape[0]):
            for lag in range(num_lags):
                lag_ar[i, :, :, lag] = tf[i-lag, :, :]

        # We're starting at TR 9
        ar = lag_ar[8:, :, :, :]
        # Reshape to (248, 31 * 40 * 9)
        ar_reshape = ar.reshape(ar.shape[0], -1)

        return ar_reshape



    def _update_log(self, message, mi):
        # Either create or update log with message

        log_path = Path('analysis/scripts/modules/preprocessing/logs')
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        now = datetime.now()
        time = self.time
        log_file = log_path / Path(time + '_log.txt')

        message = '\n' + message + '\n'
        print(message)

        if os.path.exists(log_file):
            with open(log_file, 'r') as file:
                log = file.read()
                log += message
        else:
            log = message

        with open(log_file, 'w') as file:
            file.write(log)
        file.close()


    def _get_metainfo(self, file):
        '''
        Get subject, session, run from file
        '''
        sub_p = r'sub-\d+'
        ses_p = r'ses-\d+'
        run_p = r'(?:bld|run-)\d+'
        subject = re.search(sub_p, file).group(0)
        session = re.search(ses_p, file).group(0)
        run = re.search(run_p, file).group(0)

        return {'subject': subject, 'session': session, 'run': run}


    def _process_fmri(self, file_fmri):
        '''
        Convert eeg file path to a (248, 31 * 40 * 9) array
        '''

        with open(file_fmri, 'r') as f:
            d = f.readlines()

        d = np.array([float(x.strip()) for x in d])

        # Chop off first 8 observations
        return d[8:]


if __name__ == '__main__':

    os.chdir(here())
    subjects = sorted([Path(x).name for x in glob('analysis/data/original/*') if 'sub' in x])
    reformat = Reformat(subjects)
    reformat.run()



