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
from scipy.signal import butter, filtfilt

class Reformat:

    def __init__(self, subjects, num_lags, overwrite=False):
        self.subjects = subjects
        now = datetime.now()
        self.time = now.strftime('%Y%m%d%H%M%S')
        self.data_path = Path('analysis/data/formatted')
        self.num_lags = num_lags
        self.data_path.mkdir(parents=True, exist_ok=True)
        completed = glob(str(self.data_path / Path('sub')) + '*')
        self.completed = [Path(x).stem for x in completed]
        self.overwrite = overwrite


    def run(self):

        subjects = self.subjects

        for subject in subjects:
            # Subject specific data only
            d = {}
            print('\n')
            print('-----------------------------')
            print(f'Subject: {subject}')
            print('\n')

            if subject in self.completed and not self.overwrite:
                continue

            for session in ['ses-001', 'ses-002']:
                d[session] = {}
                path_eeg = Path(f'analysis/data/original/{subject}/{session}/eeg')
                path_fmri= Path(f'analysis/data/original/{subject}/{session}/func')

                # GradCPT is run 1
                files_eeg = glob(str(path_eeg / Path('*.set')))
                files_eeg = sorted(files_eeg, key=self._sort)
                files_dmn = glob(str(path_fmri / Path('DMN_*')))
                files_dmn = sorted(files_dmn, key=self._sort)
                files_dan = glob(str(path_fmri / Path('DAN_*')))
                files_dan = sorted(files_dan, key=self._sort)
                files_dna = glob(str(path_fmri / Path('DNa*')))
                files_dna = sorted(files_dna, key = self._sort)
                files_dnb = glob(str(path_fmri / Path('DNb*')))
                files_dnb = sorted(files_dnb, key = self._sort)
                files_fmri = [files_dan, files_dmn, files_dna, files_dnb]


                if not files_eeg or not files_dan or not files_dmn:
                    mi = self._get_metainfo(files_eeg)
                    message = (f'Missing data for subject {subject} '
                               f'session {session}. Skipping session.')
                    print(message + '\n')
                    self._update_log(message, mi)
                    continue
                    

                if not len(files_eeg) == len(files_dan) or not len(files_eeg) == len(files_dmn):
                    mi = self._get_metainfo(files_eeg)
                    message = ("Number of files detected for EEG not equal to "
                             "number detected for fMRI.\n" 
                            f"Subject: {subject}, Session: {session}, Run: {run}\n"
                            f"EEG files: {files_eeg}\n"
                            f"fMRI files: {files_fmri}\n"
                            "Skipping run.")
                    print(message + '\n')
                    self._update_log(message, mi)
                    continue

                for run, (file_eeg, file_fmri) in enumerate(zip(files_eeg, zip(*files_fmri)), start=1):
                    # file_fmri is a tuple of each of one run files for all
                    # fmri
                    run = 'run-' + str(run).zfill(3)
                    print('\n')
                    print(f'Subject: {subject}, Session: {session}, Run: {run}')
                    print('\n')
                    X = self._process_eeg(file_eeg)

                    # If no EEG data
                    if X is None:
                        continue

                    y_dan, y_dmn, y_dna, y_dnb = self._process_fmri(file_fmri)

                    # Ensure fmri data same observation count
                    if not all([len(y_dan) == len(y_dmn), 
                                len(y_dmn) == len(y_dna), 
                                len(y_dna) == len(y_dnb)]):
                        mi = self._get_metainfo(files_eeg[0])
                        message = ("Number of observations for DMN not "
                                   "equal to those for DAN.\n"
                                   f"Subject: {subject}, Session: {session}, Run: {run}\n"
                                   f"DMN: {len(y_dmn)}\n"
                                   f"DAN: {len(y_dan)}\n"
                                   f"DMNa: {len(y_dna)}\n"
                                   f"DMNb: {len(y_dnb)}\n"
                                   "Skipping run")
                        print(message + '\n')
                        self._update_log(message, mi)
                        continue

                    # Try chopping off last TR
                    if X[:-1, :].shape[0] == len(y_dan):
                        X = X[:-1, :]

                    # Validate equal observations across X and y
                    if X.shape[0] != len(y_dan) or X.shape[0] != len(y_dmn):
                        mi = self._get_metainfo(file_eeg)
                        message = (f"Unequal observations for {mi['subject']} "
                                   f"{mi['session']} {mi['run']}, X: {X.shape[0]}, "
                                   f"y_dan: {len(y_dan)}, "
                                   f"y_dmn: {len(y_dmn)}. Skipping run.")
                        print(message + '\n')
                        self._update_log(message, mi)
                        continue

                    d[session][run] = {'X': X, 'y': {'dan': y_dan, 
                                                     'dmn': y_dmn,
                                                     'dmn_a': y_dna,
                                                     'dmn_b': y_dnb}}

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
        Needs to update to epoch 2 s back from TR marker
        num_lags includes lag 0
        '''

        num_lags = self.num_lags
        # Open EEGlab file
        try:
            raw = mne.io.read_raw_eeglab(file_eeg)
        except FileNotFoundError as exception:

            mi = self._get_metainfo(file_eeg)
            message = (f"Error with {mi['subject']} {mi['session']}"
                       f" {mi['run']}. Message: {exception}. Skipping run.")
            self._update_log(message, mi)

            return None


        # Obtain time-frequency data
        freqs = np.array(range(1, 41))
        njobs = int(os.cpu_count() - 1)
        sfreq = raw.info['sfreq']

        tf_full = tfr_array_morlet(
                raw.get_data()[np.newaxis, :, :],
                sfreq=sfreq,
                freqs = freqs,
                n_cycles=7,
                n_jobs=njobs,
                output='complex')[0]


        # Convert to power
        power = np.abs(tf_full.data) ** 2
        # Normalize within frequency band
        rel_power = power / power.sum(axis=2, keepdims=True)

        # --- DOWNSAMPLE --- #
        # Get TR events
        tr_events, _ = mne.events_from_annotations(raw, event_id={'T  1': 4})
        tr_samples_idxs = tr_events[:, 0]

        # Low pass filter
        order = 4
        cutoff = 0.25
        b, a = butter(4, cutoff / (sfreq / 2), btype='low')
        filtered = filtfilt(b, a, rel_power)

        # Downsample
        tf = filtered[:, :, tr_samples_idxs]
        
        # --- RESHAPE, GET LAGS --- #
        # Reshape (TRs, channels, freqs)
        tf = tf.transpose(2, 0, 1)

        # Get the lags
        lag_ar = np.zeros((tf.shape[0], tf.shape[1], tf.shape[2], num_lags))

        # Only fill in starting at TR 9 so can fully backfill with lags
        for i in range(num_lags-1, tf.shape[0]):
            for lag in range(num_lags):
                lag_ar[i, :, :, lag] = tf[i-lag, :, :]

        # We're starting at TR num_lags-1
        ar = lag_ar[(num_lags-1):, :, :, :]
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


    def _process_fmri(self, files):
        '''
        Input is [file_dan, file_dmn, file_dna, file_dnb]
        Convert eeg file path to a (248, 31 * 40 * 9) array
        '''
        out = []
        num_lags = self.num_lags

        for file in files:

            with open(file, 'r') as f:
                d = f.readlines()

            # Chop off first 8 observations
            d = np.array([float(x.strip()) for x in d])[(num_lags-1):]

            out.append(d)

        return out


if __name__ == '__main__':

    os.chdir(here())
    subjects = sorted([Path(x).name for x in glob('analysis/data/original/*') if 'sub' in x])
    num_lags=11
    reformat = Reformat(subjects, num_lags=num_lags)
    reformat.run()



