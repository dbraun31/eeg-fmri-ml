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

        # Hardcode fmri networks
        self.fmri_networks = ['DAN', 'DANa', 'DANb', 'DMN', 'DNa', 'DNb', 'SAL', 'FPCNa', 'FPCNb']


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

                files_eeg = glob(str(path_eeg / Path('*.set')))
                files_eeg = sorted(files_eeg, key=self._sort)

                files_fmri = {}
                for network in self.fmri_networks:
                    # Assuming network name is start of filename
                    out = glob(str(path_fmri / Path(f'{network}_*')))
                    files_fmri[network] = sorted(out, key=self._sort)
                    
                if not files_eeg or not all([x for x in files_fmri]):
                    message = (f'Missing data for subject {subject} '
                               f'session {session}. Skipping session.')
                    print(message + '\n')
                    self._update_log(message)
                    continue
                    

                fmri_lens = [len(x) for x in files_fmri.values()]
                if not all([fmri_lens[0] == x for x in fmri_lens]) or len(files_eeg) != fmri_lens[0]:
                    message = ("Number of files detected for EEG not equal to "
                             "number detected for fMRI.\n" 
                            f"Subject: {subject}, Session: {session}, Run: {run}\n"
                            f"EEG files: {files_eeg}\n"
                            f"fMRI files: {files_fmri}\n"
                            "Skipping run.")
                    print(message + '\n')
                    self._update_log(message)
                    continue

                for run, (file_eeg, file_fmri) in enumerate(zip(files_eeg, zip(*files_fmri.values())), start=1):

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

                    # Hard assumption that network name is first arg in
                    # filename, separated with _
                    network_names = [Path(x).stem.split('_')[0] for x in file_fmri]
                    # y is dict of {network: np.array, ...}
                    y = self._process_fmri(file_fmri, network_names)

                    # Ensure fmri data same observation count
                    if not all([len(x) == len(list(y.values())[0]) for x in y.values()]):
                        message = ("Number of observations across fmri "
                                   "networks is not equal to one another.\n"
                                   f"Subject: {subject}, Session: {session}, Run: {run}\n")
                        for network in y:
                            message += f'{network}: {len(y[network])}\n'
                        message += "Skipping run"
                        print(message + '\n')
                        self._update_log(message)
                        continue

                    # Try chopping off last TR
                    if X[:-1, :].shape[0] == len(y['DAN']):
                        X = X[:-1, :]

                    # Validate equal observations across X and y
                    if X.shape[0] != len(y['DMN']):
                        message = (f"Unequal observations for {subject} "
                                   f"{session} {run}, X: {X.shape[0]}, ")
                        for network in y:
                            message += f'{network}: {len(y[network])}, '
                        message += 'Skipping run.'
                        print(message + '\n')
                        self._update_log(message)
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
        Convert eeg file path to a (N timepoints, 31 * 40 * 9) array
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
            self._update_log(message)

            return None


        # Obtain time-frequency data
        freqs = np.array(range(1, 41))
        njobs = int(os.cpu_count() - 1)
        sfreq = raw.info['sfreq']

        power = tfr_array_morlet(
                raw.get_data()[np.newaxis, :, :],
                sfreq=sfreq,
                freqs = freqs,
                n_cycles=7,
                n_jobs=njobs,
                output='power')[0]

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



    def _update_log(self, message):
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

    def _process_fmri(self, files, network_names):
        '''
        Input is tuple of paths to each network for one run
        Return dict of np.array, chopping off the correct lag numbers
        '''
        out = {}
        num_lags = self.num_lags

        for file, name in zip(files, network_names):

            with open(file, 'r') as f:
                d = f.readlines()

            # Chop off first k observations
            d = np.array([float(x.strip()) for x in d])[(num_lags-1):]

            out[name] = d

        return out


if __name__ == '__main__':

    os.chdir(here())
    subjects = sorted([Path(x).name for x in glob('analysis/data/original/*') if 'sub' in x])
    num_lags=11
    reformat = Reformat(subjects, num_lags=num_lags)
    reformat.run()



