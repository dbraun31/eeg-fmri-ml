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
import sys

class Reformat:
    """
    Reformat EEG and fMRI data into structured arrays suitable for analysis.

    This class processes EEG and fMRI datasets for a list of subjects, organizes 
    the data by session and run, computes time-lagged features for EEG, and aligns 
    them with fMRI network data. The processed data is saved as pickle files.

    Parameters
    ----------
    subjects : list of str
        List of subject identifiers to process.
    n_lags : int
        Number of lags to include for EEG time series (including lag 0).
    in_path : pathlib.Path
        Path to the root input directory containing subject/session data.
    out_path : pathlib.Path
        Path to the directory where processed data will be saved.
    overwrite : bool, default=False
        If True, previously processed subject files will be overwritten.
    n_freq : int, default=40
        Number of frequency bins to use for EEG time-frequency analysis.

    Methods
    -------
    run()
        Process all subjects, sessions, and runs, performing EEG and fMRI 
        preprocessing and saving the results.
    _write_data(d)
        Saves processed data dictionary `d` to a pickle file.
    _sort(run)
        Returns the integer run index from a filename for sorting purposes.
    _process_eeg(file_eeg)
        Loads EEG data, computes time-frequency power, applies lags, and 
        returns a reshaped array of shape (timepoints, channels*freqs*lags).
    _process_fmri(files, network_names)
        Loads fMRI network data from files, trims initial lags, and returns a 
        dictionary mapping network names to numpy arrays.
    _update_log(message)
        Writes or appends a log message to a timestamped log file.
    _get_metainfo(file)
        Extracts subject, session, and run identifiers from a file path.

    Notes
    -----
    - Assumes EEG data is in EEGLAB `.set` format.
    - Assumes fMRI data is stored as text files with one observation per line, 
      and that filenames start with the network name.
    - The number of EEG timepoints (after lags) must match the number of fMRI 
      observations; otherwise, the run is skipped.
    - Currently hardcoded fMRI networks: 'DAN', 'DANa', 'DANb', 'DMN', 'DNa', 
      'DNb', 'SAL', 'FPCNa', 'FPCNb'.
    """

    def __init__(self, subjects, n_lags, in_path, out_path, overwrite=False, n_freq=40):

        # Initializing class-level variables
        self.subjects = subjects
        now = datetime.now()
        self.time = now.strftime('%Y%m%d%H%M%S')
        self.in_path = in_path
        self.out_path = out_path
        self.num_lags = n_lags
        self.n_freq = n_freq
        self.out_path.mkdir(parents=True, exist_ok=True)
        completed = glob(str(self.out_path / Path('sub')) + '*')
        self.completed = [Path(x).stem for x in completed]
        self.overwrite = overwrite

        # Hardcode fmri networks
        self.fmri_networks = ['DAN', 'DANa', 'DANb', 'DMN', 'DNa', 'DNb', 'SAL', 'FPCNa', 'FPCNb']


    def run(self):

        subjects = self.subjects

        # Iterate over subjects
        for subject in subjects:
            
            d = {}
            print('\n')
            print('-----------------------------')
            print(f'Subject: {subject}')
            print('\n')

            # If the subject is already formatted and overwrite=False, skip
            if subject in self.completed and not self.overwrite:
                continue

            # Iterate over sessions
            for session in ['ses-001', 'ses-002']:
                d[session] = {}
                path_eeg = self.in_path / Path(f'{subject}/{session}/eeg')
                path_fmri= self.in_path / Path(f'{subject}/{session}/func')

                # Retrieve data files for each modality
                files_eeg = glob(str(path_eeg / Path('*.set')))
                files_eeg = sorted(files_eeg, key=self._sort)

                # Build a {'network': [path to network runs txt], ...}
                # Where the list of txts per network is ordered by run
                files_fmri = {}
                for network in self.fmri_networks:
                    # Assuming network name is start of filename
                    out = glob(str(path_fmri / Path(f'{network}_*')))
                    files_fmri[network] = sorted(out, key=self._sort)
                    
                # Check for missing data files
                if not files_eeg or not all([x for x in files_fmri]):
                    message = (f'Missing data for subject {subject} '
                               f'session {session}. Skipping session.')
                    print(message + '\n')
                    self._update_log(message)
                    continue
                    
                # Check for inconsistent number of runs across networks / eeg data 
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

                # Iterate over runs
                for run, (file_eeg, file_fmri) in enumerate(zip(files_eeg, zip(*files_fmri.values())), start=1):
                    # - file_eeg is the (string) path to the EEG .set file for the current run
                    # - file_fmri is a tuple of (string) paths to the txt 
                    #    data for each network in the current run

                    run = 'run-' + str(run).zfill(3)
                    print('\n')
                    print(f'Subject: {subject}, Session: {session}, Run: {run}')
                    print('\n')

                    # Convert EEG .set path to formatted X array
                    # see _process_eeg function below
                    X = self._process_eeg(file_eeg)

                    # If no EEG data, skip the run
                    if X is None:
                        continue

                    # NETWORK NAME INFERENCE
                    # Hard assumption that network name is first arg in
                    # filename, separated with _
                    network_names = [Path(x).stem.split('_')[0] for x in file_fmri]

                    # Ensure the assumption above checks out
                    # This will throw an error if there are differences
                    # even in capitalization or spelling between the
                    # network names in the fmri txt data and the hard coded
                    # networks above (~ line 85)

                    try:
                        assert(sorted(network_names) == sorted(self.fmri_networks))
                    except Exception as e:
                        print('Inferred fMRI network names do not match '
                        'hard coded network names.')
                        print(f'Inferred names: {sorted(network_names)}')
                        print('\n')
                        print(f'Coded names: {sorted(self.fmri_networks)}')
                        print('\n')
                        print('Quitting...')
                        sys.exit(1)

                    # Convert fmri files for each network to a dict with:
                    # {'network1': np.array, 'network2': ...}
                    # see _process_fmri function below
                    y = self._process_fmri(file_fmri, network_names)

                    # Ensure fmri data have same observation count
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

                    # Try chopping off last TR in EEG if it makes obs equal
                    # with fMRI
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
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        out_file = self.out_path / Path(self.subject + '.pkl')
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
        """
        Process a single EEG file and convert it into a time-lagged feature array.

        This method reads an EEGLAB `.set` file, computes time-frequency power 
        across channels and frequencies using Morlet wavelets, normalizes the 
        power within each frequency band, low-pass filters the result, downsamples 
        to TR events, and constructs a time-lagged representation for modeling.

        Parameters
        ----------
        file_eeg : str or pathlib.Path
            Path to the EEG file to process.

        Returns
        -------
        np.ndarray or None
            A 2D array of shape (n_timepoints, n_channels * n_freqs * n_lags) 
            containing the EEG features for each timepoint. Returns `None` if 
            the file cannot be read or processing fails.

        Notes
        -----
        - The number of lags (`self.num_lags`) is included in the output shape; 
          lag 0 corresponds to the current timepoint.
        - Frequency decomposition uses 40 logarithmically spaced bins from 1–40 Hz 
          by default (`n_freq=40`).
        - Assumes TR events are marked with annotation 'T  1' in the EEGLAB file.
        - The output array starts at TR index `num_lags-1` to ensure all lags are 
          available.
        - The EEG array is reshaped to combine channels, frequencies, and lags 
          into a single feature dimension.
        - If any file is missing or corrupted, a message is logged via `_update_log`
          and the function returns `None`.
        """

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


        # OBTAIN TIME-FREQUENCY DATA
        # Set params
        freqs = np.logspace(0, np.log10(40), 40)
        n_cycles = freqs / 2
        njobs = int(os.cpu_count() - 1)
        sfreq = raw.info['sfreq']

        # Use mne morlet decomposition function
        power = tfr_array_morlet(
                # Adds a bogus new dimension because the function needs an epoch
                raw.get_data()[np.newaxis, :, :], 
                sfreq=sfreq,
                freqs = freqs,
                n_cycles=n_cycles,
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

        log_path = here() / Path('analysis/scripts/modules/preprocessing/logs')
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
    
    '''
    THIS IS WHERE IT STARTS RUNNING WHEN YOU FIRST CALL THE SCRIPT
    Here I'm doing basic input argument parsing and calling the functions
    above
    '''

    # This controls how many lags the formatted data has
    # includes lag 0, so (eg) n_lags=11 means current TR plus the ten
    # previous
    n_lags=11

    # Prompt user for overwrite arg
    valid_response = False
    while not valid_response:
        raw = input('\nDo you want to overwrite existing data? (y/n): ')
        print('\n')
        response = raw.strip().lower()
        if response in ['y', 'n']:
            valid_response = True
            overwrite = True if response=='y' else False


    # Parse input commands and change WD if needed
    args = sys.argv[1:]
    if not args:
        in_path = Path('analysis/data/original')
        out_path = Path('analysis/data/formatted/full')

        # Change WD to project root only if user hasn't specified paths
        os.chdir(here())
        sys.path.append(str(here()))

    else:
        in_path = Path(args[0])
        if len(args) > 1:
            out_path = Path(args[1])
        else:
            out_path = in_path / Path('../formatted/full')

    # Infer subject numbers
    subjects = sorted([Path(x).name for x in glob(str(in_path / Path('*'))) if 'sub' in x])

    if not subjects and not args:
        raise RuntimeError('Could not find data at default location '
                           '(analysis/data/original). Please specify '
                           'data path with: python 01-format_data.py '
                           'path/to/data')
    if not subjects:
        raise RuntimeError('Could not find subjects in the provided '
                           'data path. Ensure subject directories are '
                           'labeled within data path as "sub-\d\d\d" ')

    # Initialize and run reformatting
    reformat = Reformat(subjects, n_lags=n_lags, in_path=in_path,
                        out_path=out_path, overwrite=overwrite)
    reformat.run()



