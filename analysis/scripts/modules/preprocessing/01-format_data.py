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

# Dev
from IPython import embed

class Reformat:
    """
    Reformat EEG and fMRI data into structured arrays suitable for modeling and analysis.

    This class processes multimodal EEG and fMRI datasets for multiple subjects,
    aligning the modalities across sessions and runs. It computes time–frequency
    EEG features using Morlet wavelets, applies time lags, and synchronizes them
    with network-level fMRI signals. The results are organized into nested
    dictionaries and serialized as pickle files for downstream modeling.

    Parameters
    ----------
    in_path : pathlib.Path
        Path to the root directory containing raw subject/session data.
    out_path : pathlib.Path
        Path to the directory where processed data and frequency files will be saved.
    fmri_networks : list of str
        List of fMRI network names to extract (e.g., ['DANa', 'DANb', 'DNa', 'DNb', 'SAL', 'FPCNa', 'FPCNb']).
    n_lags : int, default=11
        Number of time lags to include for EEG features (including lag 0).
    overwrite : bool, default=False
        If True, reprocess and overwrite any previously formatted subject data.
    n_freq : int, default=40
        Number of frequency bins for EEG time–frequency decomposition.

    Attributes
    ----------
    subjects : list of str
        Subject identifiers automatically inferred from `in_path` using the pattern 'sub[-_][0-9]*'.
    completed : list of str
        Subjects already processed (based on existing output files).
    freqs : np.ndarray
        Array of frequencies used for Morlet decomposition (written to `../freqs.txt` after processing).
    time : str
        Timestamp for the current processing session, used in log file names.

    Methods
    -------
    run()
        Iterates over subjects, sessions, and runs; processes EEG and fMRI data,
        aligns them, and saves structured outputs.
    _process_eeg(file_eeg)
        Loads and preprocesses EEG data: computes time–frequency power, normalizes,
        filters, downsamples to TRs, applies time lags, and reshapes features.
    _process_fmri(files, network_names)
        Loads and trims fMRI network signals, removing initial volumes to match
        EEG lag structure.
    _write_data(d)
        Saves the processed subject dictionary to a pickle file in `out_path`.
    _sort(run)
        Extracts the integer run index from a filename (supports both 'run-' and 'bld' patterns).
    _get_metainfo(file)
        Extracts subject, session, and run identifiers from a file path string.
    _update_log(message)
        Writes a timestamped message to a log file under
        `analysis/scripts/modules/preprocessing/logs`.

    Notes
    -----
    - EEG data are assumed to be in EEGLAB `.set` format with TR markers labeled 'T  1'.
    - fMRI data are assumed to be plain-text files where filenames begin with
      a network name (e.g., 'DANa_run-001.txt'), and each line corresponds to
      one observation.
    - Each subject directory should follow the structure:
        sub-XXX/ses-001/{eeg,func}/
        sub-XXX/ses-002/{eeg,func}/
    - The number of EEG and fMRI runs per session must be consistent; mismatches
      trigger log messages and are skipped.
    - Output format:
        {session: {run: {'X': EEG_features, 'y': {network: fMRI_signal}}}}
    - The file `freqs.txt` is saved one directory above `out_path` and contains
      the frequency bins used for EEG decomposition.
    """

    def __init__(self,  in_path, out_path, fmri_networks, freq_space='log', 
                 n_lags=11, overwrite=False, n_freq=40):


        # Infer subject numbers
        subjects = sorted([Path(x).name for x in glob(str(in_path / Path('sub[-_][0-9]*')))])
        if not subjects and in_path != Path('analysis/data/original'):
            raise RuntimeError('Could not find data at default location '
                               '(analysis/data/original). Please specify '
                               'data path with: python 01-format_data.py '
                               'path/to/data')
        if not subjects:
            raise RuntimeError('Could not find subjects in the provided '
                               'data path. Ensure subject directories are '
                               'labeled within data path as "sub-\d\d\d" ')

        # Initializing class-level variables
        if freq_space == 'linear':
            freqs = np.linspace(1, 40, 40)
        elif freq_space == 'log':
            freqs = np.logspace(np.log10(1), np.log10(40), 40)
        else:
            raise ValueError('freq_space must be either "linear" or "log"')
        self.freqs = freqs
        self.subjects = subjects
        now = datetime.now()
        self.time = now.strftime('%Y%m%d%H%M%S')
        self.in_path = in_path
        self.out_path = out_path
        self.num_lags = n_lags
        self.n_freq = n_freq
        self.fmri_networks = fmri_networks
        self.out_path.mkdir(parents=True, exist_ok=True)
        completed = glob(str(self.out_path / Path('sub')) + '*')
        self.completed = [Path(x).stem for x in completed]
        self.overwrite = overwrite


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
                            f"Subject: {subject}, Session: {session}\n"
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

                    # Convert fmri files for each network to a dict with:
                    # {'network1': np.array, 'network2': ...}
                    # see _process_fmri function below
                    y = self._process_fmri(file_fmri, self.fmri_networks)

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

                    # Handle observation alignment
                    self.subject = subject
                    self.session = session
                    self.run = run
                    X, y = self._align_observations(X, y)

                    # If observations are misaligned
                    if X is None:
                        continue

                    # Save TR marker
                    trs = np.arange(self.num_lags, y[self.fmri_networks[0]].shape[0]+self.num_lags)
                    assert(len(trs) == len(y[self.fmri_networks[0]]))

                    # -- DROP NANS -- #
                    
                    # Mask nans in fMRI data and drop in EEG
                    mask = ~np.isnan(y[self.fmri_networks[0]])
                    X = X[mask, :]
                    y = {k: y[k][mask] for k in y}
                    trs = trs[mask]

                    d[session][run] = {'X': X, 'y': y, 'trs': trs}

            self.subject = subject
            self._write_data(d)


        # Write out frequencies
        # Frequencies will get stored in ./formatted
        out_file = self.out_path / Path('../freqs.txt')
        with open(out_file, 'w') as file:
            file.write('\n'.join([str(x) for x in self.freqs]))

        # Write out number of lags
        out_file = self.out_path / Path('../num_lags.txt')
        with open(out_file, 'w') as file:
            file.write(str(self.num_lags))




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
        n_cycles = self.freqs / 2
        njobs = int(os.cpu_count() - 1)
        sfreq = raw.info['sfreq']

        # Use mne morlet decomposition function
        power = tfr_array_morlet(
                # Adds a bogus new dimension because the function needs an epoch
                raw.get_data()[np.newaxis, :, :], 
                sfreq=sfreq,
                freqs = self.freqs,
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

        # Only fill in starting at TR num_lags-1 so can fully backfill with lags
        for i in range(num_lags-1, tf.shape[0]):
            for lag in range(num_lags):
                lag_ar[i, :, :, lag] = tf[i-lag, :, :]

        # We're starting at TR num_lags-1
        ar = lag_ar[(num_lags-1):, :, :, :]
        # Reshape to (248, 31 * 40 * 9)
        ar_reshape = ar.reshape(ar.shape[0], -1)

        return ar_reshape


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

    def _align_observations(self, X, y):

        # If already aligned, return
        if X.shape[0] == len(y[self.fmri_networks[0]]):
            return X, y

        # The most general case: Extra marker in EEG data for ES runs
        eeg_extra = X.shape[0] - len(y[self.fmri_networks[0]])
        if eeg_extra in [1, 2]:
            # Chop an EEG TR off and return
            X = X[:-eeg_extra, :]
            assert(X.shape[0] == len(y[self.fmri_networks[0]]))
            return X, y

        # If EEG obs are short by 1 or 2, chop off fMRI to align
        fmri_extra =  len(y[self.fmri_networks[0]]) - X.shape[0]
        if fmri_extra in [1, 2]:
            y = {x: y[x][:-fmri_extra] for x in y}
            assert(X.shape[0] == len(y[self.fmri_networks[0]]))
            return X, y

        # If obs are still not equal, log and skip run
        if X.shape[0] != len(y[self.fmri_networks[0]]):
            message = (f"Unequal observations for {self.subject} "
                       f"{self.session} {self.run}, X: {X.shape[0]}, ")
            for network in y:
                message += f'{network}: {len(y[network])}, '
            message += 'Skipping run.'
            print(message + '\n')
            self._update_log(message)

            return None, None


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

# This tells Python to execute the below if this script is called directly
# from the command line
if __name__ == '__main__':
    '''
    Usage: python 01-format_data.py path/to/input_data path/to/output
        specifying paths is optional
        if not specified, it will look in ./analysis/data/original
        (relative to repo root)
    '''

    # fmri networks
    fmri_networks = ['dATNa', 'dATNb', 'DNa', 'DNb', 'SAL', 'FPCNa', 'FPCNb']

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



    # Initialize and run reformatting
    reformat = Reformat(fmri_networks=fmri_networks, in_path=in_path,
                        out_path=out_path, overwrite=overwrite)
    reformat.run()



