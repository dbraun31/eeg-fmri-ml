import mne
from mne.time_frequency import tfr_array_morlet
import numpy as np
import os

raw = mne.io.read_raw_eeglab('sub-015_ses-002_bld002_eeg_Bergen_CWreg_filt_ICA_rej.set')
sfreq = raw.info['sfreq']
raw = raw.get_data()

time_range = 2
np.random.seed(42)
start = np.random.choice(range(len(raw)))
end = int(start + time_range * sfreq)


freqs = np.logspace(np.log10(1), np.log10(40), 40)
n_cycles = freqs / 2

power = tfr_array_morlet(
        raw[np.newaxis, :, :],
        freqs=freqs,
        n_cycles=n_cycles,
        sfreq=sfreq,
        n_jobs=os.cpu_count()-1,
        output='power')[0]

raw_s = raw.transpose()[np.arange(start, end)]
power = power.transpose(2, 0, 1)
power_s = power[np.arange(start, end)]

np.save('timeseries.npy', raw_s)
np.save('power.npy', power_s)
