# EEG-fMRI Preprocessing Pipeline

This quick reference outlines the core preprocessing pipeline for EEG and
fMRI data.

## Data Requirements

The pipeline expects raw EEG and fMRI data organized as:

```
analysis/data/original/
├── sub-001/
│ ├── ses-001/
│ │ ├── eeg/
│ │ │ └── .set
│ │ └── func/
│ │ └── {network}_.txt
│ └── ses-002/
│ ├── eeg/
│ └── func/
├── sub-002/
└── ...
```
This structure matches the format of the "Dimitrios" data on OneDrive/Sharepoint.

- EEG files must be in EEGLAB `.set` format.
- fMRI files must be text files per network per run, named with the
    network as the prefix.

## Pipeline Overview

1. **Reformat Stage (`01-format_data.py`)**
   - Converts EEG to time-frequency arrays with lags.
   - Aligns EEG (`X`) with fMRI network signals (`y`) for each run.
   - Saves outputs in `analysis/data/formatted/full/`.

2. **Reduce Stage (`02-reduce_data.py`)**
   - Aggregates EEG frequencies into canonical bands.
   - Reduces number of lags considered.
   - Reshapes data into a pivoted matrix suitable for modeling.
   - Saves outputs in `analysis/data/formatted/reduced/`.

**To execute:**  

``` bash
# Reformat data
# (If no path provided, script assumes analysis/data/original)
python 01-format_data.py path/to/data
# Reduce formatted data
python 02-reduce_data.py
```

## Output

After running the pipeline, you will find:
```
analysis/data/formatted/
├── full/
│ └── sub-001.pkl
├── reduced/
│ └── sub-001.pkl
```

- **Full:** raw EEG-fMRI arrays per session/run.  
- **Reduced:** band-aggregated and lag-reduced EEG arrays aligned with fMRI signals.

Each `.pkl` contains:

```python
{
    'ses-001': {
        'run-001': {
            'X': np.ndarray,  # EEG features
            'y': dict         # fMRI network signals
        },
    },
}
```


## Logging

- `01-format_data.py` logs skipped runs, missing files, or mismatched observations.  
- Logs are saved in `analysis/scripts/modules/preprocessing/logs/` with a timestamped filename.

