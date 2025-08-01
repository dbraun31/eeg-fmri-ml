# --- RUN THIS SCRIPT TO CONVERT DATA/MERGED_DATA.FEATHER TO SCRIPTS/SANDBOX/CORRELATIONS/CORRELATIONS_LONG*.CSV -- #
# (run before running visualize correlations.r) #
# (computationally intensive) #

rm(list=ls())
library(data.table)
library(arrow)
library(tidyverse)
library(reticulate)
library(fs)
library(here)
setwd(path(here(), 'analysis'))
root <- path('scripts/sandbox/correlations')

# Get channel names from raw data
use_condaenv('eeg-fmri')
py_run_string("
import mne
raw = mne.io.read_raw_eeglab('data/original/sub-001/ses-001/eeg/sub-001_ses-001_bld001_eeg_Bergen_CWreg_filt_ICA_rej.set')
ch_names = raw.info['ch_names']
")
ch_names <- py$ch_names

# Make channel order more logical
ch_names <- c(ch_names[grepl('^F', ch_names)], ch_names[grepl('^T', ch_names)],
              ch_names[grepl('^C', ch_names)], ch_names[grepl('^P', ch_names)],
              ch_names[grepl('^O', ch_names)])

# Import merged data
d <- data.table(read_feather('data/merged_data.feather'))

# Assumes first feature column is 'Fp1_1_0'
voltage_cols <- colnames(d)[which(colnames(d)=='Fp1_1_0'):length(colnames(d))]

# Big data table energy (expensive)
# Run correlations
result_run <- d[,
                .(dmn_cors = list(sapply(.SD, function(col) cor(dmn, col, use = 'pairwise.complete.obs', method='spearman'))),
                  dan_cors = list(sapply(.SD, function(col) cor(dan, col, use = 'pairwise.complete.obs', method='spearman'))),
                  dmna_cors = list(sapply(.SD, function(col) cor(dmn_a, col, use = 'pairwise.complete.obs', method='spearman'))),
                  dmnb_cors = list(sapply(.SD, function(col) cor(dmn_b, col, use = 'pairwise.complete.obs', method='spearman')))),
                by = .(subject, session, run),
                .SDcols = voltage_cols
]
# Average by run
result <- result_run[
    ,
    .(mean_dmn = list(Reduce(`+`, dmn_cors) / length(dmn_cors)),
      mean_dan = list(Reduce(`+`, dan_cors) / length(dan_cors)),
      mean_dmna = list(Reduce(`+`, dmna_cors) / length(dmna_cors)),
      mean_dmnb = list(Reduce(`+`, dmnb_cors) / length(dmnb_cors))),
    by = subject
][
    # Unlist
    ,
    .(feature = voltage_cols, dmn_cors = unlist(mean_dmn),
      dan_cors = unlist(mean_dan),
      dmna_cors = unlist(mean_dmna),
      dmnb_cors = unlist(mean_dmnb)),
    by = subject
]

# Unlist
result_run <- result_run[
    ,
    .(feature = voltage_cols,
      dan_cors = unlist(dan_cors),
      dmn_cors = unlist(dmn_cors),
      dmna_cors = unlist(dmna_cors),
      dmnb_cors = unlist(dmnb_cors)),
    by = .(subject, session, run)
]

# Final formatting
result <- result %>% 
    separate(feature, into = c('channel', 'frequency', 'lag'), sep = '_') %>% 
    gather(region, cors, dmn_cors, dan_cors, dmna_cors, dmnb_cors) %>% 
    mutate(region = str_replace(region, '_cors', ''),
           lag = as.integer(lag),
           frequency = as.integer(frequency),
           channel = factor(channel, levels=ch_names)) 

result_run <- result_run %>% 
    separate(feature, into = c('channel', 'frequency', 'lag'), sep = '_') %>% 
    gather(region, cors, dmn_cors, dan_cors, dmna_cors, dmnb_cors) %>% 
    mutate(region = str_replace(region, '_cors', ''),
           lag = as.integer(lag),
           frequency = as.integer(frequency),
           channel = factor(channel, levels=ch_names)) 

write.csv(result_run, path(root, 'correlations_long_byrun.csv'), row.names=FALSE)
write.csv(result, path(root, 'correlations_long.csv'), row.names=FALSE)