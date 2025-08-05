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
saveRDS(ch_names, path(root, 'ch_names.rds'))
# Import merged data
d <- data.table(read_feather('data/merged_data.feather'))

# Assumes first feature column is 'Fp1_1_0'
voltage_cols <- colnames(d)[which(colnames(d)=='Fp1_1_0'):length(colnames(d))]

d <- d[, dan_dmna_diff := dan - dmn_a]


# Big data table energy (expensive)
# Run correlations
cor_func <- function(ref_col, data) {
    sapply(data, function(col) cor(ref_col, col, use = 'pairwise.complete.obs', method='spearman'))
}
    
result_run <- d[,
                .(dmn_cors = list(cor_func(dmn, .SD)),
                  dan_cors = list(cor_func(dan, .SD)),
                  dmna_cors = list(cor_func(dmn_a, .SD)),
                  dmnb_cors = list(cor_func(dmn_b, .SD)),
                  diff_cors = list(cor_func(dan_dmna_diff, .SD))),
                by = .(subject, session, run),
                .SDcols = voltage_cols
]


# Unlist
result_run <- result_run[
    ,
    .(feature = voltage_cols,
      dan_cors = unlist(dan_cors),
      dmn_cors = unlist(dmn_cors),
      dmna_cors = unlist(dmna_cors),
      dmnb_cors = unlist(dmnb_cors),
      diff_cors = unlist(diff_cors)),
    by = .(subject, session, run)
]

# Final formatting

result_run <- result_run %>% 
    separate(feature, into = c('channel', 'frequency', 'lag'), sep = '_') %>% 
    gather(region, cors, dmn_cors, dan_cors, dmna_cors, dmnb_cors, diff_cors) %>% 
    mutate(region = str_replace(region, '_cors', ''),
           lag = as.integer(lag),
           frequency = as.integer(frequency),
           channel = factor(channel, levels=ch_names)) 

result <- result_run %>% 
    group_by(subject, channel, lag, frequency, region) %>% 
    summarize(cors = mean(cors))

write.csv(result_run, path(root, 'correlations_long_byrun.csv'), row.names=FALSE)
write.csv(result, path(root, 'correlations_long.csv'), row.names=FALSE)



