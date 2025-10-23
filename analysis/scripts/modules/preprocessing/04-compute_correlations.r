# Import libraries
rm(list=ls())
library(tidyverse)
library(data.table)
library(glue)
library(future.apply)
library(here)
library(fs)
library(arrow)
source(path(here(), 'analysis/scripts/modules/preprocessing/04a-compile_condition_codes.r'))

# Function defs

extract_matrices <- function(run_df, eeg_cols, fmri_cols) {
    # Given a data frame for one run
    # Compute two separate matrices for a given run
    # One for EEG data, one for fMRI data
    # fMRI matrix is (TRs, networks)
    # Returns a list with the EEG and fMRI matrices
    
    eeg_mat <- as.matrix(run_df[, eeg_cols])
    fmri_mat <- as.matrix(run_df[, fmri_cols])
    return(list(eeg=eeg_mat, fmri=fmri_mat))
}

compute_spearman_cor <- function(eeg, fmri) {
    # Given EEG and fMRI matrices
    # Return an fmri_networks X eeg_features matrix of correlations for each run
    
    # Apply column-wise rank
    eeg_rank <- apply(eeg, 2, rank)
    fmri_rank <- apply(fmri, 2, rank)
    
    # Normalize
    eeg_z <- scale(eeg_rank)
    fmri_z <- scale(fmri_rank)
    n_trs <- nrow(eeg)
    
    # Correlate
    cor <- t(fmri_z) %*% eeg_z / (n_trs - 1)
    
    return(cor)
    
}


# Calculate correlations

calculate_correlations <- function(d, by_task=FALSE) {
    
    # Extract modality columns
    first_eeg_col <- colnames(select(d, starts_with('Fp1')))[1]
    eeg_cols <- colnames(d)[which(colnames(d) == first_eeg_col):(ncol(d))]
    fmri_cols <- colnames(d)[(which(colnames(d) == 'tr')+1):(which(colnames(d) == first_eeg_col)-1)]
    
    # Get networks from d
    # Make running by condition conditional on user input
    if (by_task) {
        condition_codes <- get_condition_codes(data_root, network=fmri_cols[1])
        d <- inner_join(d, condition_codes)
        # Rearrange columns
        d <- d[, c(colnames(condition_codes), colnames(d)[!colnames(d) %in% colnames(condition_codes)])]
    }
    
    # Split data by subject, session, run
    if (by_task) {
        d$run_id <- paste(d$subject, d$session, d$run, d$condition, sep = '_')
    } else {
        d$run_id <- paste(d$subject, d$session, d$run, sep = '_')
    }
    
    d_split <- split(d, d$run_id)
    
    # Calculate correlations in parallel
    plan(multisession, workers=parallel::detectCores()-5)
    
    print('Computing correlations...')
    cors <- future_lapply(d_split, function(run_df) {
        mats <- extract_matrices(run_df, eeg_cols, fmri_cols)
        compute_spearman_cor(mats$eeg, mats$fmri)
    })
    
    return(cors)
    
}

format_cors <- function(data, label, by_task=FALSE) {
    # Transpose
    data <- data.frame(t(data))
    
    # Parse metadata
    info <- str_split(label, '_')[[1]]
    subject <- info[1]
    session <- info[2]
    run <- info[3]
    if (by_task) task <- info[4]
    
    # Concatenate
    header <- data.frame(subject=subject, session=session, run=run, eeg_feature=rownames(data))
    if (by_task) header$task <- task
    out <- cbind(header, data)
    return(out)
}

# Look for user command line arguments specifying data path
args <- commandArgs(trailingOnly=TRUE)

by_task <- any(args %in% c('-t', '--bytask'))
data_root <- args[!args %in% c('-t', '--bytask')]

if (length(data_root) == 0) {
    data_root <- path('analysis/data/original')
    setwd(here())
} else{
    data_root <- path(args[1])
}

print(data_root)

# IF YOU ARE RUNNING FROM THE SCRIPT, YOU CAN SET YOUR PATH TO THE DATA HERE
# and uncomment below
# data_root <- 'path/to/data'

# Import data
if (!file.exists(path(data_root, '../correlation_data/merged_data.feather'))) {
    stop('Need to run 03-make_flat_data.py before this script.')
}
d <- read_feather(path(data_root, '../correlation_data/merged_data.feather'))


# Get channel names
ch_names <- suppressWarnings(readLines(path(data_root, '../correlation_data/ch_names.txt')))

# -- GET CORRELATIONS -- # 
cors <- calculate_correlations(d, by_task=by_task)
# Cors is a list where each element has name 'sub-\d\d\d_ses-\d\d\d_run-\d\d\d'
# and shape (fmri_networks, eeg_features)

# -- FORMAT RESULT -- #
# For each (fmri_networks, eeg_features) matrix, transpose it, add it subject/session/run
# info, and concatenate
print('Formatting the result...')
d <- do.call(rbind, lapply(seq_along(cors), function(i) format_cors(cors[[i]], names(cors[i]), by_task=by_task)))

d <- d %>% 
    separate(eeg_feature, into = c('channel', 'frequency', 'lag'), sep = '_') %>%
    mutate(frequency = as.numeric(frequency),
           lag = as.numeric(lag),
           channel = factor(channel, levels = ch_names))

fmri_cols <- colnames(d)[(which(colnames(d) == 'lag')+1):(ncol(d))]
fmri_cols <- fmri_cols[fmri_cols != 'task']

d <- d %>% 
    gather(network, cors, all_of(fmri_cols)) 

# Save
file <- ifelse(by_task, 'correlations_long_bytask.feather', 'correlations_long.feather')
write_feather(d, path(data_root, glue('../correlation_data/{file}')))







