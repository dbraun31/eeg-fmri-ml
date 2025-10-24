# Import libraries
rm(list=ls())
library(tidyverse)
library(data.table)
library(glue)
library(future.apply)
library(here)
library(fs)
library(arrow)
script_root <- path(here(), 'analysis/scripts/modules/preprocessing')
source(path(script_root, '04a-compile_condition_codes.r'))

# --- FUNCTION DEFINITIONS --- #
# ---------------------------- #

# Calculate correlations

compute_correlations <- function(d, script_root, by_task=FALSE) {
    
    setDT(d)
    
    # Make / refresh cache dir
    if (dir.exists(path(script_root, 'cache'))) {
        dir_delete(path(script_root, 'cache'))
        dir.create(path(script_root, 'cache'))
    }
    
    # Identify EEG and fMRI columns
    first_eeg_col <- colnames(d)[which(grepl('^Fp1', colnames(d)))[1]]
    if (is.na(first_eeg_col)) stop('No EEG column starting with "Fp1" found.')
    eeg_cols <- colnames(d)[which(colnames(d) == first_eeg_col):(ncol(d))]
    fmri_cols <- colnames(d)[(which(colnames(d) == 'tr')+1):(which(colnames(d) == first_eeg_col)-1)]
    
    # Get networks from d
    # Make running by condition conditional on user input
    if (by_task) {
        condition_codes <- get_condition_codes(data_root, network=fmri_cols[1])
        d <- d[condition_codes, on = c('subject', 'session', 'run', 'tr'), nomatch = 0]
        # Rearrange columns
        d <- d[, c(colnames(condition_codes), colnames(d)[!colnames(d) %in% colnames(condition_codes)]), with = FALSE]
    }
    
    # Split data by subject, session, run
    if (by_task) {
        d[, run_id := paste(subject, session, run, condition, sep = '_')]
    } else {
        d[, run_id := paste(subject, session, run, sep = '_')]
    }
    
    # Get unique run_ids to distribute to workers
    run_ids <- unique(d$run_id)

    # Plan parallelization
    workers <- max(1, parallel::detectCores()-10)
    if (interactive() && .Platform$OS.type == 'unix') {
        # Interactive RStudio, use safer multisession
        plan(multisession, workers = 1)
    } else if (.Platform$OS.type == 'unix') {
        plan(multicore, workers=workers)
    } else {
        plan(multisession, workers = min(workers, 2))
    }

    # Increase globals size to avoid serialization errors
    options(future.globals.maxSize = 16 * 1024^3) # 16 GB
    
    print('Computing correlations...')

    future_lapply(run_ids, function(rid) {
        run_df <- d[run_id == rid]
        eeg_mat <- as.matrix(run_df[, ..eeg_cols])
        fmri_mat <- as.matrix(run_df[, ..fmri_cols])
        cor_mat <- cor(eeg_mat, fmri_mat, method = 'spearman')
        formatted <- format_cors(cor_mat, rid, by_task = by_task)
        write_feather(formatted, path(script_root, glue('cache/{rid}.feather')))
        'Worker completed successfully'
    }, future.seed = FALSE)

}

format_cors <- function(data, label, by_task=FALSE) {
    
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

# --- end function definitions --- #
# -------------------------------- #


# --- START INPUT PARSING / INTERACTIVE SCRIPT --- #
# ------------------------------------------------ #


# --- INPUT PARSING --- # 

args <- commandArgs(trailingOnly=TRUE)

by_task <- any(args %in% c('-t', '--bytask'))
data_root <- args[!args %in% c('-t', '--bytask')]

if (length(data_root) == 0) {
    data_root <- path('analysis/data/original')
    setwd(here())
} else{
    data_root <- path(args[1])
}

# --- INTERACTIVE --- # 
# if you are running interactively (eg, rstudio) and want a custom data path, 
# set your path to the data by uncommenting below
# data_root <- 'path/to/data'

# --- IMPORT DATA --- #
if (!file.exists(path(data_root, '../correlation_data/merged_data.feather'))) {
    stop('Need to run 03-make_flat_data.py before this script.')
}
d <- read_feather(path(data_root, '../correlation_data/merged_data.feather'))

# Get channel names
ch_names <- suppressWarnings(readLines(path(data_root, '../correlation_data/ch_names.txt')))

# -- GET CORRELATIONS -- # 
compute_correlations(d, script_root, by_task)

# -- FORMAT RESULT -- #
# For each (fmri_networks, eeg_features) matrix, transpose it, add it subject/session/run
# info, and concatenate
print('Formatting the result...')
rm(d)
gc()
cache_dir <- path(script_root, 'cache')
files <- dir_ls(cache_dir, glob = '*.feather')

d <- do.call(rbind, lapply(files, read_feather))

d <- d %>% 
    separate(eeg_feature, into = c('channel', 'frequency', 'lag'), sep = '_') %>%
    mutate(frequency = as.numeric(frequency),
           lag = as.numeric(lag),
           channel = factor(channel, levels = ch_names))

fmri_cols <- colnames(d)[(which(colnames(d) == 'lag')+1):(ncol(d))]
fmri_cols <- fmri_cols[fmri_cols != 'task']

d <- d %>% 
    gather(network, cors, all_of(fmri_cols)) 

# --- WRITE TO FILE --- #
file <- ifelse(by_task, 'correlations_long_bytask.feather', 'correlations_long.feather')
write_feather(d, path(data_root, glue('../correlation_data/{file}')))


# Clear cache
dir_delete(cache_dir)
rm(list=ls())
gc()





