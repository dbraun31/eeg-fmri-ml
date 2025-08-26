# --- DATA AND LIBRARIES --- #
rm(list=ls())
library(arrow)
library(ggridges)
library(tidyverse)
library(eegUtils)
library(data.table)
library(ggpubr)
library(here)
library(fs)
library(scales)
library(RColorBrewer)
library(reticulate)
setwd(path(here()))
root <- path('analysis/scripts/sandbox/correlations')
size <- 16

if (file.exists(path(root, 'correlations_long.feather'))) {
    result <- read_feather(path(root, 'correlations_long.feather'))
} else {
    stop('correlations_long.csv is missing. First run make_flat_data.py, then run make_long_data.r')
}

ch_names <- readRDS(path(root, 'ch_names.rds'))

# Adjust the lag var to (s)
result_run <- result %>% 
    mutate(lag = lag * 2)

result <- result_run %>% 
    group_by(subject, channel, frequency, lag, region) %>% 
    summarize(cors = mean(cors))


# --- ASK 1 --- # 
# For DNa, DNb, and DAN only:
# Make a plot where top panel is heatmap over lags 
# Bottom is topo for each but alpha only


# --- BIVARIATE HEATMAPS --- #
# Averaged across subject and session, and the non plotted dimension
# x axis is frequency

# Get channel coordinates from Python
use_condaenv('eeg-fmri')
py_run_string("
from analysis.scripts.modules.preprocessing.eeg_utils import get_channel_coordinates
import mne
raw = mne.io.read_raw_eeglab('analysis/data/original/sub-001/ses-001/eeg/sub-001_ses-001_bld001_eeg_Bergen_CWreg_filt_ICA_rej.set')
ch_names = raw.info['ch_names']
ch_pos = get_channel_coordinates(ch_names)
")

# Prep data
pd1 <- result %>% 
    filter(region %in% c('DNa', 'DNb', 'DAN')) %>%
    group_by(subject, lag, frequency, region) %>% 
    summarize(cors = mean(cors)) %>% 
    group_by(lag, frequency, region) %>% 
    summarize(cors = mean(cors)) 

# Find range of cors
big <- ceiling(max(pd1$cors) * 100) / 100
small <- floor(min(pd1$cors) * 100) / 100

# Plot
p1 <- pd1 %>% 
    mutate(region = factor(region, levels = c('DNa', 'DNb', 'DAN'))) %>%
    ggplot(aes(x = frequency, y = lag)) +
    geom_tile(aes(fill = cors)) + 
    facet_wrap(~region) +
    scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
                         values = rescale(c(min(pd1$cors), 0, max(pd1$cors))),
                         limits = c(small, big),
                         breaks = c(small, 0, big),
                         labels = c(small, 0, big)) + 
    labs(
        x = 'Frequency (Hz)',
        y = 'Lag (s)',
        fill = latex2exp::TeX('$\\rho_{EEG, fMRI}$')
    ) + 
    scale_y_continuous(breaks = seq(0, max(result$lag), 2), labels = seq(0, max(result$lag), 2)) + 
    theme_bw() + 
    theme(strip.background = element_rect(fill = NA),
          panel.grid = element_blank(),
          axis.ticks = element_blank(),
          axis.text.y = element_text(size = 8),
          legend.position = 'bottom',
          text = element_text(size=size))

# --- PLOT TOPOS --- #

# Get frequency bands
breaks <- c(0, 1, 4, 8, 12, 30, 40)
labels <- c('init', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')
bins <- unique(cut(result$frequency, breaks=breaks))
labels <- paste(labels, bins, sep=' ')

pd <- result %>% 
    mutate(band = cut(frequency, breaks, labels)) %>% 
    filter(band != 'init (0,1]') %>% 
    inner_join(py$ch_pos) %>% 
    group_by(subject, x, y, band, region) %>% 
    summarize(cors = mean(cors), channel = unique(channel)) %>% 
    group_by(x, y, band, region) %>% 
    summarize(cors = mean(cors), channel = unique(channel)) %>% 
    mutate(z = 50, 
           region = recode(region, `dan` = 'DAN', `dmn` = 'DMN',
                           `DNa` = 'DNa', `DNb` = 'DNb'))

small <- floor(min(pd$cors)*100)/100
big <- ceiling(max(pd$cors)*100)/100

p2 <- pd %>% 
    filter(band == 'Alpha (8,12]',
           region %in% c('DNa', 'DNb', 'DAN')) %>%
    mutate(region = factor(region, levels = c('DNa', 'DNb', 'DAN'))) %>%
    ggplot(aes(x = x, y = y, z = z)) + 
    geom_topo(chan_markers = 'text', aes(fill = cors, label = channel)) +
    facet_grid(region~band) + 
    scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
                         values = rescale(c(small, 0, big)),
                         breaks = c(small, 0, big),
                         limits = c(small, big)) +
    labs(
        x = '',
        y = '',
        fill = latex2exp::TeX('$\\rho_{EEG, fMRI}$')
    ) + 
    theme_bw() + 
    theme(panel.grid = element_blank(),
          axis.text = element_blank(),
          axis.title = element_blank(),
          strip.background = element_rect(fill = NA),
          legend.position = 'bottom',
          text = element_text(size=size),
          axis.ticks = element_blank())

ggsave(plot=p1, file=path(root, '/figures/2025-08-26/temp_out.png'), 
       height = 6, width = 10, units = 'in', dpi = 300)
