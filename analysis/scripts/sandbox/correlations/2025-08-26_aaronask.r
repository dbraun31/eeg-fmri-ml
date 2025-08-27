# --- DATA AND LIBRARIES --- #
rm(list=ls())
library(arrow)
library(tidyverse)
library(psych)
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


# --- PLOT HEAT AND TOPOS --- #

# Get frequency bands
breaks <- c(0, 1, 4, 8, 12, 30, 40)
labels <- c('init', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')
bins <- unique(cut(result$frequency, breaks=breaks))
labels <- paste(labels, bins, sep=' ')

pd2 <- result %>% 
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

small <- floor(min(pd1$cors, pd2$cors)*100)/100
big <- ceiling(max(pd1$cors, pd2$cors)*100)/100

# Plot
p1 <- pd1 %>% 
    mutate(region = factor(region, levels = c('DNa', 'DNb', 'DAN'))) %>%
    ggplot(aes(x = frequency, y = lag)) +
    geom_tile(aes(fill = cors)) + 
    facet_wrap(~region) +
    scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
                         values = rescale(c(min(pd1$cors, pd2$cors), 0, 
                                            max(pd1$cors, pd2$cors))),
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


p2 <- pd2 %>% 
    filter(band == 'Alpha (8,12]',
           region %in% c('DNa', 'DNb', 'DAN')) %>%
    mutate(region = factor(region, levels = c('DNa', 'DNb', 'DAN'))) %>%
    ggplot(aes(x = x, y = y, z = z)) + 
    geom_topo(chan_markers = 'text', aes(fill = cors, label = channel)) +
    facet_wrap(~region) + 
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
          legend.position = 'none',
          text = element_text(size=size),
          axis.ticks = element_blank())

g <- ggarrange(p1, p2, nrow=2)

ggsave(plot=g, file=path(root, '/figures/2025-08-26/heatmap_topo.png'), 
       height = 6, width = 10, units = 'in', dpi = 300)



# --- ASK 2 --- #
# Compute ICC for just DNa across sessions for subjects (lag x freq)
text <- 16
get_icc <- function(s1, s2) {
    d <- data.frame(s1, s2)
    i <- ICC(d)$results
    out <- i[i$type=='ICC3',]$ICC
    return(out)
}

pd <- result_run %>%
    filter(region == 'DNa', subject != 'sub-023', lag <= 10) %>%
    group_by(subject, session, frequency, lag) %>%
    summarize(cors = mean(cors, na.rm=TRUE)) %>%
    mutate(session = str_replace(session, '-', '')) %>%
    spread(session, cors) %>%
    group_by(frequency, lag) %>%
    summarize(icc = get_icc(ses001, ses002)) 

pal <- paletteer::paletteer_c('ggthemes::Green', n = 100)

p1 <- pd %>%
    ggplot(aes(x = frequency, y = lag)) +
    geom_tile(aes(fill = icc)) + 
    labs(
         x = 'Frequency (Hz)',
         y = 'Lag (s)',
         main = 'Intraclass Correlation Coefficient',
         fill = 'ICC') + 
    scale_fill_gradientn(colors = pal) + 
    theme_bw() + 
    theme(legend.position = 'bottom',
          axis.ticks = element_blank(),
          panel.grid = element_blank(),
          legend.title = element_text(margin = margin(r = 10)),
          text = element_text(size = text))

print(glue::glue('Mean ICC: {round(mean(pd$icc), 2)}, SD = {round(sd(pd$icc), 2)}'))

pd2 <- result_run %>%
    filter(region == 'DNa', lag <= 10) %>%
    group_by(subject, session, frequency, lag) %>%
    summarize(cors = mean(cors)) %>%
    group_by(session, frequency, lag) %>%
    summarize(cors = mean(cors)) 

small <- floor(min(pd2$cors) * 100) / 100
big <- ceiling(max(pd2$cors) * 100) / 100


p2 <- pd2 %>%
    mutate(session = recode(session, `ses-001` = 'Session 1', `ses-002` = 'Session 2')) %>% 
    ggplot(aes(x = frequency, y = lag)) +
    geom_tile(aes(fill = cors)) + 
    labs(
         x = 'Frequency (Hz)',
         y = 'Lag (s)',
         main = 'Mean Correlations',
         fill = latex2exp::TeX('$\\rho_{~~EEG,fMRI}$')) + 
    scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
                         values = rescale(c(min(pd2$cors), 0, 
                                            max(pd2$cors))),
                         limits = c(small, big),
                         breaks = c(small, 0, big),
                         labels = c(small, 0, big)) + 
    facet_wrap(~session) + 
    theme_bw() + 
    theme(legend.position = 'bottom',
          axis.ticks = element_blank(),
          panel.grid = element_blank(),
          legend.text = element_text(angle = 45, hjust=1),
          strip.background = element_rect(fill = NA),
          text = element_text(size = text))
    

g <- ggarrange(p2, p1, nrow = 1)

ggsave(plot=g, file=path(root, '/figures/2025-08-26/icc_spearman.png'), 
       height = 6, width = 15, units = 'in', dpi = 300)
