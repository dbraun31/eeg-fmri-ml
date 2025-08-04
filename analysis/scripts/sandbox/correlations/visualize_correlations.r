# --- MAKE ALL CORRELATION VISUALIZATIONS --- #
# --- scripts/sandbox/correlations/correlations_long.csv needs to exist prior 
# ---- to running this script
# --- (this script is very RAM intensive)

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
setwd(path(here(), 'analysis'))
root <- path('scripts/sandbox/correlations')
size <- 16

if (file.exists(path(root, 'correlations_long.csv'))) {
    result <- fread(path(root, 'correlations_long.csv'))
    result_run <- fread(path(root, 'correlations_long_byrun.csv'))
} else {
    stop('correlations_long.csv is missing. First run make_flat_data.py, then run make_long_data.r')
}

# Adjust the lag var to (s)
result <- result %>% 
    mutate(lag = lag * 2,
           region = recode(region, `dan` = 'DAN', `dmn` = 'DMN',
                           `dmna` = 'DMNa', `dmnb` = 'DMNb', `diff` = 'DAN - DMNa')) %>% 
    filter(region != "DAN - DMNa")
result_run <- result_run %>% 
    mutate(lag = lag * 2,
           region = recode(region, `dan` = 'DAN', `dmn` = 'DMN',
                           `dmna` = 'DMNa', `dmnb` = 'DMNb', `diff` = 'DAN - DMNa')) %>% 
    filter(region != "DAN - DMNa")
    
# --- UNCONDITIONAL SUBJECT-LEVEL HISTOGRAMS --- #
# DMN only

pd <- result %>% 
    spread(region, cors) %>% 
    rename(mean_cors = DMN) %>% 
    select(-DAN) %>% 
    unite(feature, channel, frequency, lag, sep = '_') 
    
    
stat_dat <- pd %>% 
    group_by(subject) %>% 
    summarize(ci_h = quantile(mean_cors, probs = .975),
           ci_l = quantile(mean_cors, probs = .025)) 

pd %>% 
    ggplot(aes(x = mean_cors)) + 
    geom_vline(xintercept = 0, linetype = 'dashed') + 
    geom_vline(data = stat_dat, aes(xintercept = ci_h), linetype = 'dashed', color = 'orange') + 
    geom_vline(data = stat_dat, aes(xintercept = ci_l), linetype = 'dashed', color = 'orange') + 
    geom_histogram(color = 'black', fill = 'steelblue', alpha = .6) + 
    facet_wrap(~subject) + 
    xlim(-.5, .5) + 
    labs(
        x = 'Correlations between features and DMN',
        y = 'Frequency'
    ) + 
    theme_bw() + 
    theme(panel.grid = element_blank(),
          axis.ticks = element_blank(),
          strip.background = element_rect(fill = NA),
          axis.text.x = element_text(size = 12),
          text = element_text(size = size)
          )


# Screens
ggsave(path(root, 'figures/correlation_histograms.png'), width = 1920, height = 1080, units = 'px', dpi = 120)



# --- BIVARIATE HEATMAPS --- #
# Averaged across subject and session, and the non plotted dimension
# x axis is frequency

# Get channel coordinates from Python
use_condaenv('eeg-fmri')
py_run_string("
from scripts.modules.preprocessing.eeg_utils import get_channel_coordinates
import mne
raw = mne.io.read_raw_eeglab('data/original/sub-001/ses-001/eeg/sub-001_ses-001_bld001_eeg_Bergen_CWreg_filt_ICA_rej.set')
ch_names = raw.info['ch_names']
ch_pos = get_channel_coordinates(ch_names)
")
ch_names <- py$ch_names

# Prep data
pd1 <- result %>% 
    group_by(subject, lag, frequency, region) %>% 
    summarize(cors = mean(cors)) %>% 
    group_by(lag, frequency, region) %>% 
    summarize(cors = mean(cors)) 
pd2 <- result %>% 
    group_by(subject, channel, frequency, region) %>% 
    summarize(cors = mean(cors)) %>% 
    group_by(channel, frequency, region) %>% 
    summarize(cors = mean(cors)) 

# Find range of cors
big <- ceiling(max(pd1$cors, pd2$cors) * 100) / 100
small <- floor(min(pd1$cors, pd2$cors) * 100) / 100

# Plot
p1 <- pd1 %>% 
    ggplot(aes(x = frequency, y = lag)) +
    geom_tile(aes(fill = cors)) + 
    facet_wrap(~region) +
    scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
                         values = rescale(c(min(pd1$cors, pd2$cors), 0, max(pd1$cors, pd2$cors))),
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

ch_labels <- rep('', length(ch_names))
ch_labels[seq(1, length(ch_labels), 3)] <- ch_names[seq(1, length(ch_labels), 3)]
    
p2 <- pd2 %>% 
    mutate(channel = factor(channel, levels = rev(ch_names))) %>% 
    ggplot(aes(x = frequency, y = channel)) + 
    geom_tile(aes(fill = cors)) + 
    facet_wrap(~region) + 
    scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
                         values = rescale(c(min(pd1$cors, pd2$cors), 0, max(pd1$cors, pd2$cors))),
                         limits = c(small, big), 
                         breaks = c(small, 0, big),
                         labels = c(small, 0, big)) + 
    labs(
        x = 'Frequency (Hz)',
        y = 'Channel',
        fill = latex2exp::TeX('$\\rho_{EEG, fMRI}$')
    ) + 
    scale_y_discrete(labels = rev(ch_labels)) + 
    theme_bw() + 
    theme(strip.background = element_rect(fill = NA),
          panel.grid = element_blank(),
          axis.ticks = element_blank(),
          legend.position = 'none',
          text = element_text(size=size),
          axis.text.y = element_text(size = 8))


g <- ggarrange(p1, p2, nrow = 2)

# Screens
ggsave(plot = g, filename = path(root, 'figures/heat_maps.png'), 
       width = 1920, height = 1080, units = 'px', dpi = 120)



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
                           `dmna` = 'DMNa', `dmnb` = 'DMNb'))

small <- floor(min(pd$cors)*100)/100
big <- ceiling(max(pd$cors)*100)/100

pd %>% 
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

ggsave(path(root, 'figures/topo.png'), height = 1080, width = 1920, units = 'px', dpi = 120)    




# --- INDIVIDUAL DIFFERENCES --- #

# INDIVIDUAL HISTOGRAMS

# Identify strongest tiles

channel <- result %>% 
    filter(region %in% c('DAN', 'DMNa')) %>% 
    group_by(subject, channel, frequency, region) %>% 
    summarize(cors = mean(cors)) %>% 
    group_by(channel, frequency, region) %>% 
    summarize(cors = mean(cors)) %>% 
    group_by(region) %>% 
    filter(abs(cors) == max(abs(cors))) %>% 
    mutate(lag = NA)
    
lag <- result %>% 
    filter(region %in% c('DAN', 'DMNa')) %>% 
    group_by(subject, lag, frequency, region) %>% 
    summarize(cors = mean(cors)) %>% 
    group_by(lag, frequency, region) %>% 
    summarize(cors = mean(cors)) %>% 
    group_by(region) %>% 
    filter(abs(cors) == max(abs(cors))) %>% 
    mutate(channel = NA)
    
peaks <- rbind(channel, lag)

# Plot
result_run %>% 
    filter((channel=='P3' & frequency==10 & region=='DAN' & lag==2) | 
            (channel=='P3' & frequency==10 & region=='DMNa' & lag==2)) %>% 
    group_by(subject, session, run, region) %>% 
    summarize(cors = mean(cors)) %>% 
    ggplot(aes(x = cors, y = subject)) +
    geom_vline(xintercept = 0, linetype = 'dashed') + 
    geom_density_ridges(stat='binline', color = 'black', fill = 'steelblue', alpha = .6) + 
    facet_wrap(~region) + 
    labs(
        x = latex2exp::TeX('$\\rho_{EEG, fMRI}'),
        y = 'Subject',
        caption = 'DAN: freq=11, channel=P3, lag=2\nDMNa: freq=10, channel=P3, lag=2'
    ) + 
    theme_bw() + 
    theme(axis.ticks = element_blank(),
          panel.grid = element_blank(),
          text = element_text(size = size),
          strip.background = element_rect(fill = NA))

ggsave(path(root, 'figures/individual_differences.png'), height = 1080, width = 1920, units = 'px', dpi = 120)    


# --- BY TASK --- #

# Prep data
pd1 <- result_run %>% 
    mutate(task = ifelse(run == 'run-001', 'GradCPT', 'ExperienceSampling')) %>% 
    group_by(subject, lag, frequency, region, task) %>% 
    summarize(cors = mean(cors)) %>% 
    group_by(lag, frequency, region, task) %>% 
    summarize(cors = mean(cors)) 
pd2 <- result_run %>% 
    mutate(task = ifelse(run == 'run-001', 'GradCPT', 'ExperienceSampling')) %>% 
    group_by(subject, channel, frequency, region, task) %>% 
    summarize(cors = mean(cors)) %>% 
    group_by(channel, frequency, region, task) %>% 
    summarize(cors = mean(cors)) 

# Find range of cors
big <- ceiling(max(pd1$cors, pd2$cors) * 100) / 100
small <- floor(min(pd1$cors, pd2$cors) * 100) / 100

# Plot
p1 <- pd1 %>% 
    ggplot(aes(x = frequency, y = lag)) +
    geom_tile(aes(fill = cors)) + 
    facet_grid(task~region) +
    scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
                         values = rescale(c(min(pd1$cors, pd2$cors), 0, max(pd1$cors, pd2$cors))),
                         limits = c(small, big),
                         breaks = c(small, 0, big),
                         labels = c(small, 0, big)) + 
    scale_y_continuous(breaks = seq(0, max(result$lag), 2), labels = seq(0, max(result$lag), 2)) +
    labs(
        x = 'Frequency (Hz)',
        y = 'Lag(s)',
        fill = latex2exp::TeX('$\\rho_{EEG, fMRI}$')
    ) + 
    theme_bw() + 
    theme(strip.background = element_rect(fill = NA),
          panel.grid = element_blank(),
          axis.ticks = element_blank(),
          legend.position = 'bottom',
          text = element_text(size = 14),
          strip.text.y = element_text(size = 10),
          axis.text.y = element_text(size = 8))
    
p2 <- pd2 %>% 
    mutate(channel = factor(channel, levels = rev(ch_names))) %>% 
    ggplot(aes(x = frequency, y = channel)) + 
    geom_tile(aes(fill = cors)) + 
    facet_grid(task~region) + 
    scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
                         values = rescale(c(min(pd1$cors, pd2$cors), 0, max(pd1$cors, pd2$cors))),
                         limits = c(small, big), 
                         breaks = c(small, 0, big),
                         labels = c(small, 0, big)) + 
    labs(
        x = 'Frequency (Hz)',
        y = 'Channel',
        fill = latex2exp::TeX('$\\rho_{EEG, fMRI}$')
    ) + 
    scale_y_discrete(labels = rev(ch_labels)) + 
    theme_bw() + 
    theme(strip.background = element_rect(fill = NA),
          panel.grid = element_blank(),
          axis.ticks = element_blank(),
          legend.position = 'none',
          axis.text.y = element_text(size = 8),
          text = element_text(size = size),
          strip.text.y = element_text(size = 12))


g <- ggarrange(p1, p2, nrow = 2)
g


ggsave(path(root, 'figures/heatmap_by_task.png'), height = 1080, width = 1920, units = 'px', dpi = 120)    



# -- FREQ X LAG WITH SIGNIFICANCE -- #

breaks <- c(0, 1, 4, 8, 12, 30, 40)
labels <- c('init', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')
bins <- unique(cut(result$frequency, breaks=breaks))
labels <- paste(labels, bins, sep=' ')


ps <- result %>% 
    mutate(bin = cut(frequency, breaks, labels)) %>% 
    filter(bin != 'init (0,1]', lag <= 10) %>% 
    group_by(subject, lag, region, bin) %>% 
    summarize(cors = mean(cors)) %>% 
    group_by(lag, region, bin) %>% 
    summarize(p = t.test(cors, mu = 0)$p.value) %>% 
    mutate(p_adj = p.adjust(p, method='fdr')) %>% 
    filter(p_adj < .05) 
    

pd <- result %>% 
    mutate(bin = cut(frequency, breaks, labels)) %>% 
    filter(bin != 'init (0,1]', lag <= 10) %>% 
    group_by(bin, lag, region) %>% 
    summarize(cors = mean(cors)) 

small <- floor(min(pd$cors)*100)/100
big <- ceiling(max(pd$cors)*100)/100

pd %>%     
    ggplot(aes(x = bin, y = lag)) + 
    geom_tile(aes(fill = cors)) + 
    geom_point(data=ps, aes(x = bin, y = lag), shape = 8, color = 'gold', size = 5) + 
    facet_wrap(~region) + 
    scale_y_continuous(breaks = seq(0, 10, 2), labels = seq(0, 10, 2)) +
    scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
                         values = rescale(c(small, 0, big)),
                         breaks = c(small, 0, big),
                         limits = c(small, big)) + 
    labs(
        x = 'Frequency bin',
        y = 'Lag (s)',
        fill = latex2exp::TeX('$\\rho_{EEG, fMRI}$')
    ) + 
    theme_bw() + 
    theme(strip.background = element_rect(fill = NA),
          axis.ticks = element_blank(),
          panel.grid = element_blank(),
          legend.position = 'bottom',
          text = element_text(size = size))
    
ggsave(path(root, 'figures/heatmap_with_significance.png'), height = 1080, width = 1920, units = 'px', dpi = 120)    

                         






# --- BY TASK WITH DIFFERENCE REGION --- #

result_run <- fread(path(root, 'correlations_long_byrun.csv'))

result_run <- result_run %>% 
    mutate(lag = lag * 2,
           region = recode(region, `dan` = 'DAN', `dmn` = 'DMN',
                           `dmna` = 'DMNa', `dmnb` = 'DMNb', `diff` = 'DAN - DMNa')) 

# Prep data
pd1 <- result_run %>% 
    mutate(task = ifelse(run == 'run-001', 'GradCPT', 'ExperienceSampling')) %>% 
    group_by(subject, lag, frequency, region, task) %>% 
    summarize(cors = mean(cors)) %>% 
    group_by(lag, frequency, region, task) %>% 
    summarize(cors = mean(cors)) 
pd2 <- result_run %>% 
    mutate(task = ifelse(run == 'run-001', 'GradCPT', 'ExperienceSampling')) %>% 
    group_by(subject, channel, frequency, region, task) %>% 
    summarize(cors = mean(cors)) %>% 
    group_by(channel, frequency, region, task) %>% 
    summarize(cors = mean(cors)) 

# Find range of cors
big <- ceiling(max(pd1$cors, pd2$cors) * 100) / 100
small <- floor(min(pd1$cors, pd2$cors) * 100) / 100

# Plot
p1 <- pd1 %>% 
    ggplot(aes(x = frequency, y = lag)) +
    geom_tile(aes(fill = cors)) + 
    facet_grid(task~region) +
    scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
                         values = rescale(c(min(pd1$cors, pd2$cors), 0, max(pd1$cors, pd2$cors))),
                         limits = c(small, big),
                         breaks = c(small, 0, big),
                         labels = c(small, 0, big)) + 
    scale_y_continuous(breaks = seq(0, max(result$lag), 2), labels = seq(0, max(result$lag), 2)) +
    labs(
        x = 'Frequency (Hz)',
        y = 'Lag(s)',
        fill = latex2exp::TeX('$\\rho_{EEG, fMRI}$')
    ) + 
    theme_bw() + 
    theme(strip.background = element_rect(fill = NA),
          panel.grid = element_blank(),
          axis.ticks = element_blank(),
          legend.position = 'bottom',
          text = element_text(size = 14),
          strip.text.y = element_text(size = 10),
          axis.text.y = element_text(size = 8))
    
p2 <- pd2 %>% 
    mutate(channel = factor(channel, levels = rev(ch_names))) %>% 
    ggplot(aes(x = frequency, y = channel)) + 
    geom_tile(aes(fill = cors)) + 
    facet_grid(task~region) + 
    scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
                         values = rescale(c(min(pd1$cors, pd2$cors), 0, max(pd1$cors, pd2$cors))),
                         limits = c(small, big), 
                         breaks = c(small, 0, big),
                         labels = c(small, 0, big)) + 
    labs(
        x = 'Frequency (Hz)',
        y = 'Channel',
        fill = latex2exp::TeX('$\\rho_{EEG, fMRI}$')
    ) + 
    scale_y_discrete(labels = rev(ch_labels)) + 
    theme_bw() + 
    theme(strip.background = element_rect(fill = NA),
          panel.grid = element_blank(),
          axis.ticks = element_blank(),
          legend.position = 'none',
          axis.text.y = element_text(size = 8),
          text = element_text(size = size),
          strip.text.y = element_text(size = 12))


g <- ggarrange(p1, p2, nrow = 2)
g


ggsave(path(root, 'figures/heatmap_by_task_include_difference.png'), height = 1080, width = 1920, units = 'px', dpi = 120)    















