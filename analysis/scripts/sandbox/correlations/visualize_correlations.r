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

if (file.exists(path(root, 'correlations_long.csv'))) {
    result <- read.csv(path(root, 'correlations_long.csv'))
    result_run <- read.csv(path(root, 'correlations_long_byrun.csv'))
} else {

    # Get channel names
    use_condaenv('eeg-fmri')
py_run_string("
import mne
raw = mne.io.read_raw_eeglab('data/original/sub-001/ses-001/eeg/sub-001_ses-001_bld001_eeg_Bergen_CWreg_filt_ICA_rej.set')
ch_names = raw.info['ch_names']
")
    ch_names <- py$ch_names
    # Make order more logical
    ch_names <- c(ch_names[grepl('^F', ch_names)], ch_names[grepl('^T', ch_names)],
                  ch_names[grepl('^C', ch_names)], ch_names[grepl('^P', ch_names)],
                  ch_names[grepl('^O', ch_names)])
    
    d <- data.table(read_feather('data/merged_data.feather'))
    
    voltage_cols <- colnames(d)[9:length(colnames(d))]
    
    # Big data table energy (expensive)
    result_run <- d[,
      .(dmn_cors = list(sapply(.SD, function(col) cor(dmn, col, use = 'pairwise.complete.obs', method='spearman'))),
        dan_cors = list(sapply(.SD, function(col) cor(dan, col, use = 'pairwise.complete.obs', method='spearman'))),
        dmna_cors = list(sapply(.SD, function(col) cor(dmn_a, col, use = 'pairwise.complete.obs', method='spearman'))),
        dmnb_cors = list(sapply(.SD, function(col) cor(dmn_b, col, use = 'pairwise.complete.obs', method='spearman')))),
      by = .(subject, session, run),
      .SDcols = voltage_cols
      ]
    result <- result_run[
          ,
          .(mean_dmn = list(Reduce(`+`, dmn_cors) / length(dmn_cors)),
            mean_dan = list(Reduce(`+`, dan_cors) / length(dan_cors)),
            mean_dmna = list(Reduce(`+`, dmna_cors) / length(dmna_cors)),
            mean_dmnb = list(Reduce(`+`, dmnb_cors) / length(dmnb_cors))),
          by = subject
      ][
          ,
          .(feature = voltage_cols, dmn_cors = unlist(mean_dmn),
            dan_cors = unlist(mean_dan),
            dmna_cors = unlist(mean_dmna),
            dmnb_cors = unlist(mean_dmnb)),
          by = subject
      ]
    
    result_run <- result_run[
        ,
        .(feature = voltage_cols,
          dan_cors = unlist(dan_cors),
          dmn_cors = unlist(dmn_cors),
          dmna_cors = unlist(dmna_cors),
          dmnb_cors = unlist(dmnb_cors)),
        by = .(subject, session, run)
    ]
    
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
}
    
# --- UNCONDITIONAL SUBJECT-LEVEL HISTOGRAMS --- #
# DMN only

pd <- result %>% 
    spread(region, cors) %>% 
    rename(mean_cors = dmn) %>% 
    select(-dan) %>% 
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
          text = element_text(size = 16),
          axis.text.x = element_text(size = 12)
          )


# Screens
ggsave(path(root, 'figures/correlation_histograms.png'), width = 1920, height = 1080, units = 'px', dpi = 120)



# --- BIVARIATE HEATMAPS --- #
# Averaged across subject and session, and the non plotted dimension
# x axis is frequency

# Get channel coordinates
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
    mutate(region = recode(region, `dmn` = 'DMN', `dan` = 'DAN', `dmna` = 'DMNa', `dmnb` = 'DMNb')) %>% 
    ggplot(aes(x = frequency, y = lag)) +
    geom_tile(aes(fill = cors)) + 
    facet_wrap(~region) +
    scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
                         values = rescale(c(min(pd1$cors, pd2$cors), 0, max(pd1$cors, pd2$cors))),
                         limits = c(small, big),
                         breaks = c(small, 0, big),
                         labels = c(small, 0, big)) + 
    scale_y_continuous(breaks = seq(8, 0, -1), labels = seq(8, 0, -1)) +
    labs(
        x = 'Frequency (Hz)',
        y = 'Lag(s)',
        fill = latex2exp::TeX('$\\rho_{EEG, fMRI}$')
    ) + 
    theme_bw() + 
    theme(strip.background = element_rect(fill = NA),
          panel.grid = element_blank(),
          axis.ticks = element_blank(),
          legend.position = 'bottom')
    
p2 <- pd2 %>% 
    mutate(channel = factor(channel, levels = rev(ch_names)),
           region = recode(region, `dmn` = 'DMN', `dan` = 'DAN', `dmna` = 'DMNa', `dmnb` = 'DMNb')) %>% 
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
    theme_bw() + 
    theme(strip.background = element_rect(fill = NA),
          panel.grid = element_blank(),
          axis.ticks = element_blank(),
          legend.position = 'none')


g <- ggarrange(p1, p2, nrow = 2)
g

# Screens
ggsave(plot = g, filename = path(root, 'figures/heat_maps.png'), 
       width = 1920, height = 1080, units = 'px', dpi = 120)



# --- PLOT TOPOS --- #

# Get frequency bands
breaks <- c(0, 4, 8, 12, 30, 100)
labels <- c('Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')
bins <- unique(cut(result$frequency, breaks=breaks))
labels <- paste(labels, bins, sep=' ')

pd <- result %>% 
    mutate(band = cut(frequency, breaks, labels)) %>% 
    inner_join(py$ch_pos) %>% 
    group_by(subject, x, y, band, region) %>% 
    summarize(cors = mean(cors), channel = unique(channel)) %>% 
    group_by(x, y, band, region) %>% 
    summarize(cors = mean(cors), channel = unique(channel)) %>% 
    mutate(z = 50, 
           region = recode(region, `dan` = 'DAN', `dmn` = 'DMN',
                           `dmna` = 'DMNa', `dmnb` = 'DMNb'))
pd %>% 
    ggplot(aes(x = x, y = y, z = z)) + 
    geom_topo(chan_markers = 'text', aes(fill = cors, label = channel)) +
    facet_grid(region~band) + 
    scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
                         values = rescale(c(min(pd$cors), 0, max(pd$cors))),
                         limits = c(min(pd$cors), max(pd$cors))) +
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
          axis.ticks = element_blank())

ggsave(path(root, 'figures/topo.png'), height = 1080, width = 1920, units = 'px', dpi = 120)    




# --- INDIVIDUAL DIFFERENCES --- #

# INDIVIDUAL HISTOGRAMS

# Identify strongest tiles

channel <- result %>% 
    filter(region %in% c('dan', 'dmna')) %>% 
    group_by(subject, channel, frequency, region) %>% 
    summarize(cors = mean(cors)) %>% 
    group_by(channel, frequency, region) %>% 
    summarize(cors = mean(cors)) %>% 
    group_by(region) %>% 
    filter(abs(cors) == max(abs(cors))) %>% 
    mutate(lag = NA)
    
lag <- result %>% 
    filter(region %in% c('dan', 'dmna')) %>% 
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
    filter((channel=='O2' & frequency==11 & region=='dan' & lag==1) | 
            (channel=='P3' & frequency==10 & region=='dmna' & lag==1)) %>% 
    group_by(subject, session, run, region) %>% 
    summarize(cors = mean(cors)) %>% 
    mutate(region = recode(region, `dan` = 'DAN', `dmna` = 'DMNa')) %>% 
    ggplot(aes(x = cors, y = subject)) +
    geom_vline(xintercept = 0, linetype = 'dashed') + 
    geom_density_ridges(stat='binline', color = 'black', fill = 'steelblue', alpha = .6) + 
    facet_wrap(~region) + 
    labs(
        x = latex2exp::TeX('$\\rho_{EEG, fMRI}'),
        y = 'Subject',
        caption = 'DAN: freq=11, channel=O2, lag=1\nDMNa: freq=10, channel=P3, lag=1'
    ) + 
    theme_bw() + 
    theme(axis.ticks = element_blank(),
          panel.grid = element_blank(),
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
    mutate(region = recode(region, `dmn` = 'DMN', `dan` = 'DAN', `dmna` = 'DMNa', `dmnb` = 'DMNb')) %>% 
    ggplot(aes(x = frequency, y = lag)) +
    geom_tile(aes(fill = cors)) + 
    facet_grid(task~region) +
    scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
                         values = rescale(c(min(pd1$cors, pd2$cors), 0, max(pd1$cors, pd2$cors))),
                         limits = c(small, big),
                         breaks = c(small, 0, big),
                         labels = c(small, 0, big)) + 
    scale_y_continuous(breaks = seq(8, 0, -1), labels = seq(8, 0, -1)) +
    labs(
        x = 'Frequency (Hz)',
        y = 'Lag(s)',
        fill = latex2exp::TeX('$\\rho_{EEG, fMRI}$')
    ) + 
    theme_bw() + 
    theme(strip.background = element_rect(fill = NA),
          panel.grid = element_blank(),
          axis.ticks = element_blank(),
          legend.position = 'bottom')
    
p2 <- pd2 %>% 
    mutate(channel = factor(channel, levels = rev(ch_names)),
           region = recode(region, `dmn` = 'DMN', `dan` = 'DAN', `dmna` = 'DMNa', `dmnb` = 'DMNb')) %>% 
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
    theme_bw() + 
    theme(strip.background = element_rect(fill = NA),
          panel.grid = element_blank(),
          axis.ticks = element_blank(),
          legend.position = 'none')


g <- ggarrange(p1, p2, nrow = 2)
g


ggsave(path(root, 'figures/heatmap_by_task.png'), height = 1080, width = 1920, units = 'px', dpi = 120)    

























