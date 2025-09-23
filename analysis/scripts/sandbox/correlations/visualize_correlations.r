# --- MAKE ALL CORRELATION VISUALIZATIONS --- #
# --- scripts/sandbox/correlations/correlations_long.csv needs to exist prior 
# ---- to running this script
# --- (this script is very RAM intensive)


# --- LIBRARIES --- #
rm(list=ls())
library(arrow)
library(ggridges)
library(tidyverse)
library(eegUtils)
library(glue)
library(data.table)
library(ggpubr)
library(here)
library(fs)
library(psych)
library(scales)
library(RColorBrewer)
library(reticulate)
setwd(path(here()))
root <- path('analysis/scripts/sandbox/correlations')
source(path(root, 'figures/heat_maps/plot_heat_maps.r'))
source(path(root, 'figures/topos/plot_topos.r'))
source(path(root, 'figures/inter_session/plot_inter_session.r'))
source(path(root, 'figures/significance/plot_significance.r'))
text <- 16
size <- 16

if (file.exists(path(root, 'correlations_long.feather'))) {
    result <- read_feather(path(root, 'correlations_long.feather'))
} else {
    stop('correlations_long.csv is missing. First run make_flat_data.py, then run make_long_data.r')
}

# --- PRELIMS ---#

# Get channel coordinates from Python
use_condaenv('eeg-fmri')
py_run_string("
from analysis.scripts.modules.preprocessing.eeg_utils import get_channel_coordinates
import mne
raw = mne.io.read_raw_eeglab('analysis/data/original/sub-001/ses-001/eeg/sub-001_ses-001_bld001_eeg_Bergen_CWreg_filt_ICA_rej.set')
ch_names = raw.info['ch_names']
ch_pos = get_channel_coordinates(ch_names)
")
ch_pos <- py$ch_pos
ch_names <- py$ch_names

# Adjust the lag var to (s)
result_run <- result %>% 
    mutate(lag = lag * 2)

result <- result_run %>% 
    group_by(subject, channel, frequency, lag, network) %>% 
    summarize(cors = mean(cors))



# ========================================= # 


# --- GENERATE PLOTS --- #

# - DNa through DAN heat over topo - #

p1 <- plot_heat(c('DNa', 'DNb', 'DAN'))
p2 <- plot_dna_through_dan_topo()

g <- ggarrange(p1, p2, nrow=2)

ggsave(plot=g, file=path(root, 'figures/heat_maps/dna_through_dan.png'), 
       height = 6, width = 10, units = 'in', dpi = 300)

# - FPN A and B - #

# Heat maps across full lags
p1 <- plot_heat(c('FPNa', 'FPNb'))
ggsave(plot = p1, file=path(root, 'figures/heat_maps/FPNa_FPNb_heat.png'),
       height = 6, width = 10, units = 'in', dpi = 300)

# Heat maps with significance to 10 s lags
p <- plot_significance(c('FPNa', 'FPNb'), label_middle=FALSE)
    
ggsave(plot=p, file=path(root, 'figures/significance/FPN_significance.png'),
       height = 6, width = 10, units = 'in', dpi = 300)


# - Inter session plot - #

text <- 30
p1 <- plot_inter_session('DNa', label_middle = FALSE)
p2 <- plot_inter_session('SAL', label_middle = FALSE)
# Mean ICC for SAL = 0.53 (0.14)

g <- ggarrange(p1, p2, nrow = 2, labels = c('A.', 'B.'),
               font.label = list(size = 20))

ggsave(plot=g, file=path(root, 'figures/inter_session/inter_session.png'),
       height = 12, width = 15, units = 'in', dpi = 300)


































