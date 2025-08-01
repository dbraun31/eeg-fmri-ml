rm(list=ls())
library(tidyverse)
library(here)
library(fs)
library(arrow)
library(scales)
library(RColorBrewer)
setwd(path(here(), 'analysis'))
root <- path('scripts/sandbox/timeseries')

# Import
d <- read_feather('data/merged_data.feather')
d <- d %>% 
    filter(run != 'run-001') %>% 
    select(subject:dmn_b)



d %>% 
    filter(subject == 'sub-024', session == 'ses-001', run == 'run-002',
           tr < 50) %>% 
    gather(region, bold, dmn:dmn_b) %>% 
    mutate(contrast = ifelse(region %in% c('dmn', 'dan'), 'DMN-DAN', 'DMNa-DMNb'),
           region = recode(region, `dan` = 'DAN', `dmn` = 'DMN', `dmn_a` = 'DMNa', `dmn_b` = 'DMNb'),
           time = tr * 2) %>% 
    ggplot(aes(x = time, y = bold)) + 
    geom_line(aes(color = region)) + 
    geom_point(size = 1.5, aes(color = region)) + 
    facet_wrap(~contrast, nrow = 2) + 
    scale_color_manual(values = brewer.pal(4, 'Dark2')) + 
    labs(
        x = 'Time (s)',
        y = 'BOLD signal',
        color = ''
    ) + 
    theme_bw() + 
    theme(axis.ticks = element_blank(),
          strip.background = element_rect(fill = NA),
          legend.position = c(.9, .4),
          text = element_text(size = 16),
          legend.title = element_blank())
    

ggsave(path(root, 'timeseries.png'), width = 1920, height = 1080, units = 'px', dpi = 120)



