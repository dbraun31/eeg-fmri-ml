
plot_by_task <- function(d_task, networks, bands, axis_text=14, overall_text=18,
                         scales=NA) {
    
    pd <- d_task %>% 
        group_by(subject, network, frequency, lag, task) %>% 
        summarize(cors = mean(cors)) %>% 
        group_by(network, frequency, lag, task) %>% 
        summarize(cors = mean(cors)) 
    
    # Set scale constraints
    if (!is.na(scales)) {
        small <- floor(scales[1] * 100) / 100
        big <- ceiling(scales[2] * 100) / 100
    } else {
        small <- floor(min(pd$cors)*100)/100
        big <- ceiling(max(pd$cors)*100)/100
    }
    
    p <- pd %>% 
        ggplot(aes(x = frequency, y = lag)) + 
        geom_raster(aes(fill = cors), interpolate = TRUE) + 
        facet_grid(task ~ network) + 
        labs(
            x = 'Frequency (Hz)',
            y = 'Lag (s)',
            fill = latex2exp::TeX('$\\rho_{~~EEG,fMRI}$')
        ) + 
        scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
                             values = rescale(c(small, 0, big)),
                             limits = c(small, big),
                             breaks = c(small, 0, big),
                             labels = c(small, 0, big)) + 
        theme_bw() + 
        theme(panel.grid = element_blank(),
              axis.ticks = element_blank(),
              strip.background = element_rect(fill = NA, color = 'black'),
              text = element_text(size = overall_text),
              axis.text = element_text(size = axis_text),
              legend.position = 'bottom')
    
    if(maybe_log_scale(d$frequency)) p <- p + scale_x_log10()
    
    return(p)
    
}
