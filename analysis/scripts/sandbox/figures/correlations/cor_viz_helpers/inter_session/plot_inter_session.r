plot_inter_session <- function(d_run, networks, bands, label_middle=TRUE, 
                               return_icc=TRUE, n_lags=10, scales=NA,
                               overall_text=18, axis_text=16) {
    
    #' Plot Inter-Session Reliability and Mean Correlations
    #'
    #' Computes and visualizes inter-session reliability (Intraclass Correlation Coefficient, ICC)
    #' of EEG–fMRI correlations across frequency and lag, optionally alongside session-specific
    #' mean correlation heatmaps.
    #'
    #' @param d_run A data frame containing columns `subject`, `session`, `frequency`, `lag`, `network`, and `cors`.
    #' @param networks A character vector of network names to include in the analysis.
    #' @param bands (Unused in current implementation; reserved for future frequency filtering.)
    #' @param label_middle Logical; if `TRUE`, centers the correlation color scale at zero. Default is `TRUE`.
    #' @param return_icc Logical; if `TRUE`, returns a combined plot of both sessions and ICC heatmap. Default is `TRUE`.
    #' @param n_lags Numeric; maximum lag value to include. Default is 10.
    #' @param scales Optional numeric vector `c(min, max)` to fix the color scale limits. Defaults to automatic scaling.
    #' @param overall_text Numeric, base font size for all plot text. Default is 18.
    #' @param axis_text Numeric, font size for axis labels. Default is 16.
    #'
    #' @return A `ggplot` or `ggarrange` object showing:
    #' - Per-session mean EEG–fMRI correlations (Session 1 & 2)
    #' - Inter-session reliability as ICC heatmap
    #'
    #' @details
    #' - ICC(3,1) is computed between sessions for each frequency–lag pair within each network.
    #' - Excludes subject `sub-023` from ICC calculations.
    #' - If `return_icc = TRUE`, returns a side-by-side plot of session correlations and ICC map.
    #' - Uses a green sequential palette for ICC values and a red–blue diverging palette for correlations.
    #'
    #' @examples
    #' plot_inter_session(df, networks = c("DMN", "DAN"), n_lags = 8)
    
    
    

	# --- GENERATE CORRELATION HEATMAPS --- #
	
	# Summarize across conditions and compute data to plot
	pd1 <- d_run %>%
		filter(network %in% !!networks, lag <= n_lags) %>%
		group_by(subject, session, frequency, lag, network) %>%
		summarize(cors = mean(cors)) %>%
		group_by(session, frequency, lag, network) %>%
		summarize(cors = mean(cors)) 

	# Whether to use user-specified scales
	if (!is.na(scales)) {
	    small <- floor(scales[1] * 100) / 100
	    big <- ceiling(scales[2] * 100) / 100
	} else {
	    small <- floor(min(pd1$cors) * 100) / 100
	    big <- ceiling(max(pd1$cors) * 100) / 100
	}
	
	# Whether to label the middle of the scale
	if (label_middle) {
	    sfg <- scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
	                                values = rescale(c(small, 0, big)),
	                                breaks = c(small, 0, big),
	                                limits = c(small, big))  
	} else {
	    sfg <- scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
	                                values = rescale(c(small, 0, big)),
	                                breaks = c(small, big),
	                                limits = c(small, big))  
	}
	
	# Generate correlation heat map plot
	p1 <- pd1 %>%
		mutate(session = recode(session, `ses-001` = 'Session 1', `ses-002` = 'Session 2')) %>% 
		ggplot(aes(x = frequency, y = lag)) +
		geom_raster(aes(fill = cors), interpolate=TRUE) + 
		labs(
			 x = 'Frequency (Hz)',
			 y = 'Lag (s)',
			 fill = latex2exp::TeX('$\\rho_{~~EEG,fMRI}$')) + 
	    sfg + 
		facet_grid(network~session) + 
		theme_bw() + 
		theme(legend.position = 'bottom',
			  axis.ticks = element_blank(),
			  panel.grid = element_blank(),
			  legend.text = element_text(angle = 55, hjust=1),
			  strip.background = element_rect(fill = NA),
			  text = element_text(size = overall_text),
			  axis.text = element_text(size = axis_text))
	
	if (maybe_log_scale(pd1$frequency)) p1 <- p1 + scale_x_log10()
	
	# Either return correlations and ICC or just correlations
	
	if (!return_icc) return(p1)
	
	# --- GENERATE ICC PLOT --- #
	
    # Function for computing ICC between two vectors
	get_icc <- function(s1, s2) {
		d <- data.frame(s1, s2)
		i <- ICC(d)$results
		out <- i[i$type=='ICC3',]$ICC
		return(out)
	}

	# Summarize across conditions and compute ICC
	pd <- d_run %>%
		filter(network %in% !!networks, lag <= n_lags) %>%
		group_by(subject, session, frequency, lag, network) %>%
		summarize(cors = mean(cors, na.rm=TRUE)) %>%
		mutate(session = str_replace(session, '-', '')) %>%
		spread(session, cors) %>%
		group_by(frequency, lag, network) %>%
		summarize(icc = get_icc(ses001, ses002)) 

	pal <- paletteer::paletteer_c('ggthemes::Green', n = 100)

	# Generate ICC plot
	p2 <- pd %>%
		ggplot(aes(x = frequency, y = lag)) +
		geom_raster(aes(fill = icc), interpolate=TRUE) + 
		labs(
			 x = 'Frequency (Hz)',
			 y = 'Lag (s)',
			 main = 'Intraclass Correlation Coefficient',
			 fill = 'ICC') + 
		scale_fill_gradientn(colors = pal) + 
	    facet_wrap(~network) + 
		theme_bw() + 
		theme(legend.position = 'bottom',
			  axis.ticks = element_blank(),
			  panel.grid = element_blank(),
			  legend.title = element_text(margin = margin(r = 10)),
			  text = element_text(size = overall_text),
			  axis.text = element_text(size = axis_text),
			  strip.background = element_rect(fill = NA, color = 'black'))
	
	if (maybe_log_scale(pd$frequency)) p2 <- p2 + scale_x_log10()

	print(glue::glue('Mean ICC: {round(mean(pd$icc), 2)}, SD = {round(sd(pd$icc), 2)}'))
	
    if (return_icc) {
        # Putting two first since it's the correlations
        g <- ggarrange(p2, p1, nrow = 1)
        return(g)
    }	
		
}
