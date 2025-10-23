maybe_log_scale <- function(x, tol = 1e-6) {
    # Check whether a vector x is log scaled
    x <- unique(x)
    n <- length(x)
    ideal <- 10^seq(log10(min(x)), log10(max(x)), length.out = n)
    return(all(abs(sort(x) - ideal) < tol))
}

plot_heat <- function(d, networks, scales = NA,
                      overall_text=18, axis_text=16) {
    #' Plot Heatmap of Mean Correlations by Frequency and Lag
    #'
    #' Generates a faceted heatmap of mean correlation values (`cors`) across
    #' subjects for selected networks, as a function of frequency and lag.
    #'
    #' @param d A data frame containing columns `subject`, `lag`, `frequency`, `network`, and `cors`.
    #' @param networks A character vector of network names to include in the plot.
    #' @param scales Optional numeric vector `c(min, max)` to set heatmap fill limits. Defaults to automatic scaling.
    #' @param overall_text Numeric, base font size for overall plot text. Default is 18.
    #' @param axis_text Numeric, font size for axis labels. Default is 16.
    #'
    #' @return A `ggplot` object representing the faceted heatmap.
    #'
    #' @details
    #' - Frequencies are binned and labeled according to standard EEG bands:
    #'   Delta, Theta, Alpha, Beta, Gamma.
    #' - Mean correlations are computed first per subject, then averaged across subjects.
    #' - Heatmap fill uses a diverging red-blue color scale centered at 0.
    #' - Optionally applies a log scale to the x-axis if `maybe_log_scale()` returns TRUE.
    #'
    #' @examples
    #' plot_heat(df, networks = c("DMN", "FPN"), scales = c(-0.2, 0.3))
    
	# Get frequency bands
	breaks <- c(0, 1, 4, 8, 12, 30, 40)
	labels <- c('init', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')
	bins <- levels(cut(d$frequency, breaks=breaks))
	labels <- paste(labels, bins, sep=' ')

	# Prep data
	pd <- d %>% 
		filter(network %in% !!networks) %>%
		group_by(subject, lag, frequency, network) %>% 
		summarize(cors = mean(cors)) %>% 
		group_by(lag, frequency, network) %>% 
		summarize(cors = mean(cors)) 

	# Set scale constraints
	if (!is.na(scales)) {
	    small <- floor(scales[1] * 100) / 100
	    big <- ceiling(scales[2] * 100) / 100
	} else {
    	small <- floor(min(pd$cors)*100)/100
    	big <- ceiling(max(pd$cors)*100)/100
	}

	# Plot
	p1 <- pd %>% 
		mutate(network = factor(network, levels = !!networks)) %>%
		ggplot(aes(x = frequency, y = lag)) +
	    geom_raster(aes(fill = cors), interpolate = TRUE) + 
		facet_wrap(~network) +
		scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
							 values = rescale(c(small, 0, big)),
							 limits = c(small, big),
							 breaks = c(small, 0, big),
							 labels = c(small, 0, big)) + 
		labs(
			x = 'Frequency (Hz)',
			y = 'Lag (s)',
			fill = latex2exp::TeX('$\\rho_{~~EEG, fMRI}$')
		) + 
		scale_y_continuous(breaks = seq(0, max(d$lag), 2), labels = seq(0, max(d$lag), 2)) + 
		theme_bw() + 
		theme(strip.background = element_rect(fill = NA),
			  panel.grid = element_blank(),
			  axis.ticks = element_blank(),
			  axis.text = element_text(size = axis_text),
			  legend.position = 'bottom',
			  text = element_text(size=overall_text))
	
	if (maybe_log_scale(pd$frequency)) p1 <- p1 + scale_x_log10()

	return(p1)

}
