library(glue)

plot_topo <- function(d, networks, bands, scales = NA, overall_text=18) {
    #' Plot Topographic Map of Mean Correlations
    #'
    #' Generates faceted EEG topographic maps of mean correlation values (`cors`) 
    #' for selected frequency bands and networks, using channel coordinates.
    #'
    #' @param d A data frame containing columns `subject`, `frequency`, `network`, and `cors`.
    #' @param networks A character vector of network names to include in the plot.
    #' @param bands A character vector of frequency bands to plot (e.g., "delta", "theta", "alpha", "beta", "gamma").
    #' @param scales Optional numeric vector `c(min, max)` to set fill limits for the color scale. Defaults to automatic scaling.
    #' @param overall_text Numeric, base font size for plot text. Default is 18.
    #'
    #' @return A `ggplot` object representing the faceted topographic map.
    #'
    #' @details
    #' - Frequencies are binned according to standard EEG bands: Delta, Theta, Alpha, Beta, Gamma.
    #' - Mean correlations are computed per subject, then averaged across subjects.
    #' - Uses a diverging red-blue color scale centered at zero.
    #' - Requires `geom_topo()` for plotting topographic maps using EEG channel coordinates.
    #'
    #' @examples
    #' plot_topo(df, networks = c("DMN", "DAN"), bands = c("alpha", "beta"))

	# Get frequency bands
	breaks <- c(0, 1, 4, 8, 12, 30, 40)
	labels <- c('init', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')
	labels_simple <- tolower(labels[2:(length(labels))])
	bins <- levels(cut(d$frequency, breaks=breaks))
	labels <- paste(labels, bins, sep=' ')
	
	bands <- tolower(bands)
	if (!all(bands %in% labels_simple)) {
	    stop(glue('Labels must be one of {paste(labels_simple, collapse = ", ")}'))
	}
	
	bands_grab <- labels[2:(length(labels))][which(labels_simple %in% bands)]
	

	pd <- d %>% 
		mutate(band = cut(frequency, breaks, labels)) %>% 
		filter(band != 'init (0,1]') %>% 
		inner_join(py$ch_pos) %>% 
		group_by(subject, x, y, band, network) %>% 
		summarize(cors = mean(cors), channel = unique(channel)) %>% 
		group_by(x, y, band, network) %>% 
		summarize(cors = mean(cors), channel = unique(channel)) %>% 
		mutate(z = 50, 
			   network = recode(network, `dan` = 'DAN', `dmn` = 'DMN',
							   `DNa` = 'DNa', `DNb` = 'DNb'))

	if (!is.na(scales)) {
	    small <- floor(scales[1] * 100) / 100
	    big <- ceiling(scales[2] * 100) / 100
	} else {
    	small <- floor(min(pd$cors) * 100) / 100
    	big <- ceiling(max(pd$cors) * 100) / 100
	}

	p <- pd %>% 
		filter(band %in% bands_grab,
			   network %in% networks) %>%
		mutate(network = factor(network, levels = networks)) %>%
		ggplot(aes(x = x, y = y, z = z)) + 
		geom_topo(chan_markers = 'text', aes(fill = cors, label = channel)) +
		facet_grid(band~network) + 
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
			  text = element_text(size=overall_text),
			  axis.ticks = element_blank())

	return(p)

}

