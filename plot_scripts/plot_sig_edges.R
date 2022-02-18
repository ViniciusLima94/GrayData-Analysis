# Libraries
library(ggraph)
library(igraph)
library(tidyverse)
library(RColorBrewer)

session = 0

################################################################################
# Loading data
################################################################################
df = read.csv(
  paste(
    c("Results/lucy/significance_analysis/n_edges_",
      session, ".csv"),
    collapse="")
)

# Get frequencies
freqs <- unique(df$freqs)
stats <- unique(df$p)

################################################################################
# Method to plot network
################################################################################
create_graph <- function(frequency, stat) {
  # Filtering properties
  df_filt <- df %>% filter(freqs==frequency,
                      p==stat)
  weights <- df_filt$n_edges
  ################################################################################
  # Creating network
  ################################################################################
  edges <- df_filt %>% select(6:7)
  edges$weights <- weights
  edges <- edges %>% 
    rename(from = sources,
           to = targets)
  
  rois <- unique(c(as.character(edges$from), as.character(edges$to)))
  n_rois <- length(rois)
  n_pairs <- length(weights)
  
  # create a vertices data.frame. One line per object of our hierarchy
  vertices <- data.frame(
    name =  rois, 
    value = runif(n_rois)
  ) 
  
  # Create a graph object
  graph <- igraph::graph_from_data_frame( edges, directed=FALSE, vertices=vertices )
  
  strengths <- igraph::strength(graph = graph, weights = edges$weights)
  
  #edge_width <- edges$weights
  #edge_width <- (edge_width/max(edge_width))**10
  
  # Creating plot
  p <- ggraph(graph, layout = 'linear', circular = TRUE) + 
    geom_edge_arc(aes(filter=edges$weights>=20, color=edges$weights),
                  width=1, alpha=0.8) +
    scale_edge_colour_distiller(palette = "YlOrRd", direction=1,
                                name="# sig. edges") +
    geom_node_point(aes(x = x*1.07, y=y*1.07, size=strengths, color=name,
                        alpha=0.2), show.legend=FALSE) +
    geom_node_text(aes(label=name, x=x*1.2, y=y*1.2, color='black'), size=2,
                   alpha=1, show.legend=FALSE) +
    theme_void() +
    #ggtitle(paste(c(stat, "-test, freq = ", frequency, " Hz"), collapse="")) +
    theme(
      #plot.title = element_text(hjust = 0.5),
      plot.margin=unit(c(0,0,0,0),"cm")
    )
  
  ggsave(
    paste(c("figures/sig_anal/", stat, "_test_freq_", frequency, "_session_",
            session, ".png"),
          collapse=""),
  dpi=600)
}

################################################################################
# saving plots
################################################################################
idx <- 1
for(f in freqs) {
    for(s in stats) {
    create_graph(f, s)
    idx <- idx + 1
  }
}

