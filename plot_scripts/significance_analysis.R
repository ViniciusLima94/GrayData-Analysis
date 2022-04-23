# Libraries
library(ggraph)
library(ggpubr)
library(igraph)
library(tidyverse)
library(RColorBrewer)


# Session names
sessions <- c(
  '141017', '141014', '141015', '141016', '141023', '141024', '141029',
  '141103', '141112', '141113', '141125', '141126', '141127', '141128',
  '141202', '141203', '141205', '141208', '141209', '141211', '141212',
  '141215', '141216', '141217', '141218', '150114', '150126', '150128',
  '150129', '150205', '150210', '150211', '150212', '150213', '150217',
  '150219', '150223', '150224', '150226', '150227', '150302', '150303',
  '150304', '150305', '150403', '150407', '150408', '150413', '150414',
  '150415', '150416', '150427', '150428', '150429', '150430', '150504',
  '150511', '150512', '150527', '150528', '150529', '150608'
)

# Path to save figures
results <- "/home/vinicius/storage1/projects/GrayData-Analysis/figures/significance_analysis/"
# Root path to read the data
ROOT <- "/home/vinicius/funcog/gda/Results/lucy/significance_analysis/"

# Function to return file name
get_file_name <- function(s_idx) {
  FILE_NAME <- paste(
    c(ROOT, "nedges_coh_", sessions[s_idx], ".csv"),
    collapse = "")
  return(FILE_NAME)
}

read_data <- function(file_name) {
  out <- read.csv(file_name)
  out$X <- NULL
  out$sources <- NULL
  out$targets <- NULL 
  return(out)
}

out <- NULL
for(s_idx in 1:length(sessions)) {
  file_name <- get_file_name(s_idx)
  if(file.exists(file_name)) {
    if(is.null(out)) {
      out <- read_data(file_name)
    } else {
      out <- rbind(out, read_data(file_name))
    }
  }
}

freqs <- unique(out$freqs)
stats <- unique(out$p)

out <- out %>% group_by(roi, freqs, p) %>% summarise(n = sum(n_edges))
out$n <- out$n / length(sessions)
names <- unlist( strsplit(out$roi, "-", fixed=T) )

sources <- NULL
targets <- NULL

count <- 1
for(i in 1:nrow(out)) {
  sources[i] <- names[count]
  count <- count + 1
  targets[i] <- names[count]
  count <- count + 1
}

out$s <- sources
out$t <- targets

################################################################################
# Define function to plot graph with significant links
################################################################################

create_graph <- function(f, stat) {
  # Get freq and stat of interest
  idx <- (out$freqs == f) & (out$p == stat)
  # Filter data-frame
  df <- out[idx, ]
  
  # Binary network
  weights <- df$n
  # Creating network
  edges <- as.data.frame(cbind(unlist(df$s), unlist(df$t)))
  colnames(edges) <- c("from", "to")
  edges$weights <- weights
  
  edges <- edges[order(edges$weights),]
  
  rois <- unique(c(as.character(edges$from), as.character(edges$to)))
  n_rois <- length(rois)
  n_pairs <- length(weights)
  
  nodes <- as.data.frame(rois)
  nodes <- nodes %>% rename(id = rois)
  
  # Create a graph object
  graph <- igraph::graph_from_data_frame( d=edges, vertices=nodes, directed=F )
  strengths <- igraph::strength(graph = graph, weights = edges$weights)
  
  filter <- (edges$weights>0.0)
  p <- ggraph(graph, layout = 'linear', circular = TRUE) + 
    geom_edge_arc(aes(filter=filter, width=edges$weights,
                      color=edges$weights),
                  show.legend=F) +
    scale_edge_color_continuous(low = "white", high = "black", limits=c(0,1),
                                na.value="black") +
    scale_edge_width_continuous(range = c(0, 1)) +
    geom_node_point(aes(x = x*1.07, y=y*1.07),
                    color="orange",
                    size=strengths/10,
                    show.legend=F,
                    alpha=0.6) +
    geom_node_text(aes(label=rois, x=x*1.15, y=y*1.15), color="black",
                   size=2, alpha=1, show.legend=F) +
    theme_void() +
    ggtitle(paste(c(f, " Hz"), collapse = "")) +
    theme(
      plot.title = element_text(hjust = 0.5, size=10),
      plot.margin=unit(c(0,0,0,0),"cm"),
    )
  return(p)
}

################################################################################
# KS-test
################################################################################
myplots <- vector('list', length(freqs))
i<-1
for(f in freqs) {
  p1 <- create_graph(f, "ks")
  myplots[[i]] <- local({
    i <- i
    print(p1)
  })
  i <- i + 1
}

plot <- ggarrange(plotlist=myplots,
                  ncol = length(freqs) / 2, nrow = 2) 
plot
annotate_figure(plot, top = text_grob("Significant links (ks-test)", 
                color = "black", face = "bold", size = 12))
ggsave(
  paste(
    c(results,
      "sig_links_ks_test.png"),
    collapse = ""),
  width = 18, height = 6)

################################################################################
# t-test
################################################################################
myplots <- vector('list', length(freqs))
i<-1
for(f in freqs) {
  p1 <- create_graph(f, "t")
  myplots[[i]] <- local({
    i <- i
    print(p1)
  })
  i <- i + 1
}

plot <- ggarrange(plotlist=myplots,
                  ncol = length(freqs) / 2, nrow = 2) 
annotate_figure(plot, top = text_grob("Significant links (t-test)", 
                                      color = "black", face = "bold", size = 12))
ggsave(
  paste(
    c(results,
      "sig_links_t_test.png"),
    collapse = ""),
  width = 18, height = 6)
