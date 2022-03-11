# Libraries
library(ggraph)
library(ggpubr)
library(igraph)
library(tidyverse)
library(RColorBrewer)

args = commandArgs(trailingOnly=TRUE)
################################################################################
# Loading data
################################################################################
sessions = c("141017", "141014", "141015", "141016", "141023", "141024", "141029",
             "141103", "141112", "141113", "141125", "141126", "141127", "141128",
             "141202", "141203", "141205", "141208", "141209", "141211", "141212",
             "141215", "141216", "141217", "141218", "150114", "150126", "150128",
             "150129", "150205", "150210", "150211", "150212", "150213", "150217",
             "150219", "150223", "150224", "150226", "150227", "150302", "150303",
             "150304", "150305", "150403", "150407", "150408", "150413", "150414",
             "150415", "150416", "150427", "150428", "150429", "150430", "150504",
             "150511", "150512", "150527", "150528", "150529", "150608")
session <- session[as.integer(args[1])] 
session <- "141017"

# Path to save figures
results <- "/home/vinicius/storage1/projects/GrayData-Analysis/figures/nli/"
# Root path to read the data
ROOT <- "/home/vinicius/funcog/gda/Results/lucy/nli/"

# Function to return file name
get_file_name <- function(s_idx) {
  FILE_NAME <- paste(
    c(ROOT, "nli_coh_", sessions[s_idx], ".csv"),
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

df <- read.csv(
  paste(c(ROOT, "nli_coh_",
          session, ".csv"),
        collapse="")
)

# File with area names
power <- read.csv(
  paste(
    c(ROOT, "mean_power_coh_",
      session, ".csv"),
      collapse="")
)

# Get frequencies
frequencies <- unique(df$f)

################################################################################
# Create igraph object
################################################################################
create_graph <- function(frequency, plot, top) {

  p_filt <- power %>% filter(freqs==frequency)
  df_filt <- df %>% filter(f==frequency)
  edges <- df_filt %>% select(s, t, plot)
  edges <- edges %>% 
    rename(from = s,
           to = t,
           weights = plot)
  
  rois <- unique(c(as.character(edges$from), as.character(edges$to)))
  n_rois <- length(rois)
  n_pairs <- length(edges$weights)
  
  # Keep only top edges
  x <- sort(edges$weights, index.return = TRUE)
  up <- n_pairs- top
  idx <- x$ix[1:up]
  edges$weights[idx]<-0
  
  # create a vertices data.frame. One line per object of our hierarchy
  vertices <- data.frame(
    name =  rois, 
    value = runif(n_rois)
  ) 
  
  # Create a graph object
  graph <- igraph::graph_from_data_frame( edges, directed=FALSE, vertices=vertices )
  
  strengths <- igraph::strength(graph = graph, weights = edges$weights)
  width <- edges$weights/max(edges$weights)
  if(plot=="nli") {
      name <- "NLI"
  } else {
      name <- "COH"
  }
  ################################################################################
  # Create plot
  ################################################################################
  #filter=edges$weights>=cut,
  p<-ggraph(graph, layout = 'linear', circular = TRUE) + 
        geom_edge_arc(aes(filter=edges$weights>0,
                          color=edges$weights),
                      alpha=0.8, width=1) +
        scale_edge_colour_distiller(palette = "YlOrRd", direction=1,
                                    name=name) +
        geom_node_point(aes(x = x*1.07, y=y*1.07, size=p_filt$power*1e10,
                            color=p_filt$roi,
                            alpha=0.2), show.legend=FALSE) +
        geom_node_text(aes(label=p_filt$roi, x=x*1.15, y=y*1.15), color="black",
                       size=2, alpha=1, show.legend=FALSE) +
        theme_void() +
        ggtitle(paste(c(frequency, " Hz"), collapse="")) +
        theme(
          plot.title = element_text(hjust = 0.5, size=10),
          plot.margin=unit(c(0,0,0,0),"cm")
        )
  return(p)
}

################################################################################
# Creating plots
################################################################################
myplots <- vector('list', length(frequencies))
i<-1
for(f in frequencies) {
  p1 <- create_graph(f, "nli", 100)
  myplots[[i]] <- local({
    i <- i
    print(p1)
  })
  i <- i + 1
}

ggarrange(myplots[[1]],
          myplots[[2]],
          myplots[[3]],
          myplots[[4]],
          myplots[[5]],
          myplots[[6]],
          myplots[[7]],
          myplots[[8]],
          myplots[[9]],
          myplots[[10]],
          labels = c("A", "B", "C", "D", "E",
                     "F", "G", "H", "I", "J"),
          ncol = 5, nrow = 2) 
ggsave(
    paste(
    c("figures/nli/nli_",
      session, ".png"),
      collapse="")
    , dpi=300, width = 30, height = 8)

myplots <- vector('list', length(frequencies))
i<-1
for(f in frequencies) {
  p1 <- create_graph(f, "coh_st", 100)
  myplots[[i]] <- local({
    i <- i
    print(p1)
  })
  i <- i + 1
}

ggarrange(myplots[[1]],
          myplots[[2]],
          myplots[[3]],
          myplots[[4]],
          myplots[[5]],
          myplots[[6]],
          myplots[[7]],
          myplots[[8]],
          myplots[[9]],
          myplots[[10]],
          labels = c("A", "B", "C", "D", "E",
                     "F", "G", "H", "I", "J"),
          ncol = 5, nrow = 2) 
ggsave(
    paste(
    c("figures/nli/coh_",
      session, ".png"),
      collapse="")
    , dpi=300, width = 30, height = 8)
