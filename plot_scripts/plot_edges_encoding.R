# Libraries
library(ggraph)
library(ggpubr)
library(igraph)
library(tidyverse)
library(RColorBrewer)

################################################################################
# Loading data
################################################################################
root = "/home/vinicius/funcog/gda"
results = "/home/vinicius/storage1/projects/GrayData-Analysis/figures/"
df = read.csv(
  paste(
    c(root,
      "/Results/lucy/mutual_information_csd/mi_coh_fdr.csv"),
    collapse="")
)

################################################################################
# Number of sig. effects
################################################################################
freqs <- unique(df$freqs)
times <- unique(df$times)

n_freqs <- length(freqs)
n_times <- length(times)

neff <- data.frame(matrix(ncol = 4, nrow = 0))
x <- c("freqs", "times", "metric", "n")
colnames(neff) <- x

for(f in freqs) {
  for(t in times) {
    # Filter for a freq and time
    
    idx = df$freqs == f & df$times==t
    out <- df[idx, ]
    # Coherence
    neff_coh <- sum(out$coh > 0)
    # Row for dataframe
    row <- c(f, t, "coh", neff_coh)
    neff[nrow(neff) + 1,] <- row
  }
}


times.labs <- c("baseline", "cue", "e. delay", "l. delay", "match")
names(times.labs) <- 0:4

neff %>% ggplot(aes(x=as.factor(times), y = as.numeric(n),
                    group=as.factor(metric))) +
  geom_line(aes(color=metric))  + 
  geom_point(aes(color=metric)) + 
  scale_x_discrete(labels=times.labs) +
  theme_minimal() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.x = element_text(angle = 45, hjust=1)) +
  facet_wrap(~as.numeric(freqs), ncol=5,
             scales = "free_y") +
  labs(x = "", y = "#sig. edges")


ggsave(
  paste(
    c(results,
      "/mi_edge_enconding_nedges.png"),
    collapse = ""),
  width = 10, height = 12)

################################################################################
# Average effect
################################################################################

# Function for standard error
se <- function(x) sqrt(var(x) / length(x))

meff <- data.frame(matrix(ncol = 5, nrow = 0))
x <- c("freqs", "times", "metric", "n", "sde")
colnames(meff) <- x
for(f in freqs) {
  for(t in times) {
    # Filter for a freq and time
    idx = df$freqs == f & df$times==t
    out <- df[idx, ]
    # Coherence
    meff_coh <- mean(out$coh)
    meff_coh_se <- se(out$coh)
    # Row for dataframe
    row <- c(f, t, "coh", meff_coh, meff_coh_se)
    meff[nrow(meff) + 1,] <- row
  }
}

meff %>% ggplot(aes(x=as.factor(times), y = as.numeric(n),
                    group=as.factor(metric))) +
  geom_errorbar(aes(ymin=as.numeric(n)-as.numeric(sde),
                    ymax=as.numeric(n)+as.numeric(sde), color=metric)) +
  geom_line(aes(color=metric))  + 
  geom_point(aes(color=metric)) + 
  scale_x_discrete(labels=times.labs) +
  theme_minimal() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.x = element_text(angle = 45, hjust=1)) +
  facet_wrap(~as.numeric(freqs), ncol=5,
             scales = "free_y") +
  labs(x = "", y = "Average effect")


ggsave(
  paste(
    c(results,
      "/mi_edge_enconding_avgeffect.png"),
    collapse = ""),
  width = 10, height = 12)

################################################################################
# Encoding networks
################################################################################

create_graph <- function(f, t, metric) {
  # Load data
  df = read.csv(
    paste(
      c(root,
        "/Results/lucy/mutual_information/mi_",
        metric,
        "_fdr.csv"),
      collapse="")
  )
  
  idx = as.logical((df$freqs == f) & (df$times == t))
  # Filter frequency and time of interest
  df_filt <- df[idx, ]
  
  # Binary network
  weights <-df_filt[metric]
  # Creating network
  edges <- df_filt %>% select(6:7)
  edges$weights <- unlist(weights)
  edges <- edges %>% 
    rename(from = s,
           to = t)
  
  edges <- edges[order(edges$from),]
  
  
  rois <- unique(c(as.character(edges$from), as.character(edges$to)))
  n_rois <- length(rois)
  n_pairs <- length(weights)
  
  nodes <- as.data.frame(rois)
  nodes <- nodes %>% rename(id = rois)
  
  # Create a graph object
  graph <- igraph::graph_from_data_frame( d=edges, vertices=nodes, directed=F )
  
  strengths <- igraph::strength(graph = graph, weights = edges$weights)
  
  if(t == 0) {
    stage <- "baseline"
  } else if(t == 1) {
    stage <- "cue"
  } else if(t == 2) {
    stage <- "e. delay"
  } else if(t == 3) {
    stage <- "l. delay"
  } else {
    stage <- "match"
  }
  
  if(f==3) {
    title <- stage
  } else {
    title <- " "
  }
  
  if(t==0) {
    ylabel <- paste(c(f, "Hz"),
                  collapse=" ") 
  } else {
    ylabel <- " "
  }
  
  filter <- (edges$weights>=5) #& ((edges$from=="a8L") | (edges$to=="a8L"))
  
  p<-ggraph(graph, layout = 'linear', circular = TRUE) + 
    geom_edge_arc(aes(filter=filter, color=edges$weights),
                  width=1,
                  show.legend=F) +
    scale_edge_colour_distiller(palette = "RdPu", direction=1,
                                name="", limits=c(0, 20)) +
    geom_node_point(aes(x = x*1.07, y=y*1.07, color=rois),
                    show.legend=F,
                    alpha=0.6) +
    geom_node_text(aes(label=rois, x=x*1.15, y=y*1.15), color="black",
                   size=2, alpha=1, show.legend=F) +
    theme_void() +
    ggtitle(title) +
    ylab(ylabel) +
    theme(
      plot.title = element_text(hjust = 0.5, size=10),
      plot.margin=unit(c(0,0,0,0),"cm"),
    ) 
  p
  return(p)
}

################################################################################
# Coherence
################################################################################
myplots <- vector('list', length(times))
i<-1
for(f in freqs) {
  for(t in times) {
    p1 <- create_graph(f, t, "coh")
    myplots[[i]] <- local({
    i <- i
    print(p1)
  })
  i <- i + 1
  }
}

ggarrange(plotlist=myplots,
          ncol = n_times, nrow = n_freqs) 

ggsave(
  paste(
    c(results,
      "/mi_edge_enconding_coh_net.png"),
    collapse = ""),
  width = 14, height = 20)

################################################################################
# PLV
################################################################################
myplots <- vector('list', length(times))
i<-1
for(f in freqs) {
  for(t in times) {
    p1 <- create_graph(f, t, "plv")
    myplots[[i]] <- local({
      i <- i
      print(p1)
    })
    i <- i + 1
  }
}

ggarrange(plotlist=myplots,
          ncol = n_times, nrow = n_freqs) 

ggsave(
  paste(
    c(results,
      "/mi_edge_enconding_plv_net.png"),
    collapse = ""),
  width = 14, height = 20)

################################################################################
# PEC
################################################################################
myplots <- vector('list', length(times))
i<-1
for(f in freqs) {
  for(t in times) {
    p1 <- create_graph(f, t, "pec")
    myplots[[i]] <- local({
      i <- i
      print(p1)
    })
    i <- i + 1
  }
}

ggarrange(plotlist=myplots,
          ncol = n_times, nrow = n_freqs) 

ggsave(
  paste(
    c(results,
      "/mi_edge_enconding_pec_net.png"),
    collapse = ""),
  width = 14, height = 20)
