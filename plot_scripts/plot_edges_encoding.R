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
results = "/home/vinicius/storage1/projects/GrayData-Analysis/figures/edge_encoding"
df_coh = read.csv(
  paste(
    c(root,
      "/Results/lucy/mutual_information/mi_coh_fdr.csv"),
    collapse="")
)

df_plv = read.csv(
  paste(
    c(root,
      "/Results/lucy/mutual_information/mi_plv_fdr.csv"),
    collapse="")
)

df_pec = read.csv(
  paste(
    c(root,
      "/Results/lucy/mutual_information/mi_pec_fdr.csv"),
    collapse="")
)

df <- df_coh %>% select(1:5)
df$plv <- df_plv$plv
df$pec <- 0*df_pec$pec

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
    out <- df %>% filter(freqs==f,
                         times==t)
    # Coherence
    neff_coh <- sum(out$coh > 0)
    # PLV
    neff_plv <- sum(out$plv > 0)
    # PEC
    neff_pec <- sum(out $pec > 0)
    # Row for dataframe
    row <- c(f, t, "coh", neff_coh)
    neff[nrow(neff) + 1,] <- row
    row <- c(f, t, "plv", neff_plv)
    neff[nrow(neff) + 1,] <- row
    row <- c(f, t, "pec", neff_pec)
    neff[nrow(neff) + 1,] <- row
  }
}

# Define labels
freqs.labs <- c("3 Hz", "11 Hz", "19 Hz", "27 Hz", "35 Hz",
                "43 Hz", "51 Hz", "59 Hz", "67 Hz", "75 Hz")
names(freqs.labs) <- as.character(unique(neff$freqs))

times.labs <- c("P", "S", "D1", "D2", "Dm")
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
             labeller = labeller(freqs = freqs.labs),
             scales = "free_y") +
  labs(x = "", y = "#sig. edges")

ggsave(
  paste(
    c(results,
      "/mi_edge_enconding_nedges.png"),
    collapse = ""),
  width = 10, height = 4)

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
    out <- df %>% filter(freqs==f,
                         times==t)
    # Coherence
    meff_coh <- mean(out$coh)
    meff_coh_se <- se(out$coh)
    # PLV
    meff_plv <- mean(out$plv)
    meff_plv_se <- se(out$plv)
    # PEC
    meff_pec <- mean(out $pec)
    meff_pec_se <- se(out$pec)
    # Row for dataframe
    row <- c(f, t, "coh", meff_coh, meff_coh_se)
    meff[nrow(meff) + 1,] <- row
    row <- c(f, t, "plv", meff_plv, meff_plv_se) 
    meff[nrow(meff) + 1,] <- row
    row <- c(f, t, "pec", meff_pec, meff_pec_se)
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
             labeller = labeller(freqs = freqs.labs),
             scales = "free_y") +
  labs(x = "", y = "Average effect")


ggsave(
  paste(
    c(results,
      "/mi_edge_enconding_avgeffect.png"),
    collapse = ""),
  width = 10, height = 4)

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
  
  idx = (df$freqs == f) & (df$times == t)
  # Filter frequency and time of interest
  df_filt <- df[idx, ]
  
  # Binary network
  weights <- as.numeric(df_filt[metric] > 0)
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
    stage <- "P"
  } else if(t == 1) {
    stage <- "S"
  } else if(t == 2) {
    stage <- "D1"
  } else if(t == 3) {
    stage <- "D2"
  } else {
    stage <- "Dm"
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
  
  filter <- (edges$weights>0) #& ((edges$from=="a8L") | (edges$to=="a8L"))
  
  p<-ggraph(graph, layout = 'linear', circular = TRUE) + 
    geom_edge_arc(aes(filter=filter),
                  width=.3, color="black",
                  show.legend=F) +
    scale_edge_colour_distiller(palette = "Set1", direction=1,
                                name="", limits=c(0, 20)) +
    geom_node_point(aes(x = x*1.07, y=y*1.07, color=rois, size=strengths),
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
  return(p)
}

################################################################################
# Coherence
################################################################################
myplots <- vector('list', length(times) * length(freqs))
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
myplots <- vector('list', length(times) * length(freqs))
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
myplots <- vector('list', length(times) * length(freqs))
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

################################################################################
# Encoding networks + power encoding
################################################################################

# Load-power encoding
power <- read.csv(
  paste(
    c(root, "/Results/lucy/mutual_information/mi_df_coh_fdr.csv"),
    collapse = "")
)[, 1:4]

create_graph <- function(f, t, metric, title) {
  # Load data
  df = read.csv(
    paste(
      c(root,
        "/Results/lucy/mutual_information/mi_",
        metric,
        "_fdr.csv"),
      collapse="")
  )
  
  # Filter frequency and time of interest
  idx = (df$freqs == f) & (df$times == t)
  df_filt <- df[idx, ]
  # Power filtered t-values
  idx_p = (power$freqs == f) & (power$times == t)
  power_filt <- power[idx_p, ]
  
  # Store t_values
  t_values <- unlist(df_filt[metric])
  t_values <- mean(t_values)
  # Binary network
  weights <- unlist(df_filt[metric] > 0)
  # Creating network
  edges <- df_filt %>% select(6:7)
  edges$weights <- weights
  edges <- edges %>% 
    rename(from = s,
           to = t)
  
  edges <- edges[order(edges$from),]
  
  rois <- unique(c(as.character(edges$from), as.character(edges$to)))
  n_rois <- length(rois)
  n_pairs <- length(weights)
  
  # Getting power t-values for each ROI
  power_t <- rep(0, length(rois))
  i <- 1
  for(roi in rois) {
    if(roi %in% power_filt$roi) {
      power_t[i] <- power_filt[power_filt$roi==roi, ]$power
    } else {
      power_t[i] <- 0
    }
    i <- i + 1
  }
  
  nodes <- as.data.frame(rois)
  nodes <- nodes %>% rename(id = rois)
  
  # Create a graph object
  graph <- igraph::graph_from_data_frame( d=edges, vertices=nodes, directed=F )
  
  strengths <- igraph::strength(graph = graph, weights = edges$weights)
  
  if(title == T) {
    if(t == 0) {
      title <- "P"
    } else if(t == 1) {
      title <- "S"
    } else if(t == 2) {
      title <- "D1"
    } else if(t == 3) {
      title <- "D2"
    } else {
      title <- "Dm"
    }
  } else {
    title <- ""
  }
  
  if(t==0) {
    ylabel <- paste(c(f, "Hz"),
                    collapse=" ") 
  } else {
    ylabel <- " "
  }
  
  filter <- (edges$weights>0) 
  
  p <- ggraph(graph, layout = 'linear', circular = TRUE) + 
    geom_edge_arc(aes(filter = filter,),
                  width = t_values,
                  color = "black",
                  show.legend=F) +
    scale_edge_width_continuous(name="", minor_breaks=seq(0, 20, 0.1),
                                limits=c(0,20)) +
    scale_edge_color_brewer(palette = "Blues",
                            name="") +
    geom_node_point(aes(x = x*1.07, y=y*1.07),
                    color=power_t+1e-10,
                    size=power_t/2,
                    alpha=0.6) +
    geom_node_text(aes(label=rois, x=x*1.15, y=y*1.15),
                   color="black",
                   size=2, alpha=1, show.legend=F) +
    theme_void() +
    ggtitle(title) +
    ylab(ylabel) +
    theme(
      plot.title = element_text(hjust = 0.5, size=10),
      plot.margin=unit(c(0,0,0,0),"cm"),
    ) 
  return(p)
}

count <- 1
for(f in freqs) {
  
  myplots <- vector('list', length(times) * 2)
  i <- 1
  for(t in 0:4) {
    p1 <- create_graph(3, t, "coh", T)
    myplots[[i]] <- local({
      i <- i
      print(p1)
    })
    i <- i + 1
  }
  
  for(t in 0:4) {
    p1 <- create_graph(3, t, "plv", F)
    myplots[[i]] <- local({
      i <- i
      print(p1)
    })
    i <- i + 1
  }
  
  plot <- ggarrange(plotlist=myplots,
            ncol =length(times), nrow = 2,
            labels = c("A", " ", " ", " ", " ",
                       "B", " ", " ", " ", " ")) 
  
  title <- paste(c("Enconding netoworks, band ", count), collapse = "")
  annotate_figure(plot, top = text_grob(title, 
                  color = "black", face = "bold", size = 14))
  
  ggsave(
    paste(
      c(results,
        "/mi_edge_power_encoding_", count,".png"),
      collapse = ""),
    width = 18, height = 6)
  
  count <- count + 1
  
}


################################################################################
# Encoding networks + power encoding
################################################################################

cols <- c("freqs", "times", "edge", "agg")
agg_df <- data.frame(matrix(ncol = length(cols), nrow = 0))
colnames(agg_df) <- cols

for(freq in freqs) {
  for(time in times) {
    
    # Filter frequency and time of interest
    idx = (df$freqs == freq) & (df$times == time)
    df_filt <- df[idx, ]
    # Power filtered t-values
    idx_p = (power$freqs == freq) & (power$times == time)
    power_filt <- power[idx_p, ]
    # Get valid rois from power encoding
    roi <- power_filt$roi
    
    # Iterate over all edges
    for(i in 1:nrow(df_filt)) {
      # Get source and target ROI
      s <- df_filt$s[i]
      t <- df_filt$t[i]
      # Get t_value for coherence encoding
      coh_st <- df_filt$coh[i] 
      if((s %in% roi) && (t %in% roi)) {
        # Edge
        edge <- df_filt$roi[i]
        # Get t-value of power encoding for source and target
        power_s <- power_filt[power_filt$roi==s, ]$power
        power_t <- power_filt[power_filt$roi==t, ]$power
        # Check relations of power-edge enconding
        if((power_s > 0 || power_t > 0) && coh_st > 0) {
          # Has power encoding and edge enconding
          agg <- 1
        } else if((power_s <= 0 && power_t <= 0) && coh_st > 0) {
          # Has no power encoding and edge enconding
          agg <- 0
        } else if((power_s <= 0 && power_t <= 0) && coh_st <= 0) {
          # Has no power encoding and no edge enconding
          agg <- -1
        }
        # Store in data-frame
        row <- c(freq, time, edge, agg)
        agg_df[nrow(agg_df) + 1, ] <- row
      }
    }
  }
}

freqs.labs <- c("3 Hz", "11 Hz", "19 Hz", "27 Hz", "35 Hz",
                "43 Hz", "51 Hz", "59 Hz", "67 Hz", "75 Hz")
names(freqs.labs) <- c(3, 11, 19, 27, 35, 43, 51, 59, 67, 75)

# Stages names
times.labs <- c("P", "S", "D1", "D2", "Dm")
names(times.labs) <- 0:4

mycolors <- c("#72CB3B", "#0341AE", "#FF3213")

agg_df <- agg_df[order(agg_df$edge), ]

agg_df%>% ggplot(aes(x=as.factor(edge),
                      y=as.factor(times),
                      fill = as.factor(agg))) + 
  geom_tile() + 
  scale_y_discrete(labels=times.labs, guide = guide_axis(check.overlap = T)) +
  scale_x_discrete(guide = guide_axis(check.overlap = T)) +
  facet_wrap(~as.numeric(freqs), nrow=10,
             labeller = labeller(freqs = freqs.labs)) +
  theme_minimal() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y = element_text(hjust = 1,
                                   size=6, angle=0),
        axis.text.x = element_text(face="bold", color="#993333", hjust = 1,
                                   size=5, angle=90)) +
  ggtitle("Power-coherence enconding") +
  scale_fill_manual(values = mycolors, name="RPC") +
  xlab("Edges") +
  ylab("")
 
  
ggsave(
  paste(
    c(results,
      "/power_coherence_agg.png"),
    collapse = ""),
  width = 10, height = 16)