library(tidyverse)
library(umap)

args = commandArgs(trailingOnly=TRUE)

# Metric to use
metric = args[1]  #"coh"

# Path were to save plots
ROOT = "/home/vinicius/storage1/projects/GrayData-Analysis/figures/local_enconding"

# Titles for frequencies
freqs.labs <- c("3 Hz", "11 Hz", "19 Hz", "27 Hz", "35 Hz",
                "43 Hz", "51 Hz", "59 Hz", "67 Hz", "75 Hz")
names(freqs.labs) <- list(3, 11, 19, 27, 35, 43, 51, 59, 67, 75)

# Stages names
stages = c("Baseline", "Cue", "Early delay", "Late delay", "Match")

# Path to the data
data_path = "/home/vinicius/funcog/gda/Results/lucy/mutual_information/"

# File name
file = paste(
  c(data_path, "mi_df_", metric, "_fdr.csv"),
  collapse = "")

# Read data
data <- read.csv(
  file
)

# Re-organize t-values
data <- data %>% gather(key = metric,
                        value = t_values,
                        c("power", "degree", "coreness", "efficiency"))

################################################################################
# CREATE BAR PLOT WITH T-VALUES
################################################################################
for(i in 0:4){
  
  # Gets data for a specific stage
  idx <- (data$times == i)
  df <- data[idx, ]
  
  # Create plot
  title <- paste(
    c(stages[i+1], " (", metric, ")"),
    collapse = "")
  ggplot(data=df, aes(x=roi, y=t_values, fill = metric)) +
    geom_bar(alpha = 0.7, stat='identity', position="stack") +
    ylim(0, 80) +
    coord_flip() +
    facet_wrap(~freqs, ncol=5,
               labeller = labeller(freqs = freqs.labs)) +
    ggtitle(title) +
    theme(plot.title = element_text(hjust=0.5)) +
    labs(y="t-values", x="ROI")
  
  # Save plot
  fig_name <- paste(
    c(ROOT,
      "/mi_hist_degree_",
      stages[i+1], "_",
      metric,
      ".png"),
    collapse = "")
  
  ggsave(
    fig_name,
    width = 12, height = 8)
}

################################################################################
# CREATE ATARI PLOTS
################################################################################
features <- c("power", "degree", "coreness", "efficiency")

data <- read.csv(
  file
)

for(feat in features) {
  # For each roi determines if the stim is encoded by power uniquely,
  # by FC network feature uniquely or both.
  
  out <- NULL
  out <- data %>% select(1:3)
  
  # Get space-time-frequency points with power encoding
  out$power <- as.integer(data$power > 0)
  # Get space-time-frequency points with encoding by a specific feature
  if(feat == "degree") {
    out$fc <- 2 * (data$degree > 0)
  } else if(feat == "coreness") {
    out$fc <- 2 * (data$coreness > 0)
  } else {
    out$fc <- 2 * (data$efficiency > 0)
  }
  # Sum both contributions
  # If power uniquely 
  out$n <- out$power + out$fc
  
  mycolors = c("#FFFFFF", "#D800FF", "#178A00", "#000000")
  times.labs <- c("P", "S", "D1", "D2", "Dm")
  names(times.labs) <- 0:4
  
  label_feature_cbar <- paste(c(feat, " uniquely"), collapse = "")
  title <- paste(c(feat, " (", metric, ") encoding"), collapse="")
  # Create plots
  out %>% ggplot(aes(x=factor(times), y=freqs, fill= as.factor(n))) + 
    scale_x_discrete(labels=times.labs) +
    geom_tile() +
    facet_wrap(~roi, ncol=8) +
    scale_fill_manual(values = mycolors, name="Encoding",
                      labels=c("no encoding","power uniquely",
                               label_feature_cbar, "both")) +
    theme_classic()  +
    theme(plot.title = element_text(hjust=0.5),
          axis.text.x = element_text(angle = 90, hjust=1)) +
    labs(x = "", y = "Freqs [Hz]") +
    ggtitle(title)
    
  # Save figure
  ggsave(
    paste(
      c(ROOT,
        "/mi_enconding_", feat, "_",
        metric,
        ".png"),
      collapse = ""),
    width = 10, height = 8)
}


################################################################################
# Number of areas that encode for each feature
################################################################################

# File name
file = paste(
  c(data_path, "mi_df_", metric, "_fdr.csv"),
  collapse = "")

# Read data
data <- read.csv(
  file
)

# Re-organize t-values
data <- data %>% gather(key = metric,
                        value = t_values,
                        c("power", "degree", "coreness", "efficiency"))

freqs = unique(data$freqs)
times = unique(data$times)
features <- unique(data$metric)

# Data-frame to store number of effects
x <- c("freqs", "times", "feature", "n")
neff <- data.frame(matrix(ncol = length(x), nrow = 0))
colnames(neff) <- x

# Count the number of channels encoding per stage
for(feat in features) {
  for(t in times) {
    for(f in freqs) {
      idx = (data$times == t) & (data$freqs == f) & (data$metric == feat)
      row <- sum(data[idx, ]$t_values > 0)
      row <- c(f, t, feat, row)
      neff[nrow(neff) + 1, ] <- row
    }
  }
}


neff %>% ggplot(aes(x=as.factor(times), y = as.numeric(n),
                    group=as.factor(feature))) +
  geom_line(aes(color=feature))  + 
  geom_point(aes(color=feature)) + 
  scale_x_discrete(labels=times.labs) +
  theme_minimal() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.x = element_text(angle = 45, hjust=1)) +
  facet_wrap(~as.numeric(freqs), ncol=5,
             labeller = labeller(freqs = freqs.labs),
             scales = "free_y") +
  labs(x = "", y = "#sig. areas") +
  ggtitle(paste(c("Number of areas that encode the stimulus (",
                  metric, ")"),
                collapse = ""))

# Save figure
ggsave(
  paste(
    c(ROOT,
      "/nedges_encode_",
      metric,
      ".png"),
    collapse = ""),
  width = 10, height = 6)


################################################################################
# Average effect over areas that encode for each feature
################################################################################

# Function for standard error
se <- function(x) sqrt(var(x) / length(x))

# Data-frame to store number of effects
x <- c("freqs", "times", "feature", "n", "sde")
meff <- data.frame(matrix(ncol = length(x), nrow = 0))
colnames(meff) <- x

# Count the number of channels encoding per stage
for(feat in features) {
  for(t in times) {
    for(f in freqs) {
      idx = (data$times == t) & (data$freqs == f) & (data$metric == feat)
      mean <- mean(data[idx, ]$t_values)
      sde <- se(data[idx, ]$t_values)
      row <- c(f, t, feat, mean, sde)
      meff[nrow(meff) + 1, ] <- row
    }
  }
}

meff %>% ggplot(aes(x=as.factor(times), y = as.numeric(n),
                    group=as.factor(feature))) +
  geom_errorbar(aes(ymin=as.numeric(n)-as.numeric(sde),
                    ymax=as.numeric(n)+as.numeric(sde), color=feature)) +
  geom_line(aes(color=feature))  + 
  geom_point(aes(color=feature)) + 
  scale_x_discrete(labels=times.labs) +
  theme_minimal() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.x = element_text(angle = 45, hjust=1)) +
  facet_wrap(~as.numeric(freqs), ncol=5,
             labeller = labeller(freqs = freqs.labs),
             scales = "free_y") +
  labs(x = "", y = "#sig. areas") +
  ggtitle(paste(c("Number of areas that encode the stimulus (",
                  metric, ")"),
                collapse = ""))


# Save figure
ggsave(
  paste(
    c(ROOT,
      "/avg_effect_encode_",
      metric,
      ".png"),
    collapse = ""),
  width = 10, height = 6)

################################################################################
# Agreement between dFCs
################################################################################

get_fname <- function(metric) {
  # File name
  file = paste(
    c(data_path, "mi_df_", metric, "_fdr.csv"),
    collapse = "")
  return(file)
}


# Read data
data_coh <- read.csv(
  get_fname('coh')
)

data_plv <- read.csv(
  get_fname('plv')
)

data_pec <- read.csv(
  get_fname('pec')
)

# Data-frame to store the coincidence of dFC encoding
x <- c("roi", "feature", "coh_plv", "coh_pec", "plv_pec")
agg <- data.frame(matrix(ncol = length(x), nrow = 0))
colnames(agg) <- x

features <- c("degree", "coreness", "efficiency")
rois <- unique(data_coh$roi)

for(feat in features) {
  for(roi in rois) {
    idx <- (data_coh$roi == roi)
    n_samp <- length(idx)
    coh_plv = mean((data_coh[idx, ][feat] > 0) == (data_plv[idx, ][feat] > 0))
    coh_pec = mean((data_coh[idx, ][feat] > 0) == (data_pec[idx, ][feat] > 0))
    plv_pec = mean((data_plv[idx, ][feat] > 0) == (data_pec[idx, ][feat] > 0))
    row <- c(roi, feat, coh_plv, coh_pec, plv_pec)
    agg[nrow(agg) + 1, ] <- row
  }
}

agg <- agg %>% gather(key = df,
                      value = agg,
                      c("coh_plv", "coh_pec", "plv_pec"))
 
agg %>% ggplot(aes(x=as.factor(roi), y = as.numeric(agg),
                   group=as.factor(df))) +
  geom_line(aes(color=df))  + 
  geom_point(aes(color=df)) + 
  scale_x_discrete(labels=times.labs) +
  labs(color = " ") +
  scale_color_discrete(labels=c("COH-PEC", "COH-PLV", "PLV-PEC")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.x = element_text(angle = 90, hjust=1)) +
  facet_wrap(~feature, ncol=3,
             scales = "free_y") +
  labs(x = "", y = "Ag.") +
  ggtitle("Agreement between the encoding obtained from each dFC")

# Save figure
ggsave(
  paste(
    c(ROOT,
      "/ag_encoding",
      metric,
      ".png"),
    collapse = ""),
  width = 10, height = 4)

################################################################################
# Agreement between dFCs
################################################################################

get_enconding_mat <- function(metric, feat) {
  
  # Read data
  data <- read.csv(
    get_fname(metric)
  )
  
  out <- NULL
  out <- data %>% select(1:3)
  
  # Get space-time-frequency points with power encoding
  out$power <- as.integer(data$power > 0)
  # Get space-time-frequency points with encoding by a specific feature
  if(feat == "degree") {
    out$fc <- 2 * (data$degree > 0)
  } else if(feat == "coreness") {
    out$fc <- 2 * (data$coreness > 0)
  } else {
    out$fc <- 2 * (data$efficiency > 0)
  }
  
  name <- paste(c(metric, "_", feat), collapse = "")
  
  out[name] <- out$power + out$fc
  out[,"power"] <- NULL
  out[,"fc"] <- NULL
  return(out)
}

# Coherence
out <-get_enconding_mat("coh", "degree")
out$coh_coreness <-get_enconding_mat("coh", "coreness")$coh_coreness
out$coh_efficiency <-get_enconding_mat("coh", "efficiency")$coh_efficiency

# PLV
out$plv_degree <-get_enconding_mat("plv", "degree")$plv_degree
out$plv_coreness <-get_enconding_mat("plv", "coreness")$plv_coreness
out$plv_efficiency <-get_enconding_mat("plv", "efficiency")$plv_efficiency

# PEC
out$pec_degree <-get_enconding_mat("pec", "degree")$pec_degree
out$pec_coreness <-get_enconding_mat("pec", "coreness")$pec_coreness
out$pec_efficiency <-get_enconding_mat("pec", "efficiency")$pec_efficiency

metrics <- c("coh", "plv", "pec")

# Data-frame to store the coincidence of dFC encoding
x <- c("roi", "feature", "coh_plv", "coh_pec", "plv_pec")
agg <- data.frame(matrix(ncol = length(x), nrow = 0))
colnames(agg) <- x

return_name <- function(metric, feature) {
  name <- paste(c(metric, "_", feature), collapse = "")
  return(name)
}

for(feat in features) {
  for(roi in rois) {
    idx <- (out$roi == roi)
    # Feature in coherence vs. feature in plv
    name1 <- return_name("coh", feat)
    name2 <- return_name("plv", feat)
    name3 <- return_name("pec", feat)
    coh_plv <- mean(out[idx, ][name1] == out[idx, ][name2])
    # Feature in coherence vs. feature in pec
    coh_pec <- mean(out[idx, ][name1] == out[idx, ][name3])
    # Feature in plv vs. feature in pec
    plv_pec <- mean(out[idx, ][name2] == out[idx, ][name3])
    row <- c(roi, feat, coh_plv, coh_pec, plv_pec)
    agg[nrow(agg) + 1, ] <- row
  }
}

agg <- agg %>% gather(key = df,
                      value = agg,
                      c("coh_plv", "coh_pec", "plv_pec"))

agg %>% ggplot(aes(x=as.factor(roi), y = as.numeric(agg),
                   group=as.factor(df))) +
  geom_line(aes(color=df))  + 
  geom_point(aes(color=df)) + 
  scale_x_discrete(labels=times.labs) +
  labs(color = " ") +
  scale_color_discrete(labels=c("COH-PEC", "COH-PLV", "PLV-PEC")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.x = element_text(angle = 90, hjust=1)) +
  facet_wrap(~feature, ncol=3,
             scales = "free_y") +
  labs(x = "", y = "Ag.") +
  ggtitle("Agreement between the encoding obtained from each dFC")

