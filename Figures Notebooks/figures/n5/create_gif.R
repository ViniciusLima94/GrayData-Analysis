library(gifski)
png_files <- list.files(".", pattern = "avalanche_example_lucy*", full.names = TRUE)
gifski(png_files, gif_file = "animation.gif", width = 800, height = 600, delay = 1)
