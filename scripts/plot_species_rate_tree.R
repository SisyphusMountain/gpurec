#!/usr/bin/env Rscript
# Plot the species tree with per-branch DTL rates (one panel per event).
#
# Usage: Rscript plot_species_rate_tree.R <tree.nwk> <rates.csv> <out.png> [title]
#
# The tree's node labels are the model's gp_idx; rates.csv joins on `label`.
# Each branch is coloured by the rate of the node it leads to (log10 viridis).

suppressPackageStartupMessages({
  library(ape)
  library(ggtree)
  library(ggplot2)
  library(treeio)
  library(tidytree)
  library(dplyr)
  library(patchwork)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("need <tree.nwk> <rates.csv> <out.png> [title]")
nwk <- args[1]; csv <- args[2]; out_png <- args[3]
title <- if (length(args) >= 4) args[4] else "Species tree — per-branch DTL rates"

tr <- read.tree(nwk)
df <- read.csv(csv, colClasses = c(label = "character"))

# join the rate table onto the tree by node label -> treedata
tbl <- as_tibble(tr) %>%
  mutate(label = as.character(label)) %>%
  left_join(df, by = "label")
trd <- as.treedata(tbl)

events <- list(
  list(col = "rate_D", name = "Duplication"),
  list(col = "rate_L", name = "Loss"),
  list(col = "rate_T", name = "Transfer")
)

make_panel <- function(ev) {
  ggtree(trd, aes(color = .data[[ev$col]]), size = 0.6) +
    geom_tiplab(aes(label = name), color = "black", size = 1.7, align = TRUE,
                linesize = 0.1) +
    scale_color_viridis_c(trans = "log10", name = paste0(ev$name, "\nrate")) +
    labs(subtitle = ev$name) +
    theme_tree2() +
    theme(legend.position = "right",
          legend.key.height = unit(0.8, "cm"),
          plot.subtitle = element_text(face = "bold")) +
    # leave room on the right for tip labels
    ggtree::xlim_tree(max(tbl$branch.length, na.rm = TRUE) * 0 + 1.35 * max(node.depth.edgelength(tr)))
}

panels <- lapply(events, make_panel)
combined <- wrap_plots(panels, ncol = 3) +
  plot_annotation(title = title,
                  theme = theme(plot.title = element_text(face = "bold", size = 14)))

ggsave(out_png, combined, width = 18, height = 9, dpi = 150, bg = "white")
cat("wrote", out_png, "\n")
