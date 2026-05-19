#!/usr/bin/env Rscript

help_text <- function() {
  cat("
Plot HOGENOM species-tree rates from a gpurec optimizer rate TSV.

Required:
  --tree PATH          Species tree in Newick format.
  --rates PATH         Rate TSV written by optimize_hogenom_ccp_wandb.py.
  --out-prefix PATH    Output prefix. The script writes *_D.*, *_T.*, *_L.*,
                       and *_DTL.* files.

Options:
  --layout NAME        ggtree layout: rectangular, circular, fan, radial,
                       slanted, equal_angle, daylight, ... [rectangular]
  --palette NAME       Rate palette, R hcl.colors palette name, or
                       comma-separated colors. Rate palettes include
                       RateHotspot, RateTail, RateContrast, RateBlueRed,
                       and RateFire. RateContrast and RateBlueRed use a
                       linear color scale; RateHotspot, RateTail, and
                       RateFire emphasize the high-rate tail. Examples:
                       RateContrast, RateHotspot, Inferno, YlOrRd,
                       '#59616E,#F97316,#B91C1C' [RateContrast]
  --transform NAME     Color transform: log10, log2, sqrt, identity [log10]
  --scale NAME         Color limits: per-rate or shared [per-rate]
  --branch-length NAME real or none [real]
  --format EXT         Output format supported by ggsave: png, pdf, svg [png]
  --width NUM          Width for each single-rate plot [7]
  --height NUM         Height for each single-rate plot [9]
  --combined-width NUM Width for the combined D/T/L plot [18]
  --combined-height NUM Height for the combined D/T/L plot [9]
  --dpi NUM            DPI for raster outputs [220]
  --line-size NUM      Tree branch line size [0.35]
  --node-size NUM      Node point size [0.65]
  --show-tip-labels    Add tip labels. Usually too dense for HOGENOM.
  --tip-label-size NUM Tip label size when --show-tip-labels is set [1.2]
  --rate-cols LIST     Comma-separated columns to plot [D,T,L]
  --help               Print this help.
")
}

parse_args <- function(argv) {
  args <- list(
    tree = NULL,
    rates = NULL,
    out_prefix = NULL,
    layout = "rectangular",
    palette = "RateContrast",
    transform = "log10",
    scale = "per-rate",
    branch_length = "real",
    format = "png",
    width = 7,
    height = 9,
    combined_width = 18,
    combined_height = 9,
    dpi = 220,
    line_size = 0.35,
    node_size = 0.65,
    show_tip_labels = FALSE,
    tip_label_size = 1.2,
    rate_cols = "D,T,L"
  )

  i <- 1
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (!startsWith(key, "--")) {
      stop(sprintf("unexpected positional argument: %s", key), call. = FALSE)
    }
    name <- gsub("-", "_", sub("^--", "", key))
    if (name %in% c("help", "h")) {
      help_text()
      quit(status = 0)
    }
    if (!name %in% names(args)) {
      stop(sprintf("unknown option: %s", key), call. = FALSE)
    }
    if (is.logical(args[[name]])) {
      args[[name]] <- TRUE
      i <- i + 1
    } else {
      if (i == length(argv) || startsWith(argv[[i + 1]], "--")) {
        stop(sprintf("missing value for option: %s", key), call. = FALSE)
      }
      args[[name]] <- argv[[i + 1]]
      i <- i + 2
    }
  }

  numeric_fields <- c(
    "width", "height", "combined_width", "combined_height", "dpi",
    "line_size", "node_size", "tip_label_size"
  )
  for (field in numeric_fields) {
    args[[field]] <- as.numeric(args[[field]])
    if (is.na(args[[field]])) {
      stop(sprintf("--%s must be numeric", gsub("_", "-", field)), call. = FALSE)
    }
  }

  for (field in c("tree", "rates", "out_prefix")) {
    if (is.null(args[[field]])) {
      stop(sprintf("--%s is required", gsub("_", "-", field)), call. = FALSE)
    }
  }

  args$rate_cols <- trimws(strsplit(args$rate_cols, ",", fixed = TRUE)[[1]])
  if (!args$scale %in% c("per-rate", "shared")) {
    stop("--scale must be either per-rate or shared", call. = FALSE)
  }
  if (!args$branch_length %in% c("real", "none")) {
    stop("--branch-length must be either real or none", call. = FALSE)
  }
  if (!args$transform %in% c("identity", "log10", "log2", "sqrt")) {
    stop("--transform must be one of identity, log10, log2, sqrt", call. = FALSE)
  }
  args
}

validate_input_file <- function(path, option) {
  if (!file.exists(path)) {
    stop(sprintf("--%s file does not exist: %s", option, path), call. = FALSE)
  }
  if (dir.exists(path)) {
    stop(sprintf("--%s must be a file, not a directory: %s", option, path), call. = FALSE)
  }
}

require_packages <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop(
      sprintf(
        "missing R package(s): %s\nInstall CRAN packages with install.packages(); install ggtree with BiocManager::install('ggtree').",
        paste(missing, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  invisible(lapply(pkgs, library, character.only = TRUE))
}

rate_palette_specs <- function() {
  list(
    RateHotspot = list(
      colours = c("#59616E", "#2C7BB6", "#00A6A6", "#80B918", "#F6D743", "#F97316", "#B91C1C"),
      values = c(0.00, 0.45, 0.72, 0.86, 0.94, 0.985, 1.00)
    ),
    RateTail = list(
      colours = c("#8A8F99", "#4D7298", "#1D6996", "#F6D743", "#E95C20", "#8C1D18"),
      values = c(0.00, 0.55, 0.82, 0.93, 0.98, 1.00)
    ),
    RateContrast = list(
      colours = c("#3B4CC0", "#1E88E5", "#00A6A6", "#7CB342", "#FDD835", "#FB8C00", "#D81B60"),
      values = NULL
    ),
    RateBlueRed = list(
      colours = c("#2166AC", "#67A9CF", "#D1E5F0", "#FDD49E", "#EF8A62", "#B2182B"),
      values = NULL
    ),
    RateFire = list(
      colours = c("#222222", "#4B0082", "#B1006E", "#E65100", "#FFB000", "#FFF176"),
      values = c(0.00, 0.45, 0.75, 0.90, 0.97, 1.00)
    )
  )
}

palette_key <- function(name) {
  tolower(gsub("[^[:alnum:]]", "", name))
}

match_palette_name <- function(spec, choices) {
  idx <- match(palette_key(spec), vapply(choices, palette_key, character(1)))
  if (is.na(idx)) {
    NULL
  } else {
    choices[[idx]]
  }
}

palette_spec <- function(spec, n = 256) {
  if (grepl(",", spec, fixed = TRUE)) {
    colors <- trimws(strsplit(spec, ",", fixed = TRUE)[[1]])
    return(list(colours = grDevices::colorRampPalette(colors)(n), values = NULL))
  }

  rate_palettes <- rate_palette_specs()
  rate_palette <- match_palette_name(spec, names(rate_palettes))
  if (!is.null(rate_palette)) {
    return(rate_palettes[[rate_palette]])
  }

  hcl_palette <- match_palette_name(spec, grDevices::hcl.pals())
  if (is.null(hcl_palette)) {
    stop(
      sprintf(
        paste(
          "unknown palette '%s'.",
          "Use a rate palette (%s), one of grDevices::hcl.pals(),",
          "or pass comma-separated colors."
        ),
        spec,
        paste(names(rate_palettes), collapse = ", ")
      ),
      call. = FALSE
    )
  }
  list(colours = grDevices::hcl.colors(n, palette = hcl_palette), values = NULL)
}

limits_for <- function(values, transform) {
  values <- values[is.finite(values)]
  if (transform %in% c("log10", "log2")) {
    values <- values[values > 0]
  }
  if (length(values) == 0) {
    stop("no finite positive values available for requested color transform", call. = FALSE)
  }
  range(values)
}

make_plot <- function(tree, rates, rate_col, args, palette, color_limits) {
  plot_data <- rates[, c("label", rate_col)]
  names(plot_data) <- c("label", "rate_value")
  plot_data$rate_value <- as.numeric(plot_data$rate_value)

  if (args$transform %in% c("log10", "log2") && any(plot_data$rate_value <= 0, na.rm = TRUE)) {
    warning(sprintf("non-positive %s values are hidden on the %s color scale", rate_col, args$transform))
    plot_data$rate_value[plot_data$rate_value <= 0] <- NA_real_
  }

  if (args$branch_length == "none") {
    p <- ggtree::ggtree(
      tree,
      layout = args$layout,
      branch.length = "none",
      size = args$line_size
    )
  } else {
    p <- ggtree::ggtree(tree, layout = args$layout, size = args$line_size)
  }

  p <- p %<+% plot_data
  p <- p +
    ggplot2::aes(color = rate_value) +
    ggtree::geom_point2(
      ggplot2::aes(subset = !is.na(rate_value), color = rate_value),
      size = args$node_size
    )

  if (args$show_tip_labels) {
    p <- p + ggtree::geom_tiplab(size = args$tip_label_size)
  }

  scale_args <- list(
    colours = palette$colours,
    limits = color_limits,
    name = paste0(rate_col, " rate"),
    na.value = "grey82"
  )
  if (!is.null(palette$values)) {
    scale_args$values <- palette$values
  }
  if (args$transform != "identity") {
    scale_args$transform <- args$transform
  }

  p +
    do.call(ggplot2::scale_color_gradientn, scale_args) +
    ggplot2::ggtitle(sprintf("%s rate", rate_col)) +
    ggplot2::theme_void() +
    ggplot2::theme(
      plot.title = ggplot2::element_text(hjust = 0.5, size = 11),
      legend.position = "right"
    )
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  validate_input_file(args$tree, "tree")
  validate_input_file(args$rates, "rates")
  require_packages(c("ape", "ggplot2", "ggtree", "patchwork"))

  tree <- ape::read.tree(args$tree)
  rates <- utils::read.delim(args$rates, stringsAsFactors = FALSE, check.names = FALSE)
  if (!"label" %in% names(rates)) {
    stop("rate TSV must contain a 'label' column", call. = FALSE)
  }
  missing_cols <- setdiff(args$rate_cols, names(rates))
  if (length(missing_cols) > 0) {
    stop(sprintf("rate TSV is missing column(s): %s", paste(missing_cols, collapse = ", ")), call. = FALSE)
  }

  tree_labels <- c(tree$tip.label, tree$node.label)
  missing_rates <- setdiff(tree_labels[nzchar(tree_labels)], rates$label)
  if (length(missing_rates) > 0) {
    warning(sprintf("%d tree labels have no matching rate row; they will be grey", length(missing_rates)))
  }

  palette <- palette_spec(args$palette)
  out_dir <- dirname(args$out_prefix)
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE)
  }

  shared_limits <- NULL
  if (args$scale == "shared") {
    shared_values <- unlist(rates[, args$rate_cols, drop = FALSE], use.names = FALSE)
    shared_limits <- limits_for(as.numeric(shared_values), args$transform)
  }

  plots <- list()
  for (rate_col in args$rate_cols) {
    color_limits <- shared_limits
    if (is.null(color_limits)) {
      color_limits <- limits_for(as.numeric(rates[[rate_col]]), args$transform)
    }
    p <- make_plot(tree, rates, rate_col, args, palette, color_limits)
    plots[[rate_col]] <- p

    single_path <- sprintf("%s_%s.%s", args$out_prefix, rate_col, args$format)
    ggplot2::ggsave(
      single_path,
      p,
      width = args$width,
      height = args$height,
      dpi = args$dpi,
      bg = "white"
    )
    message(sprintf("wrote %s", single_path))
  }

  combined <- patchwork::wrap_plots(plots, ncol = length(plots))
  combined_path <- sprintf("%s_%s.%s", args$out_prefix, paste(args$rate_cols, collapse = ""), args$format)
  ggplot2::ggsave(
    combined_path,
    combined,
    width = args$combined_width,
    height = args$combined_height,
    dpi = args$dpi,
    bg = "white"
  )
  message(sprintf("wrote %s", combined_path))
}

main()
