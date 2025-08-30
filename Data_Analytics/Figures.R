################################################################################
#  R SCRIPT FOR GENERATING MASTER THESIS FIGURES (FIXED)
#  ----------------------------------------------------------------------------
#  Fixes & safeguards:
#   - Robust package loading & dir creation.
#   - NA-robust TTCT total via rowSums; consistent character model labels.
#   - Canonize model names with vendor-prefix/suffix tolerance (incl. Claude).
#   - Radar plot: stable z-scores (sd==0 -> 0), symmetric axes, safe legend.
#   - Scatter: correct subtitle (95% CI), clean theming.
#   - UMAP: pre-filter, JSON validation, enforce equal lengths, guard empties.
#   - Cost-performance: correct isocost-per-point lines, NA-safe cost math,
#     safer join (character), explicit Claude labels.
################################################################################

# ── 0. SET-UP AND PACKAGE LOADING ─────────────────────────────────────────────
required_packages <- c(
  "DBI", "RMariaDB", "dplyr", "tidyr", "ggplot2", "jsonlite",
  "uwot", "ggrepel", "fmsb", "scales", "purrr", "tibble"
)
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}
library(dplyr, warn.conflicts = FALSE)

set.seed(42)
dir.create("thesis_tables/figures", showWarnings = FALSE, recursive = TRUE)

# ── 0a. Helper functions ─────────────────────────────────────────────────────
zscore_safe <- function(x) {
  mu  <- mean(x, na.rm = TRUE)
  sdv <- sd(x, na.rm = TRUE)
  if (!is.finite(sdv) || sdv == 0) return(rep(0, length(x)))
  (x - mu) / sdv
}

safe_ggsave <- function(path, plot, width, height, dpi = 300) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  ggplot2::ggsave(filename = path, plot = plot, width = width, height = height, dpi = dpi)
}

# Canonize heterogeneous model names to a fixed set used in pricing_df
# - tolerant to vendor prefixes (e.g., "anthropic/claude-3.7-sonnet-2025-02-19")
# - tolerant to version suffixes ("-latest", dates, etc.)
# - tolerant to hyphen/dot variants in version numbers
canonize_model <- function(x) {
  x <- tolower(trimws(as.character(x)))
  x <- gsub("_", "-", x)
  
  # Anthropic
  x <- sub(".*(claude-3[.-]?7-sonnet).*", "claude-3-7-sonnet", x, perl = TRUE)
  x <- sub(".*(claude-3[.-]?5-haiku).*",  "claude-3-5-haiku",  x, perl = TRUE)
  x <- sub(".*(claude-sonnet-4).*",       "claude-sonnet-4",   x, perl = TRUE)
  
  # OpenAI (handle mini BEFORE base; base excludes '-mini')
  x <- sub(".*(gpt-4o-mini).*",           "gpt-4o-mini",       x, perl = TRUE)
  x <- sub(".*(gpt-4o)(?!-mini).*",       "gpt-4o",            x, perl = TRUE)  # <-- key fix
  x <- sub(".*(o4-mini).*",               "o4-mini",           x, perl = TRUE)
  x <- sub(".*(o3).*",                    "o3",                x, perl = TRUE)
  x <- sub(".*(o1).*",                    "o1",                x, perl = TRUE)
  x <- sub(".*(gpt-3[.-]?5-turbo).*",     "gpt-3.5-turbo",     x, perl = TRUE)
  
  # Google
  x <- sub(".*(gemini-2[.-]?5-pro).*",    "gemini-2.5-pro",    x, perl = TRUE)
  x <- sub(".*(gemini-1[.-]?5-pro).*",    "gemini-1.5-pro",    x, perl = TRUE)
  x <- sub(".*(gemini-2[.-]?5-flash).*",  "gemini-2.5-flash",  x, perl = TRUE)
  x <- sub(".*(gemini-2[.-]?0-flash).*",  "gemini-2.0-flash",  x, perl = TRUE)
  
  # DeepSeek
  x <- sub(".*(deepseek-reasoner).*",     "deepseek-reasoner", x, perl = TRUE)
  x <- sub(".*(deepseek-chat).*",         "deepseek-chat",     x, perl = TRUE)
  x
}


# ── 1. CONNECT TO DB ─────────────────────────────────────────────────────────
message("Connecting to database…")
con <- dbConnect(
  RMariaDB::MariaDB(),
  host     = "localhost",
  port     = 3306,
  dbname   = "aut_data",
  user     = "automation",
  password = "automation"
)
on.exit({ if (DBI::dbIsValid(con)) DBI::dbDisconnect(con) }, add = TRUE)

# ── 2. LOAD & PRE-PROCESS DATA ───────────────────────────────────────────────
message("Loading and pre-processing data…")

tbl_requests <- tbl(con, "requests")
tbl_evals    <- tbl(con, "evaluations")

# Detect separate token columns if available (dbplyr supports colnames())
req_cols   <- colnames(tbl_requests)
has_in_out <- all(c("input_tokens", "output_tokens") %in% req_cols)

requests_df <- tbl_requests %>%
  dplyr::select(
    request_id = id,
    model,
    task = experiment_phase,
    dplyr::any_of(c("total_tokens", "input_tokens", "output_tokens"))
  ) %>%
  collect() %>%
  mutate(model = as.character(model))


evals_df <- tbl_evals %>%
  dplyr::select(request_id, originality, fluency, flexibility, elaboration) %>%
  collect() %>%
  mutate(
    TTCT_total = rowSums(cbind(originality, fluency, flexibility, elaboration), na.rm = TRUE)
  )

# Main DF
ttct_df <- left_join(evals_df, requests_df, by = "request_id") %>%
  mutate(source = canonize_model(model))

# Idea-level embeddings for Task 2 only
tbl_ideas_2 <- tbl(con, "ideas_aut_2")
ideas_task2 <- tbl_ideas_2 %>%
  mutate(task = 2) %>%
  dplyr::select(request_id, model, task, bullet_point, embedding) %>%
  collect()

# Parse embeddings safely
ideas_df_with_embeddings <- ideas_task2 %>%
  filter(!is.na(embedding) & embedding != "") %>%
  mutate(
    embedding_vec = purrr::map(embedding, function(x) {
      obj <- tryCatch(jsonlite::fromJSON(x), error = function(e) NULL)
      if (is.null(obj)) return(NULL)
      if (!is.numeric(obj)) obj <- as.numeric(obj)
      obj
    })
  ) %>%
  filter(!purrr::map_lgl(embedding_vec, is.null)) %>%
  mutate(emb_len = purrr::map_int(embedding_vec, length))

if (nrow(ideas_df_with_embeddings) == 0) {
  warning("No valid embeddings found for task 2; Figure 5.3 will be skipped.")
}

# ── 3. GENERATE PLOTS ────────────────────────────────────────────────────────

# 3.1 Figure 5.1 — Radar plot of composite model strategy profiles
message("\n--- Generating Figure 5.1: Radar Plot ---")
TARGET_MODELS <- c("o3", "gemini-2.5-pro", "gpt-3.5-turbo")  # adjust if desired

plot1_long <- ttct_df %>%
  dplyr::select(source, TTCT_total, fluency, originality, flexibility, elaboration) %>%
  tidyr::pivot_longer(cols = -source, names_to = "metric", values_to = "score") %>%
  group_by(metric) %>%
  mutate(z_score = zscore_safe(score)) %>%
  ungroup() %>%
  group_by(source, metric) %>%
  summarise(mean_z = mean(z_score, na.rm = TRUE), .groups = "drop") %>%
  filter(source %in% TARGET_MODELS)

if (nrow(plot1_long) > 0) {
  plot1_wide <- tidyr::pivot_wider(plot1_long, names_from = metric, values_from = mean_z) %>%
    dplyr::select(source, fluency, originality, flexibility, elaboration, TTCT_total)
  
  radar_core <- plot1_wide %>% dplyr::select(-source)
  vals <- as.numeric(unlist(radar_core, use.names = FALSE))
  max_abs <- max(abs(vals), na.rm = TRUE)
  if (!is.finite(max_abs)) max_abs <- 1
  axis_max <- ceiling(max_abs)
  axis_min <- -axis_max
  
  radar_df <- dplyr::bind_rows(
    tibble::as_tibble_row(setNames(rep(axis_max, ncol(radar_core)), colnames(radar_core))),
    tibble::as_tibble_row(setNames(rep(axis_min, ncol(radar_core)), colnames(radar_core))),
    radar_core
  )
  
  num_series  <- nrow(radar_df) - 2
  cols        <- scales::hue_pal()(num_series)
  model_names <- plot1_wide$source
  
  png("thesis_tables/figures/figure_5-1_radar_profiles.png", width = 8, height = 8, units = "in", res = 300)
  par(mar = c(1, 1, 3, 1))
  fmsb::radarchart(
    radar_df,
    axistype = 1,
    pcol  = cols,
    pfcol = scales::alpha(cols, 0.35),
    plwd  = 2,
    plty  = 1,
    cglcol = "grey70",
    cglty  = 1,
    axislabcol = "grey40",
    cglwd = 0.8,
    vlcex = 0.9
  )
  title("Figure 5.1: Composite Strategy Profiles of LLMs")
  legend(
    x = "topright",
    legend = model_names,
    bty = "n", pch = 20, col = cols,
    text.col = "black", cex = 1.1, pt.cex = 2
  )
  dev.off()
  message("Saved: thesis_tables/figures/figure_5-1_radar_profiles.png")
} else {
  warning("Insufficient data to draw Figure 5.1 (Radar Plot).")
}

# 3.2 Figure 5.2 — Fluency vs Originality (scatter with 95% CI)
message("\n--- Generating Figure 5.2: Fluency-Originality Scatter Plot ---")
plot2 <- ggplot(ttct_df, aes(x = fluency, y = originality)) +
  geom_point(alpha = 0.25, shape = 16) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(
    title = "Figure 5.2: Linear Relationship between Fluency and Originality",
    subtitle = "Fitted line with 95% confidence interval across all models and tasks",
    x = "Fluency (Count of Ideas)",
    y = "Originality Score"
  ) +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold"), panel.grid.minor = element_blank())

safe_ggsave("thesis_tables/figures/figure_5-2_fluency_originality.png", plot2, width = 10, height = 7)
message("Saved: thesis_tables/figures/figure_5-2_fluency_originality.png")

# 3.3 Figure 5.3 — UMAP visualization (Paperclip Task)
message("\n--- Generating Figure 5.3: UMAP Visualization ---")
if (nrow(ideas_df_with_embeddings) > 0) {
  # Keep only the most common embedding length to avoid rbind issues
  target_len <- ideas_df_with_embeddings %>%
    dplyr::count(emb_len, sort = TRUE) %>%
    dplyr::slice(1) %>%
    dplyr::pull(emb_len)
  ideas_clean <- ideas_df_with_embeddings %>% dplyr::filter(emb_len == target_len)
  
  n_sample <- min(nrow(ideas_clean), 20000L)
  ideas_sample <- if (n_sample > 0) dplyr::slice_sample(ideas_clean, n = n_sample) else ideas_clean
  
  if (nrow(ideas_sample) > 0) {
    embedding_matrix <- do.call(rbind, ideas_sample$embedding_vec)
    
    message("Running UMAP on sample of ", nrow(ideas_sample), " ideas…")
    umap_results <- uwot::umap(embedding_matrix, n_neighbors = 15, min_dist = 0.1, verbose = TRUE)
    
    plot3_df <- tibble::as_tibble(umap_results) %>%
      setNames(c("UMAP_1", "UMAP_2")) %>%
      mutate(model = canonize_model(ideas_sample$model)) %>%
      mutate(model_group = dplyr::case_when(
        model == "o3" ~ "o3 (High Exploration)",
        model == "gpt-3.5-turbo" ~ "gpt-3.5-turbo (Core Redundancy)",
        TRUE ~ "Other Models"
      )) %>%
      arrange(desc(model_group == "o3 (High Exploration)"))
    
    plot3 <- ggplot(plot3_df, aes(x = UMAP_1, y = UMAP_2, color = model_group)) +
      geom_point(alpha = 0.6, size = 1.5) +
      scale_color_manual(
        values = c(
          "o3 (High Exploration)" = "#0072B2",
          "gpt-3.5-turbo (Core Redundancy)" = "#D55E00",
          "Other Models" = "grey70"
        ),
        name = "Model Group"
      ) +
      guides(color = guide_legend(override.aes = list(alpha = 1, size = 4))) +
      labs(
        title = "Figure 5.3: Conceptual Space Explored by LLMs (Paperclip Task)",
        subtitle = "UMAP projection showing high-tier models exploring peripheral semantic zones",
        x = "UMAP Dimension 1",
        y = "UMAP Dimension 2"
      ) +
      theme_minimal(base_size = 14) +
      theme(plot.title = element_text(face = "bold"), legend.position = "bottom")
    
    safe_ggsave("thesis_tables/figures/figure_5-3_umap_semantic_space.png", plot3, width = 12, height = 9)
    message("Saved: thesis_tables/figures/figure_5-3_umap_semantic_space.png")
  } else {
    warning("No rows available after cleaning embeddings for Figure 5.3.")
  }
} else {
  warning("Skipping Figure 5.3 due to missing/invalid embeddings.")
}

# 3.4 Figure 5.4 — Cost-Performance Curve with isocost lines
message("\n--- Generating Figure 5.4: Cost-Performance Plot ---")

# Pricing (USD per 1k tokens) — adjust as needed
pricing_df <- tibble::tribble(
  ~source, ~cost_per_1k_output, ~cost_per_1k_input,
  "o3",                 0.008,  0.008,
  "gemini-2.5-pro",     0.010,  0.005,
  "gemini-2.5-flash",   0.0025, 0.0015,
  "gemini-2.0-flash",   0.0004, 0.0003,
  "deepseek-reasoner",  0.00219,0.0011,
  "o4-mini",            0.0044, 0.0011,
  "gemini-1.5-pro",     0.005,  0.003,
  "claude-3-7-sonnet",  0.015,  0.003,
  "claude-sonnet-4",    0.015,  0.003,
  "gpt-4o",             0.010,  0.005,
  "o1",                 0.060,  0.030,
  "gpt-3.5-turbo",      0.0015, 0.0005,
  "gpt-4o-mini",        0.0006, 0.0003,
  "claude-3-5-haiku",   0.004,  0.0015,
  "deepseek-chat",      0.0011, 0.0004
)

# Warn if any models lack pricing (after canonization)
missing_price <- setdiff(unique(ttct_df$source), pricing_df$source)
if (length(missing_price) > 0) {
  warning("No pricing found for models: ", paste(missing_price, collapse = ", "))
}

plot4_data <- ttct_df %>%
  group_by(source) %>%
  summarise(
    mean_ttct = mean(TTCT_total, na.rm = TRUE),
    mean_in   = if (has_in_out) mean(input_tokens,  na.rm = TRUE) else NA_real_,
    mean_out  = if (has_in_out) mean(output_tokens, na.rm = TRUE) else NA_real_,
    mean_tot  = if ("total_tokens" %in% req_cols) mean(total_tokens, na.rm = TRUE) else NA_real_,
    .groups = "drop"
  ) %>%
  mutate(
    mean_total_tokens = dplyr::coalesce(mean_in + mean_out, mean_tot),
    source = as.character(source)
  ) %>%
  left_join(pricing_df, by = "source") %>%
  mutate(
    avg_cost_per_response = dplyr::case_when(
      is.finite(mean_in)  & !is.na(mean_in) &
        is.finite(mean_out) & !is.na(mean_out) ~ (mean_in/1000)  * cost_per_1k_input +
        (mean_out/1000) * cost_per_1k_output,
      is.finite(mean_total_tokens) & !is.na(mean_total_tokens) ~ (mean_total_tokens/1000) *
        dplyr::coalesce(cost_per_1k_output, cost_per_1k_input),
      TRUE ~ NA_real_
    ),
    cost_per_point = avg_cost_per_response / mean_ttct
  ) %>%
  filter(is.finite(avg_cost_per_response), is.finite(mean_ttct), mean_ttct > 0)

if (nrow(plot4_data) > 0) {
  # Choose isocost lines based on observed cost-per-point distribution
  cpp_vals <- signif(quantile(plot4_data$cost_per_point, probs = c(0.25, 0.50, 0.75), na.rm = TRUE), 2)
  regimes <- tibble::tibble(cpp = as.numeric(cpp_vals)) %>%
    mutate(label = paste0("$", format(cpp, trim = TRUE, scientific = FALSE), " per point"))
  
  x_range <- range(plot4_data$avg_cost_per_response, na.rm = TRUE)
  y_max   <- max(plot4_data$mean_ttct, na.rm = TRUE) * 1.15
  
  # Label strategy: top cost, best cpp, top TTCT + force Claude variants
  force_sources <- plot4_data %>% slice_max(avg_cost_per_response, n = 1, with_ties = FALSE) %>% pull(source)
  force_sources <- unique(c(force_sources, "claude-3-7-sonnet", "claude-3-5-haiku", "claude-sonnet-4", "gpt-4o"))
  
  label_df <- dplyr::bind_rows(
    plot4_data %>% arrange(cost_per_point) %>% slice_head(n = 8),
    plot4_data %>% slice_max(mean_ttct, n = 4),
    plot4_data %>% filter(source %in% force_sources)
  ) %>% distinct(source, .keep_all = TRUE)
  
  plot4 <- ggplot(plot4_data, aes(x = avg_cost_per_response, y = mean_ttct)) +
    geom_point(aes(color = cost_per_point), size = 3.5) +
    # Isocost-per-point lines: y = (1/cpp) * x
    geom_abline(data = regimes, aes(slope = 1/cpp, intercept = 0),
                linewidth = 0.6, linetype = "dashed", color = "grey40", show.legend = FALSE) +
    # Non-overlapping labels for selected points
    ggrepel::geom_text_repel(
      data = label_df, aes(label = source), size = 3.4, max.overlaps = Inf,
      seed = 42, box.padding = 0.3, point.padding = 0.2, min.segment.length = 0
    ) +
    scale_color_gradient(low = "green", high = "red", name = "Cost per TTCT point") +
    scale_x_log10(labels = scales::dollar_format(), expand = expansion(mult = c(0.02, 0.20))) +
    coord_cartesian(ylim = c(0, y_max)) +
    labs(
      title = "Figure 5.4: Marginal Cost of Creativity",
      subtitle = "Mean TTCT vs. average cost per response; gradient lines = equal $ per TTCT point",
      x = "Average Cost per Response (USD, log scale)",
      y = "Mean TTCT Total Score"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title   = element_text(face = "bold"),
      legend.position = "right",
      legend.title = element_text(size = 10),
      legend.text  = element_text(size = 9)
    ) +
    guides(color = guide_colorbar(title.position = "top", ticks = FALSE))
  
  # Add numeric labels for isocost lines near the right edge
  line_label_df <- regimes %>% mutate(x = x_range[2], y = x / cpp) %>% filter(y <= y_max)
  plot4 <- plot4 + geom_text(data = line_label_df, aes(x = x, y = y, label = label),
                             hjust = 1.02, vjust = -0.3, size = 3.2, inherit.aes = FALSE, color = "grey30")
  
  safe_ggsave("thesis_tables/figures/figure_5-4_cost_performance.png", plot4, width = 12, height = 8)
  message("Saved: thesis_tables/figures/figure_5-4_cost_performance.png")
} else {
  warning("No data to compute cost-performance plot.")
}

# --- Appendix: Token Usage & Efficiency Table ---------------------------------
message("\n--- Generating Appendix Table for Token Usage and Efficiency ---")

# Dependencies (explicit dplyr namespace to avoid masking issues)
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

# ---- 1) Normalize column names (id->request_id, experiment_phase->task, model->source if needed)
if ("id" %in% names(ttct_df) && !"request_id" %in% names(ttct_df)) {
  ttct_df <- ttct_df %>% dplyr::rename(request_id = id)
}
if ("experiment_phase" %in% names(ttct_df) && !"task" %in% names(ttct_df)) {
  ttct_df <- ttct_df %>% dplyr::rename(task = experiment_phase)
}
if (!"source" %in% names(ttct_df) && "model" %in% names(ttct_df)) {
  ttct_df <- ttct_df %>% dplyr::rename(source = model)
}

# Ensure TTCT_total exists (fail early with a clear error)
if (!"TTCT_total" %in% names(ttct_df)) {
  stop("`TTCT_total` column not found in `ttct_df`. Please include it before running this section.")
}

# ---- 2) Keep only the columns we actually need (if present)
needed_cols <- c("request_id", "source", "task", "TTCT_total",
                 "total_tokens", "input_tokens", "output_tokens")
ttct_df <- ttct_df %>%
  dplyr::select(dplyr::any_of(needed_cols))

# ---- 3) Compute total_tokens if missing (using input+output, NA-safe)
if (!"total_tokens" %in% names(ttct_df)) {
  # Create placeholders if token parts are missing
  if (!"input_tokens" %in% names(ttct_df))  ttct_df$input_tokens  <- NA_real_
  if (!"output_tokens" %in% names(ttct_df)) ttct_df$output_tokens <- NA_real_
  
  ttct_df <- ttct_df %>%
    mutate(total_tokens = rowSums(cbind(input_tokens, output_tokens), na.rm = TRUE))
}

# ---- 4) Compute efficiency metric (TTCT points per 1k tokens), guarding /0
ttct_df <- ttct_df %>%
  mutate(
    creativity_per_k = dplyr::if_else(
      is.finite(total_tokens) & !is.na(total_tokens) & total_tokens > 0,
      TTCT_total / (total_tokens / 1000),
      NA_real_
    )
  )

# ---- 5) Build reproducibility summary table by source
# Determine which token parts are available to summarize
has_input  <- "input_tokens"  %in% names(ttct_df)
has_output <- "output_tokens" %in% names(ttct_df)

token_summary_table <- ttct_df %>%
  group_by(source) %>%
  summarise(
    mean_input_tokens  = if (has_input)  mean(input_tokens,  na.rm = TRUE) else NA_real_,
    mean_output_tokens = if (has_output) mean(output_tokens, na.rm = TRUE) else NA_real_,
    mean_total_tokens  = mean(total_tokens, na.rm = TRUE),
    mean_ttct_score    = mean(TTCT_total, na.rm = TRUE),
    ttct_points_per_1k_tokens = mean(creativity_per_k, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(ttct_points_per_1k_tokens)) %>%
  mutate(across(where(is.numeric), ~ round(., 2)))

# ---- 6) Write CSV
if (!dir.exists("thesis_tables")) dir.create("thesis_tables", recursive = TRUE)
readr::write_csv(token_summary_table, "thesis_tables/appendix_token_summary.csv")

message("Saved: thesis_tables/appendix_token_summary.csv")
# ------------------------------------------------------------------------------


message("\n✓ All available plots generated and saved to 'thesis_tables/figures/'.")
