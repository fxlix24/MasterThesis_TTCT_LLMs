################################################################################
#  SCRIPT FOR LLM-AS-A-JUDGE VALIDATION AND RELIABILITY ANALYSIS
#  ----------------------------------------------------------------------------
#  This script performs the following steps:
#  1. Loads the reliability data containing scores from the LLM judge and a human juror.
#  2. Cleans the data, ensuring correct data types.
#  3. Calculates the agreement (Spearman's rank correlation) between the human
#     and automated scores for Originality, Elaboration, and Flexibility.
#  4. Saves the results into a tidy CSV file for reporting in the thesis appendix.
################################################################################

# ── 1. LOAD LIBRARIES ────────────────────────────────────────────────────────
# Run install.packages(c("dplyr", "tidyr", "corrr", "readr")) if you don't have them
library(dplyr)
library(tidyr)
library(corrr) # A great package for correlation analysis
library(readr)
library(ggplot2)

# ── 2. LOAD AND PREPARE DATA ─────────────────────────────────────────────────
cat("--- Loading and preparing reliability data ---\n")

# Load the CSV file. Note the semicolon separator.
reliability_df <- read_delim("thesis_tables/Judge_Reliability.csv", delim = ";", show_col_types = FALSE)

# Clean and prepare the data for analysis
reliability_clean <- reliability_df %>%
  dplyr::select(
    auto_originality,
    auto_elaboration,
    auto_flexibility,
    jury1_originality,
    jury1_elaboration,
    jury1_flexibiliy
  ) %>%
  rename(jury1_flexibility = jury1_flexibiliy) %>%
  mutate(across(everything(), as.numeric)) %>%
  tidyr::drop_na()

# Check the number of complete cases
cat(paste("Analysis will be run on", nrow(reliability_clean), "complete response pairs.\n"))


# ── 3. CALCULATE SPEARMAN'S RANK CORRELATION ─────────────────────────────────
cat("\n--- Calculating Spearman's correlation between LLM and Human Juror ---\n")

# --- Originality ---
corr_originality <- cor.test(
  reliability_clean$auto_originality,
  reliability_clean$jury1_originality,
  method = "spearman",
  exact = FALSE # Use approximation for datasets with ties
)

# --- Elaboration ---
corr_elaboration <- cor.test(
  reliability_clean$auto_elaboration,
  reliability_clean$jury1_elaboration,
  method = "spearman",
  exact = FALSE
)

# --- Flexibility ---
corr_flexibility <- cor.test(
  reliability_clean$auto_flexibility,
  reliability_clean$jury1_flexibility,
  method = "spearman",
  exact = FALSE
)

# Print the results to the console to see them immediately
cat("\n--- Correlation for Originality ---\n")
print(corr_originality)

cat("\n--- Correlation for Elaboration ---\n")
print(corr_elaboration)

cat("\n--- Correlation for Flexibility ---\n")
print(corr_flexibility)


# ── 4. CREATE AND SAVE A SUMMARY TABLE (masking-safe) ────────────────────────
cat("\n--- Saving results to a summary table ---\n")

summary_table <- dplyr::bind_rows(
  broom::tidy(corr_originality) %>% dplyr::mutate(trait = "Originality"),
  broom::tidy(corr_elaboration) %>% dplyr::mutate(trait = "Elaboration"),
  broom::tidy(corr_flexibility) %>% dplyr::mutate(trait = "Flexibility")
) %>%
  # transmute = select + rename in one step; avoids non-dplyr select()
  dplyr::transmute(
    trait,
    correlation_rho = estimate,
    p_value        = p.value,
    statistic_S    = statistic
  ) %>%
  dplyr::mutate(dplyr::across(where(is.numeric), ~ round(.x, 3)))

readr::write_csv(summary_table, "thesis_tables/tbl_judge_reliability_summary.csv")

cat("\n✓ Reliability analysis complete.\n")
cat("Results saved to 'thesis_tables/tbl_judge_reliability_summary.csv'\n")
print(summary_table)

# ── 5. BLAND–ALTMAN AGREEMENT ANALYSIS ───────────────────────────────────────
cat("\n--- Bland–Altman agreement analysis (LLM vs Human) ---\n")

# Ensure output dir exists
dir.create("thesis_tables/figures", recursive = TRUE, showWarnings = FALSE)

# Helper: compute BA stats and return list(data, stats, plot)
bland_altman <- function(df, auto_col, jury_col, trait_label) {
  # Prepare
  x <- df[[auto_col]]
  y <- df[[jury_col]]
  stopifnot(is.numeric(x), is.numeric(y))
  
  ba_df <- tibble::tibble(
    mean_score = (x + y) / 2,
    diff_score = x - y
  )
  
  # Core stats
  bias <- mean(ba_df$diff_score, na.rm = TRUE)
  sd_diff <- stats::sd(ba_df$diff_score, na.rm = TRUE)
  loa_lower <- bias - 1.96 * sd_diff
  loa_upper <- bias + 1.96 * sd_diff
  n <- sum(stats::complete.cases(ba_df))
  
  stats_tbl <- tibble::tibble(
    trait = trait_label,
    n = n,
    bias_mean_diff = bias,
    sd_diff = sd_diff,
    loa_lower = loa_lower,
    loa_upper = loa_upper
  )
  
  # Plot
  p <- ggplot(ba_df, aes(x = mean_score, y = diff_score)) +
    geom_point(alpha = 0.35) +
    geom_smooth(method = "loess", formula = y ~ x, se = FALSE, linewidth = 0.6) +
    geom_hline(yintercept = bias, linewidth = 0.7, linetype = "solid") +
    geom_hline(yintercept = loa_lower, linewidth = 0.5, linetype = "dashed") +
    geom_hline(yintercept = loa_upper, linewidth = 0.5, linetype = "dashed") +
    labs(
      title = paste0("Bland–Altman: ", trait_label, " (Auto − Human)"),
      x = "Mean of Auto & Human scores",
      y = "Difference (Auto − Human)"
    ) +
    annotate("text", x = Inf, y = bias, label = sprintf("Bias = %.3f", bias),
             hjust = 1.05, vjust = -0.5) +
    annotate("text", x = Inf, y = loa_upper, label = sprintf("+1.96 SD = %.3f", loa_upper),
             hjust = 1.05, vjust = -0.5) +
    annotate("text", x = Inf, y = loa_lower, label = sprintf("−1.96 SD = %.3f", loa_lower),
             hjust = 1.05, vjust = 1.5) +
    theme_minimal(base_size = 12)
  
  list(data = ba_df, stats = stats_tbl, plot = p)
}

traits <- list(
  list(auto = "auto_originality",  jury = "jury1_originality",  label = "Originality"),
  list(auto = "auto_elaboration",  jury = "jury1_elaboration",  label = "Elaboration"),
  list(auto = "auto_flexibility",  jury = "jury1_flexibility",  label = "Flexibility")
)

ba_stats_all <- dplyr::bind_rows(lapply(traits, function(t) {
  res <- bland_altman(reliability_clean, t$auto, t$jury, t$label)
  # Save figure
  outfile <- file.path("thesis_tables/figures", paste0("ba_", tolower(t$label), ".png"))
  ggsave(outfile, res$plot, width = 7, height = 5, dpi = 300)
  cat("Saved Bland–Altman plot for", t$label, "->", outfile, "\n")
  res$stats
}))

# Round for reporting convenience (keep unrounded if you prefer exact values)
ba_stats_rounded <- ba_stats_all %>%
  mutate(
    dplyr::across(where(is.numeric), ~ round(.x, 3))
  )

readr::write_csv(ba_stats_rounded, "thesis_tables/tbl_bland_altman_stats.csv")
cat("✓ Bland–Altman stats saved to 'thesis_tables/tbl_bland_altman_stats.csv'\n")
print(ba_stats_rounded)