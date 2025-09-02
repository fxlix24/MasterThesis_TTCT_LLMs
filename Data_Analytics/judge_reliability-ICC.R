###############################################################################
# LLM-as-a-Judge: Validation & Reliability (multi-judge incl. ICC + Δρ/LoA)
# Judges:
#   - auto_*  (original judge, e.g., gpt-4o)
#   - llm2_*  (gemini-2.5-pro)
# Human:
#   - jury1_*
###############################################################################

# 0) Libraries ----------------------------------------------------------------
# install.packages(c("dplyr","tidyr","readr","ggplot2","broom","irr"))
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(broom)
library(irr)

cat("— Loading and preparing reliability data —\n")

# 1) Load data ----------------------------------------------------------------
setwd("/Users/felix/MasterThesis")
df_main  <- read_delim("thesis_tables/Judge_Reliability.csv", delim = ";", show_col_types = FALSE)
df_llm2  <- read_delim("thesis_tables/llm2_test.csv",          delim = ";", show_col_types = FALSE)

# 2) Prepare & join -----------------------------------------------------------
df_main <- df_main %>% rename(jury1_flexibility = jury1_flexibiliy)

keep_main <- c(
  "response_id","request_id","model","experiment_phase",
  "bullet_point","bullet_details",
  "auto_originality","auto_elaboration","auto_flexibility",
  "jury1_originality","jury1_elaboration","jury1_flexibility"
)
df_main <- df_main %>% select(any_of(keep_main))

keep_llm2 <- c("response_id","llm2_originality","llm2_elaboration","llm2_flexibility")
df_llm2  <- df_llm2 %>% select(any_of(keep_llm2))

df <- df_main %>%
  left_join(df_llm2, by = "response_id")

num_cols <- c(
  "auto_originality","auto_elaboration","auto_flexibility",
  "llm2_originality","llm2_elaboration","llm2_flexibility",
  "jury1_originality","jury1_elaboration","jury1_flexibility"
)
df <- df %>% mutate(across(all_of(num_cols), ~ suppressWarnings(as.numeric(.x))))

# 3) Correlations (ρ) judge vs human -----------------------------------------
cat("\n— Calculating Spearman correlations (judge vs human) —\n")

judge_map <- list(
  auto = list(orig = "auto_originality", ela = "auto_elaboration", flex = "auto_flexibility"),
  llm2 = list(orig = "llm2_originality", ela = "llm2_elaboration",  flex = "llm2_flexibility")
)
human_cols   <- list(orig = "jury1_originality", ela = "jury1_elaboration", flex = "jury1_flexibility")
trait_labels <- c(orig = "Originality", ela = "Elaboration", flex = "Flexibility")

corr_rows <- list()

for (judge in names(judge_map)) {
  for (k in names(trait_labels)) {
    auto_col  <- judge_map[[judge]][[k]]
    jury_col  <- human_cols[[k]]
    trait_lab <- trait_labels[[k]]
    
    sub <- df %>% select(all_of(c(auto_col, jury_col))) %>% drop_na()
    if (nrow(sub) == 0L) next
    
    ct <- suppressWarnings(cor.test(sub[[auto_col]], sub[[jury_col]],
                                    method = "spearman", exact = FALSE))
    tr <- broom::tidy(ct) %>%
      mutate(
        judge = judge,
        trait = trait_lab,
        n     = nrow(sub)
      ) %>%
      transmute(
        judge, trait,
        correlation_rho = estimate,
        p_value         = p.value,
        statistic_S     = statistic,
        n
      )
    corr_rows[[length(corr_rows) + 1L]] <- tr
    
    cat(sprintf("  %s × %s: ρ = %.3f (n=%d, p=%.3g)\n",
                judge, trait_lab, tr$correlation_rho, tr$n, tr$p_value))
  }
}

summary_table <- bind_rows(corr_rows) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))

dir.create("thesis_tables", recursive = TRUE, showWarnings = FALSE)
write_csv(summary_table, "thesis_tables/tbl_judge_reliability_summary.csv")
cat("\n✓ Saved: thesis_tables/tbl_judge_reliability_summary.csv\n")

# 4) Bland–Altman (judge − human) --------------------------------------------
cat("\n— Bland–Altman agreement analysis (judge − human) —\n")
dir.create("thesis_tables/figures", recursive = TRUE, showWarnings = FALSE)

bland_altman <- function(x, y, trait_label, judge_label) {
  stopifnot(is.numeric(x), is.numeric(y))
  ba_df <- tibble::tibble(mean_score = (x + y)/2, diff_score = x - y)
  
  bias     <- mean(ba_df$diff_score, na.rm = TRUE)
  sd_diff  <- stats::sd(ba_df$diff_score, na.rm = TRUE)
  loa_low  <- bias - 1.96 * sd_diff
  loa_high <- bias + 1.96 * sd_diff
  n        <- sum(stats::complete.cases(ba_df))
  
  stats_tbl <- tibble::tibble(
    judge = judge_label,
    trait = trait_label,
    n = n,
    bias_mean_diff = bias,
    sd_diff = sd_diff,
    loa_lower = loa_low,
    loa_upper = loa_high,
    loa_width = loa_high - loa_low
  )
  
  p <- ggplot(ba_df, aes(x = mean_score, y = diff_score)) +
    geom_point(alpha = 0.35) +
    geom_smooth(method = "loess", formula = y ~ x, se = FALSE, linewidth = 0.6) +
    geom_hline(yintercept = bias,      linewidth = 0.7) +
    geom_hline(yintercept = loa_low,   linewidth = 0.5, linetype = "dashed") +
    geom_hline(yintercept = loa_high,  linewidth = 0.5, linetype = "dashed") +
    labs(
      title = paste0("Bland–Altman: ", trait_label, " (", judge_label, " − Human)"),
      x = "Mean of Judge & Human scores",
      y = "Difference (Judge − Human)"
    ) +
    annotate("text", x = Inf, y = bias,     label = sprintf("Bias = %.3f", bias),     hjust = 1.05, vjust = -0.5) +
    annotate("text", x = Inf, y = loa_high, label = sprintf("+1.96 SD = %.3f", loa_high), hjust = 1.05, vjust = -0.5) +
    annotate("text", x = Inf, y = loa_low,  label = sprintf("−1.96 SD = %.3f", loa_low),  hjust = 1.05, vjust = 1.5) +
    theme_minimal(base_size = 12)
  
  list(stats = stats_tbl, plot = p)
}

ba_rows <- list()

for (judge in names(judge_map)) {
  for (k in names(trait_labels)) {
    auto_col  <- judge_map[[judge]][[k]]
    jury_col  <- human_cols[[k]]
    trait_lab <- trait_labels[[k]]
    
    sub <- df %>% select(all_of(c(auto_col, jury_col))) %>% drop_na()
    if (nrow(sub) == 0L) next
    
    res <- bland_altman(sub[[auto_col]], sub[[jury_col]], trait_lab, judge)
    
    outfile <- file.path("thesis_tables/figures",
                         paste0("ba_", tolower(trait_lab), "_", judge, ".png"))
    ggsave(outfile, res$plot, width = 7, height = 5, dpi = 300)
    cat("  Saved:", outfile, "\n")
    
    ba_rows[[length(ba_rows) + 1L]] <- res$stats
  }
}

ba_stats <- bind_rows(ba_rows) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))

write_csv(ba_stats, "thesis_tables/tbl_bland_altman_stats.csv")
cat("\n✓ Saved: thesis_tables/tbl_bland_altman_stats.csv\n")

# 5) NEW: ICC(2,1) across (Human, Auto, LLM2) per trait ----------------------
cat("\n— ICC(2,1) across 3 raters (Human, Auto, LLM2) —\n")

icc_rows <- list()

icc_for_trait <- function(df, trait_key, trait_label) {
  cols <- c(
    human = human_cols[[trait_key]],
    auto  = judge_map$auto[[trait_key]],
    llm2  = judge_map$llm2[[trait_key]]
  )
  sub <- df %>% select(all_of(cols)) %>% drop_na()
  if (nrow(sub) == 0L) return(NULL)
  
  # irr::icc expects columns = raters
  icc_res <- irr::icc(
    sub, model = "twoway", type = "agreement", unit = "single", conf.level = 0.95
  )
  tibble::tibble(
    trait   = trait_label,
    icc_type = "ICC(2,1)",
    icc      = icc_res$value,
    lbound   = icc_res$lbound,
    ubound   = icc_res$ubound,
    p_value  = icc_res$p.value,
    k_raters = ncol(sub),
    n_items  = nrow(sub)
  )
}

for (k in names(trait_labels)) {
  out <- icc_for_trait(df, k, trait_labels[[k]])
  if (!is.null(out)) icc_rows[[length(icc_rows) + 1L]] <- out
}

icc_summary <- bind_rows(icc_rows) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))

write_csv(icc_summary, "thesis_tables/tbl_icc_summary.csv")
cat("✓ Saved: thesis_tables/tbl_icc_summary.csv\n")
print(icc_summary)

# 6) NEW: Δρ vs Human and ΔLoA vs Human (LLM2 − Auto) ------------------------
cat("\n— Δρ and ΔLoA (LLM2 − Auto) vs Human —\n")

# reshape ρ table wide by judge so we can compute deltas
rho_wide <- summary_table %>%
  select(judge, trait, correlation_rho) %>%
  tidyr::pivot_wider(names_from = judge, values_from = correlation_rho)

# pick BA rows per judge and join to compare
ba_auto <- ba_stats %>% filter(judge == "auto") %>%
  select(trait, bias_auto = bias_mean_diff,
         loa_lower_auto = loa_lower, loa_upper_auto = loa_upper,
         loa_width_auto = loa_width)
ba_llm2 <- ba_stats %>% filter(judge == "llm2") %>%
  select(trait, bias_llm2 = bias_mean_diff,
         loa_lower_llm2 = loa_lower, loa_upper_llm2 = loa_upper,
         loa_width_llm2 = loa_width)

delta_tbl <- rho_wide %>%
  left_join(ba_auto, by = "trait") %>%
  left_join(ba_llm2, by = "trait") %>%
  mutate(
    delta_rho          = llm2 - auto,
    delta_bias         = bias_llm2 - bias_auto,
    delta_loa_width    = loa_width_llm2 - loa_width_auto
  ) %>%
  select(
    trait,
    rho_auto_human = auto,
    rho_llm2_human = llm2,
    delta_rho,
    bias_auto, bias_llm2, delta_bias,
    loa_lower_auto, loa_upper_auto, loa_width_auto,
    loa_lower_llm2, loa_upper_llm2, loa_width_llm2,
    delta_loa_width
  ) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))

write_csv(delta_tbl, "thesis_tables/tbl_delta_rho_loa.csv")
cat("✓ Saved: thesis_tables/tbl_delta_rho_loa.csv\n")
print(delta_tbl)

cat("\n✓ All analyses complete (ICC + Δρ/LoA added).\n")
