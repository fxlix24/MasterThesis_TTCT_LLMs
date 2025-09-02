################################################################################
#  Thesis analyses for AUT creativity dataset (with table export for all steps)
################################################################################

# ── 0  SET‑UP ──────────────────────────────────────────────────────────────────
# This script assumes you have these packages installed. If not, run:
# install.packages(c("DBI", "RMariaDB", "dplyr", "dbplyr", "tidyr", "purrr", 
#                    "ggplot2", "lme4", "lmerTest", "emmeans", "glmmTMB", 
#                    "performance", "ppcor", "segmented", "FSA", "psych", 
#                    "broom", "broom.mixed", "readr", "irr"))

library(DBI);       library(RMariaDB)
library(dbplyr);    library(tidyr);  library(purrr)
library(ggplot2);   library(lme4);   library(lmerTest)
library(emmeans);   library(glmmTMB); library(performance)
library(ppcor);     library(segmented)
library(FSA);       library(psych);
library(broom);     library(broom.mixed); library(readr); library(irr)
library(dplyr, warn.conflicts = FALSE)

# Create a directory to store all output tables
dir.create("thesis_tables", showWarnings = FALSE)

# --- Connect to your local database ---
# !!! IMPORTANT: Update these connection details if they are different for you.
con <- dbConnect(
  RMariaDB::MariaDB(),
  host     = "localhost",
  port     = 3306,
  dbname   = "aut_data",
  user     = "automation",
  password = "automation"
)
# -----------------------------------------------------------------------------

# ── 1  LOAD & PRE‑PROCESS ─────────────────────────────────────────────────────
# (No changes in this section)
tbl_requests <- tbl(con, "requests")
tbl_evals    <- tbl(con, "evaluations")
tbl_ideas_1  <- tbl(con, "ideas_aut_1")
tbl_ideas_2  <- tbl(con, "ideas_aut_2")
tbl_ideas_3  <- tbl(con, "ideas_aut_3")

requests_df <- tbl_requests %>%
  dplyr::select(request_id = id, model, task = experiment_phase, total_tokens, prompt) %>%
  collect()

evals_df <- tbl_evals %>%
  dplyr::select(request_id, originality, fluency, flexibility, elaboration) %>%
  collect() %>%
  dplyr::mutate(TTCT_total = originality + fluency + flexibility + elaboration)

ttct_df <- dplyr::left_join(evals_df, requests_df, by = "request_id") %>%
  dplyr::mutate(source = factor(model), task = factor(task), prompt_id = factor(prompt))

ideas_df <- dplyr::bind_rows(
  dplyr::collect(tbl_ideas_1 %>% dplyr::mutate(task = 1)),
  dplyr::collect(tbl_ideas_2 %>% dplyr::mutate(task = 2)),
  dplyr::collect(tbl_ideas_3 %>% dplyr::mutate(task = 3))
) %>%
  dplyr::select(request_id, model, task, cluster_id, bullet_number) %>%
  dplyr::arrange(model, request_id, bullet_number)

# ── 2  DATA‑SCREENING ─────────────────────────────────────────────────────────
cat("\n--- Running Data Screening ---\n")
skim <- ttct_df %>%
  dplyr::summarise(dplyr::across(
    c(TTCT_total, originality, fluency),
    list(min = min, q1 = ~quantile(.x, .25), med = median, q3 = ~quantile(.x, .75), max = max)
  ))
# SAVING: Save summary statistics
readr::write_csv(skim, "thesis_tables/tbl_summary_stats.csv")

set.seed(42)
n_samp <- min(nrow(ttct_df), 5000)
shapiro_res <- shapiro.test(sample(ttct_df$TTCT_total, n_samp))
# SAVING: Save normality test results
broom::tidy(shapiro_res) %>% readr::write_csv("thesis_tables/tbl_shapiro_test.csv")

poisson_null <- glm(fluency ~ 1, family = poisson, data = ttct_df)
disp_res     <- performance::check_overdispersion(poisson_null)
# SAVING: Manually create a table for overdispersion results
disp_table <- tibble(
  dispersion_ratio = disp_res$dispersion_ratio,
  p_value          = disp_res$p_value
)
readr::write_csv(disp_table, "thesis_tables/tbl_overdispersion_test.csv")

# ── 3  MAIN TTCT DIFFERENCES (LMM) ────────────────────────────────────────────
cat("\n--- Running LMM on TTCT Total Score ---\n")
lmm_ttct <- lmer(TTCT_total ~ source * task + (1|prompt_id), data = ttct_df)
emm_ttct <- emmeans(lmm_ttct, pairwise ~ source | task, adjust = "holm")
# SAVING: Save full model object
saveRDS(list(model = lmm_ttct, emmeans = emm_ttct), "model_lmm_ttct.rds")
# SAVING: Save ANOVA and Post-Hoc tables
broom.mixed::tidy(anova(lmm_ttct)) %>% readr::write_csv("thesis_tables/tbl_anova_ttct.csv")
as.data.frame(emm_ttct$contrasts)  %>% readr::write_csv("thesis_tables/tbl_posthoc_ttct.csv")

# ── 4  COUNTS: FLUENCY & ELABORATION (NB-GLMM) ───────────────────────────────
cat("\n--- Running NB-GLMM for Fluency & Elaboration ---\n")
ctrl_long <- glmmTMBControl(optCtrl = list(iter.max = 1e4, eval.max = 1e4))
nb_fluency <- glmmTMB(fluency ~ source * task + (1 | prompt_id), family = nbinom2, data = ttct_df, control = ctrl_long)
nb_elab <- glmmTMB(elaboration ~ source * task + (1 | prompt_id), family = nbinom2, data = ttct_df, control = ctrl_long)
# SAVING: Save full model objects
saveRDS(nb_fluency, "model_nb_fluency.rds")
saveRDS(nb_elab,    "model_nb_elaboration.rds")
# SAVING: Save tidied model coefficient tables
broom.mixed::tidy(nb_fluency) %>% readr::write_csv("thesis_tables/tbl_coefficients_fluency.csv")
broom.mixed::tidy(nb_elab) %>% readr::write_csv("thesis_tables/tbl_coefficients_elaboration.csv")

# ── 5  FLUENCY ↔ ORIGINALITY (partial Spearman) ──────────────────────────────
cat("\n--- Running Partial Correlation ---\n")
pcor_res <- ppcor::pcor.test(x = ttct_df$originality, y = ttct_df$fluency, z = as.numeric(ttct_df$task), method = "spearman")
# SAVING: Save partial correlation results directly (it's already a data frame)
readr::write_csv(pcor_res, "thesis_tables/tbl_partial_corr_orig_flu.csv")

# ── 6  NOVELTY‑SATURATION BREAKPOINTS (segmented) ────────────────────────────
cat("\n--- Calculating Novelty Saturation Breakpoints ---\n")
# The robust calculation from your original script is used here
saturation_results <- list()
for (m in unique(ideas_df$model)) {
  sat_df <- ideas_df %>% dplyr::filter(model == m) %>% dplyr::arrange(request_id, bullet_number) %>% dplyr::mutate(prompt_idx = dplyr::row_number(), unique_cluster = !duplicated(cluster_id), cum_unique = cumsum(unique_cluster))
  psi_start <- max(sat_df$prompt_idx) * 0.5
  seg_fit <- try(segmented::segmented(lm(cum_unique ~ prompt_idx, data = sat_df), seg.Z = ~prompt_idx, psi = psi_start), silent = TRUE)
  if (!inherits(seg_fit, "try-error") && !is.null(seg_fit$psi) && nrow(seg_fit$psi) > 0) { saturation_results[[m]] <- seg_fit } else { saturation_results[[m]] <- NULL }
}
bp_seg <- imap_dfr(saturation_results, ~ if (is.null(.x) || is.null(.x$psi)) NULL else tibble(model=.y, breakpoint=.x$psi[1,"Est."], method="segmented"))
fallback_bp <- ideas_df %>% group_by(model) %>% arrange(request_id, bullet_number) %>% mutate(prompt_idx=row_number(), unique_cluster=!duplicated(cluster_id), cum_unique=cumsum(unique_cluster), delta10=cum_unique - lag(cum_unique, 10)) %>% filter(is.na(delta10) | delta10/cum_unique < 0.01) %>% summarise(breakpoint=min(prompt_idx), .groups="drop") %>% mutate(method="fallback")
bp_tbl <- bind_rows(bp_seg, fallback_bp) %>% group_by(model) %>% slice_min(order_by = method) %>% ungroup()
# SAVING: Save the final breakpoint table
readr::write_csv(bp_tbl, "thesis_tables/tbl_saturation_breakpoints.csv")

# --- Your original code starts here ---
saturation_results <- list()

# --- NEW: Create a data frame to store the Davies test p-values ---
davies_test_p_values <- tibble(
  model = character(),
  davies_p_value = numeric()
)

for (m in unique(ideas_df$model)) {
  
  sat_df <- ideas_df %>%
    dplyr::filter(model == m) %>%
    dplyr::arrange(request_id, bullet_number) %>%
    dplyr::mutate(
      idea_index     = dplyr::row_number(),
      unique_cluster = !duplicated(cluster_id),
      cum_unique     = cumsum(unique_cluster)
    )
  
  base_lm  <- lm(cum_unique ~ idea_index, data = sat_df)
  
  # --- NEW: Run and save the Davies test p-value BEFORE fitting segmented ---
  davies_p <- NA_real_ # Default to NA
  try({
    test_result <- davies.test(base_lm, ~ idea_index)
    davies_p <- test_result$p.value
  }, silent = TRUE)
  
  # Add the result to our p-value summary table
  davies_test_p_values <- davies_test_p_values %>%
    add_row(model = m, davies_p_value = davies_p)
  # --- End of new code block ---
  
  # Your original segmented fit code (can remain as is)
  seg_fit  <- try(
    segmented::segmented(base_lm, seg.Z = ~idea_index, psi = 200),
    silent = TRUE
  )
  
  if (!inherits(seg_fit, "try-error") && !is.null(seg_fit$psi)) {
    saturation_results[[m]] <- seg_fit
  } else {
    saturation_results[[m]] <- NULL
  }
}

saveRDS(saturation_results, "thesis_tables/saturation_breakpoints_prediction.rds")

# --- Now, we have a separate table with the p-values ---
print("--- Davies Test P-Values ---")
print(davies_test_p_values)


# --- Apply the Benjamini-Hochberg correction to these p-values ---
davies_test_p_values <- davies_test_p_values %>%
  mutate(
    p_adj_bh = p.adjust(davies_p_value, method = "BH")
  )

print("--- Corrected P-Values ---")
print(davies_test_p_values)


# --- Your original summary-building code starts here ---
# (No changes needed for the code below, but we will JOIN the new p-values at the end)

sat_summary <- imap_dfr(saturation_results, function(fit, model) {
  if (is.null(fit)) {
    return(tibble(
      model = model,
      converged = FALSE,
      break_id = NA_integer_,
      breakpoint = NA_real_,
      # ... (rest of your columns)
      n_obs = NA_integer_
    ))
  }
  
  sl <- as.data.frame(slope(fit)$idea_index) # Adjusted to use 'idea_index'
  psi <- as.data.frame(fit$psi)
  names(psi) <- sub("\\.$", "", names(psi))
  
  map_dfr(seq_len(nrow(psi)), function(i) {
    before_idx <- i
    after_idx  <- min(i + 1, nrow(sl))
    
    tibble(
      model = model,
      converged = TRUE,
      break_id = i,
      breakpoint = psi$Est[i],
      breakpoint_se = psi$St.Err[i],
      slope_before = sl$Est[before_idx],
      slope_before_se = sl$St.Err[before_idx],
      slope_after = sl$Est[after_idx],
      slope_after_se = sl$St.Err[after_idx],
      AIC = AIC(fit),
      BIC = BIC(fit),
      n_obs = stats::nobs(fit)
    )
  })
})


# --- NEW: Join the corrected p-values into your final summary table ---
final_sat_summary <- sat_summary %>%
  left_join(davies_test_p_values, by = "model")


# --- Write the NEW, COMPLETE summary to CSV ---
print("--- Final Summary Table with Corrected P-Values ---")
print(final_sat_summary)

out_path <- "thesis_tables/saturation_breakpoints_summary_with_pvalues.csv"
write.csv(final_sat_summary, out_path, row.names = FALSE)


# ── 7  UNIQUE‑CLUSTER DIVERSITY (KW + Dunn) ──────────────────────────────────
cat("\n--- Analyzing Idea Diversity ---\n")
unique_clusters_per_request <- ideas_df %>% group_by(model, request_id) %>% summarise(n_unique = n_distinct(cluster_id), .groups="drop")
kw_div  <- kruskal.test(n_unique ~ model, data = unique_clusters_per_request)
dunn_div <- FSA::dunnTest(n_unique ~ model, data = unique_clusters_per_request, method = "bonferroni")
# SAVING: Save the test results
broom::tidy(kw_div) %>% readr::write_csv("thesis_tables/tbl_kruskal_wallis_diversity.csv")
readr::write_csv(dunn_div$res, "thesis_tables/tbl_dunn_test_diversity.csv")

# ── 8  COST‑EFFECTIVENESS REGRESSION ─────────────────────────────────────────
cat("\n--- Analyzing Cost-Effectiveness ---\n")
ttct_df <- ttct_df %>% mutate(tokens_k = total_tokens / 1000, creativity_per_k = TTCT_total / tokens_k)
ce_model <- lm(creativity_per_k ~ source + task, data = ttct_df)
# SAVING: Save full model object
saveRDS(ce_model, "model_cost_effectiveness.rds")
# SAVING: Save tidied model results
broom::tidy(ce_model) %>% readr::write_csv("thesis_tables/tbl_cost_effectiveness.csv")

# --- Compare the model rankings ---
# Extract coefficients and rankings from the original Linear Model (LM)
df <- tryCatch(eval(ce_model$call$data, parent.frame()), error = function(e) NULL)
if (is.null(df)) df <- your_data_frame_name_here

req <- c("source", "task")  # add any others you use
stopifnot(all(req %in% names(df)))

df <- dplyr::mutate(df,efficiency = TTCT_total / tokens_k)
df <- dplyr::filter(df, is.finite(efficiency), efficiency > 0)

ce_model <- lm(efficiency ~ source * task, data = df)

gamma_ce_model <- glm(
  efficiency ~ source * task,
  data = df,
  family = Gamma(link = "log")
)

lm_coeffs <- broom::tidy(ce_model) %>%
  filter(grepl("source", term)) %>%
  mutate(model = sub("source", "", term),
         lm_rank = rank(estimate)) %>%
  # *** FIX APPLIED HERE ***
  dplyr::select(model, lm_estimate = estimate, lm_rank)

# Extract coefficients and rankings from the new Gamma GLM
gamma_coeffs <- broom::tidy(gamma_ce_model) %>%
  filter(grepl("source", term)) %>%
  mutate(model = sub("source", "", term),
         gamma_rank = rank(estimate)) %>%
  # *** FIX APPLIED HERE ***
  dplyr::select(model, gamma_estimate = estimate, gamma_rank)

# Join them for a direct comparison
comparison_df <- lm_coeffs %>%
  full_join(gamma_coeffs, by = "model") %>%
  arrange(lm_rank)

cat("\n--- Comparison of Model Rankings (LM vs. Gamma GLM) ---\n")
print(comparison_df)

# Export the comparison table
readr::write_csv(comparison_df, "thesis_tables/tbl_robustness_efficiency_model_comparison.csv")

# Calculate the correlation between the rankings
rank_correlation <- cor(comparison_df$lm_rank, comparison_df$gamma_rank, method = "spearman")
cat(paste("\nSpearman's rank correlation between models:", round(rank_correlation, 4), "\n"))

# ── 9  RANK‑STABILITY (Kendall’s W) ─────────────────────────────────────────-
cat("\n--- Checking Rank Stability ---\n")
rank_table <- ttct_df %>% group_by(task, source) %>% summarise(mean_orig = mean(originality), .groups = "drop") %>% group_by(task) %>% mutate(rank = rank(-mean_orig)) %>% dplyr::select(source, task, rank) %>% tidyr::pivot_wider(names_from = task, values_from = rank)
rank_matrix <- rank_table %>% dplyr::select(-source) %>% as.matrix()
kw_rank <- irr::kendall(t(rank_matrix))
# SAVING: Manually create a table for Kendall's W result
kw_table <- tibble(
  kendalls_w = kw_rank$value,
  statistic  = kw_rank$statistic,
  df         = kw_rank$df,
  p.value    = kw_rank$p.value,
  subjects   = kw_rank$subjects,
  raters     = kw_rank$raters
)
readr::write_csv(rank_table, "thesis_tables/tbl_rank_stability_matrix.csv")
readr::write_csv(kw_table, "thesis_tables/tbl_kendalls_w_result.csv")

# ── 10  CLEAN‑UP ──────────────────────────────────────────────────────────────
dbDisconnect(con)
cat("\n✓  All analyses complete. Tables saved to the 'thesis_tables' folder.\n")
