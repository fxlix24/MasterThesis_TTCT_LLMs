# ── 1. LOAD LIBRARIES ────────────────────────────────────────────────────────
library(lmerTest)
library(glmmTMB)
library(emmeans)
library(FSA)
library(segmented)
library(dplyr)
library(purrr)
library(broom)
library(readr)


# ── 2. SET UP DIRECTORY AND FILE PATHS ───────────────────────────────────────
dir.create("thesis_tables", showWarnings = FALSE)

# !!! IMPORTANT: Update these paths to where you have saved your .rds files.
path_lmm_ttct <- "thesis_tables/model_lmm_ttct.rds"
path_nb_fluency <- "thesis_tables/model_nb_fluency.rds"
path_nb_elaboration <- "thesis_tables/model_nb_elaboration.rds"
path_breakpoints <- "thesis_tables/saturation_breakpoints_prediction.rds"
path_diversity <- "thesis_tables/unique_cluster_kw_dunn.rds"
path_cost <- "thesis_tables/model_cost_effectiveness.rds"


# ── 3. EXPERIMENT 1: OVERALL CREATIVITY (LMM) ────────────────────────────────
cat("\n\n### EXPERIMENT 1: OVERALL CREATIVITY ###\n")
results_ttct <- readRDS(path_lmm_ttct)

# Tidy and save the ANOVA table
anova_table <- broom::tidy(anova(results_ttct$model))
readr::write_csv(anova_table, "thesis_tables/tbl_anova_ttct.csv")
cat("\n--- ANOVA Table for TTCT_total ---\n")
print(anova_table)

# Tidy and save the post-hoc comparisons
posthoc_table <- as.data.frame(results_ttct$emmeans$contrasts)
readr::write_csv(posthoc_table, "thesis_tables/tbl_posthoc_ttct.csv")
cat("\n--- Post-Hoc Comparisons for TTCT_total ---\n")
print(posthoc_table)


# ── 4. EXPERIMENT 2: FLUENCY & ELABORATION (NB-GLMM) ─────────────────────────
cat("\n\n### EXPERIMENT 2: FLUENCY & ELABORATION ###\n")
# Optional: enforce default contrasts (harmless if already set)
options(contrasts = c("contr.treatment", "contr.poly"))

# Helper to build a safe ref_grid with factors coerced
safe_refgrid <- function(mod, facs) {
  mf <- model.frame(mod)              # model’s data as used in fitting
  for (v in facs) {
    if (v %in% names(mf)) mf[[v]] <- as.factor(mf[[v]])
  }
  ref_grid(mod, data = mf)            # tell emmeans to use this data
}

# --- Fluency ---
model_fluency <- readRDS(path_nb_fluency)
rg_fluency <- safe_refgrid(model_fluency, c("source", "task"))
posthoc_fluency <- as.data.frame(emmeans(rg_fluency, pairwise ~ source | task)$contrasts)
write_csv(posthoc_fluency, "thesis_tables/tbl_posthoc_fluency.csv")
cat("\n--- Post-Hoc Comparisons for Fluency (saved to file) ---\n")
print(posthoc_fluency)

# --- ELABORATION ---
model_elaboration <- readRDS(path_nb_elaboration)
rg_elaboration <- safe_refgrid(model_elaboration, c("source", "task"))
posthoc_elaboration <- as.data.frame(emmeans(rg_elaboration, pairwise ~ source | task)$contrasts)
write_csv(posthoc_elaboration, "thesis_tables/tbl_posthoc_elaboration.csv")
cat("\n--- Post-Hoc Comparisons for ELABORATION (saved to file) ---\n")
print(posthoc_elaboration)

# ── 5. EXPERIMENT 3: DIVERSITY & SATURATION ──────────────────────────────────
cat("\n\n### EXPERIMENT 3: DIVERSITY & SATURATION ###\n")

# --- Part A: Diversity (Kruskal-Wallis & Dunn's Test) ---
diversity_tests <- readRDS(path_diversity)
kw_table <- broom::tidy(diversity_tests$kw)
dunn_table <- diversity_tests$dunn$res
readr::write_csv(kw_table, "thesis_tables/tbl_kruskal_wallis_diversity.csv")
readr::write_csv(dunn_table, "thesis_tables/tbl_dunn_test_diversity.csv")
cat("\n--- Kruskal-Wallis Test (saved to file) ---\n")
print(kw_table)
cat("\n--- Dunn's Post-Hoc Test (saved to file) ---\n")
print(dunn_table)


# --- Part B: Novelty Saturation Breakpoints ---
cat("\n--- Results for Novelty Saturation ---\n")
saturation_models <- readRDS(path_breakpoints)
breakpoint_list <- list()

for (model_name in names(saturation_models)) {
  fit <- saturation_models[[model_name]]
  
  if (!is.null(fit) && !is.null(fit$psi) && nrow(fit$psi) > 0) {
    cat(paste("\n--- Breakpoint found for:", model_name, "---\n"))
    print(summary(fit))
    
    breakpoint_list[[model_name]] <- tibble(
      model      = model_name,
      # CORRECTED: Access the first row by index [1] instead of by name
      breakpoint = fit$psi[1, "Est."],
      std_error  = fit$psi[1, "St.Err"]
    )
  } else {
    cat(paste("\n--- No significant breakpoint found for:", model_name, "---\n"))
  }
}

# Combine and save the breakpoint results to a single table
breakpoint_table <- bind_rows(breakpoint_list)
readr::write_csv(breakpoint_table, "thesis_tables/tbl_saturation_breakpoints.csv")
cat("\n--- Summary Table of Saturation Breakpoints (saved to file) ---\n")
print(breakpoint_table)


# ── 6. EXPERIMENT 4: COST-EFFECTIVENESS ──────────────────────────────────────
cat("\n\n### EXPERIMENT 4: COST-EFFECTIVENESS ###\n")
model_ce <- readRDS(path_cost)

ce_table <- broom::tidy(model_ce)
readr::write_csv(ce_table, "thesis_tables/tbl_cost_effectiveness.csv")
cat("\n--- Linear Model for Cost-Effectiveness (saved to file) ---\n")
print(ce_table)

cat("\n\n✓ All analyses complete. Tables saved to the 'thesis_tables' folder.\n")