################################################################################
#  Thesis analyses for AUT creativity dataset
#  ─────────────────────────────────────────────────────────────────────────────
#  Run once if you still need any packages:
#  to_install <- c("DBI","RMariaDB",
#                  "dplyr","dbplyr","tidyr","purrr",
#                  "ggplot2",
#                  "lme4","lmerTest",
#                  "emmeans","glmmTMB","performance",
#                  "ppcor","segmented",
#                  "FSA",
#                  "psych","Kendall",
#                  "broom","readr")
#  install.packages(to_install)
################################################################################

# ── 0  SET‑UP ──────────────────────────────────────────────────────────────────
library(DBI);       library(RMariaDB)      # native MySQL/MariaDB driver
library(dbplyr);    library(tidyr);  library(purrr)
library(ggplot2);   library(lme4);   library(lmerTest)
library(emmeans);   library(glmmTMB); library(performance)
library(ppcor);     library(segmented)
library(FSA);       library(psych);
library(broom);     library(readr);  library(irr)
library(dplyr, warn.conflicts = FALSE)     # load *after* others → dplyr wins

# EDIT ONLY THIS PART ----------------------------------------------------------
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
tbl_requests <- tbl(con, "requests")
tbl_evals    <- tbl(con, "evaluations")
tbl_ideas_1  <- tbl(con, "ideas_aut_1")
tbl_ideas_2  <- tbl(con, "ideas_aut_2")
tbl_ideas_3  <- tbl(con, "ideas_aut_3")

# Request‑level dataset --------------------------------------------------------
requests_df <- tbl_requests %>%
  dplyr::select(
    request_id = id,
    model,
    task        = experiment_phase,
    total_tokens,
    prompt
  ) %>%
  collect()

evals_df <- tbl_evals %>%
  dplyr::select(request_id,
                originality, fluency, flexibility, elaboration) %>%
  collect() %>%
  dplyr::mutate(TTCT_total = originality + fluency + flexibility + elaboration)

ttct_df <- dplyr::left_join(evals_df, requests_df, by = "request_id") %>%
  dplyr::mutate(
    source    = factor(model),
    task      = factor(task),
    prompt_id = factor(prompt)            # random intercept
  )

# Idea‑level dataset -----------------------------------------------------------
ideas_df <- dplyr::bind_rows(
  dplyr::collect(tbl_ideas_1 %>% dplyr::mutate(task = 1)),
  dplyr::collect(tbl_ideas_2 %>% dplyr::mutate(task = 2)),
  dplyr::collect(tbl_ideas_3 %>% dplyr::mutate(task = 3))
) %>%
  dplyr::select(request_id, model, task,
                cluster_id, bullet_number) %>%
  dplyr::arrange(model, request_id, bullet_number)

# ── 2  DATA‑SCREENING ─────────────────────────────────────────────────────────
skim <- ttct_df %>%
  dplyr::summarise(dplyr::across(
    c(TTCT_total, originality, fluency),
    list(min = min,
         q1  = ~quantile(.x, .25),
         med = median,
         q3  = ~quantile(.x, .75),
         max = max)
  ))
print(skim)

# Normality on TTCT_total  ────────────────
set.seed(42)
n_samp <- min(nrow(ttct_df), 5000)               # never exceed population size
shapiro_res <- shapiro.test(sample(ttct_df$TTCT_total, n_samp))
print(shapiro_res)

# Over‑dispersion on fluency  ─────────────
poisson_null <- glm(fluency ~ 1, family = poisson, data = ttct_df)
disp_res     <- performance::check_overdispersion(poisson_null)
print(disp_res)

# ── 3  MAIN TTCT DIFFERENCES (LMM) ────────────────────────────────────────────
lmm_ttct <- lmer(TTCT_total ~ source * task + (1|prompt_id), data = ttct_df)
print(anova(lmm_ttct))
emm <- emmeans(lmm_ttct, pairwise ~ source | task, adjust = "holm")
saveRDS(list(model = lmm_ttct, emmeans = emm), "model_lmm_ttct.rds")

# ── 4  COUNTS: FLUENCY & ELABORATION (NB‑GLMM) ───────────────────────────────
ctrl_long <- glmmTMBControl(optCtrl = list(iter.max = 1e4, eval.max = 1e4))

nb_fluency <- glmmTMB(
  fluency ~ source * task + (1 | prompt_id),
  family  = nbinom2,
  data    = ttct_df,
  control = ctrl_long
)

# If that still fails, fall back to nbinom1
if (!isTRUE(nb_fluency$fit$convergence == 0)) {
  message("nbinom2 did not converge, retrying with nbinom1 …")
  nb_fluency <- glmmTMB(
    fluency ~ source * task + (1 | prompt_id),
    family  = nbinom1,
    data    = ttct_df,
    control = ctrl_long
  )
}

nb_elab <- glmmTMB(
  elaboration ~ source * task + (1 | prompt_id),
  family  = nbinom2,
  data    = ttct_df,
  control = ctrl_long
)

saveRDS(nb_fluency, "model_nb_fluency.rds")
saveRDS(nb_elab,    "model_nb_elaboration.rds")


# ── 5  FLUENCY ↔ ORIGINALITY (partial Spearman) ──────────────────────────────
pcor_res <- ppcor::pcor.test(
  x       = ttct_df$originality,
  y       = ttct_df$fluency,
  z       = as.numeric(ttct_df$task),
  method  = "spearman"
)
print(pcor_res)

# ── 6  NOVELTY‑SATURATION BREAKPOINTS (segmented) ────────────────────────────
saturation_results <- list()

for (m in unique(ideas_df$model)) {
  
  sat_df <- ideas_df %>%
    dplyr::filter(model == m) %>%
    dplyr::arrange(request_id, bullet_number) %>%
    dplyr::mutate(
      prompt_idx     = dplyr::row_number(),
      unique_cluster = !duplicated(cluster_id),
      cum_unique     = cumsum(unique_cluster)
    )
  
  base_lm  <- lm(cum_unique ~ prompt_idx, data = sat_df)
  seg_fit  <- try(
    segmented::segmented(base_lm, seg.Z = ~prompt_idx, psi = 200),
    silent = TRUE
  )
  
  ## Keep only fits that converged *and* contain a psi matrix
  if (!inherits(seg_fit, "try-error") && !is.null(seg_fit$psi)) {
    saturation_results[[m]] <- seg_fit
  } else {
    saturation_results[[m]] <- NULL           # mark as “no breakpoint”
  }
}
saveRDS(saturation_results, "saturation_breakpoints.rds")

# ── 7  UNIQUE‑CLUSTER DIVERSITY (KW + Dunn) ──────────────────────────────────
# Corrected data preparation: count unique clusters per request
unique_clusters_per_request <- ideas_df %>%
  dplyr::group_by(model, request_id) %>%
  dplyr::summarise(n_unique = n_distinct(cluster_id), .groups = "drop")

# Now, these tests are valid because they compare the *distribution* of n_unique for each model
kw_div  <- kruskal.test(n_unique ~ model, data = unique_clusters_per_request)
dunn_div <- FSA::dunnTest(n_unique ~ model, data = unique_clusters_per_request,
                          method = "bonferroni")

# You can also save this more informative table
# readr::write_csv(unique_clusters_per_request, "tbl_unique_clusters_per_request.csv")

saveRDS(list(kw = kw_div, dunn = dunn_div), "unique_cluster_kw_dunn.rds")

# You would also want to export this for your thesis
readr::write_csv(
  unique_clusters_per_request %>%
    group_by(model) %>%
    summarise(
      mean_unique_per_request = mean(n_unique),
      sd_unique_per_request = sd(n_unique),
      total_unique = n_distinct(ideas_df$cluster_id[ideas_df$model == first(model)])
    ),
  "tbl_unique_cluster_summary.csv"
)


# ── 8  COST‑EFFECTIVENESS REGRESSION ─────────────────────────────────────────
ttct_df <- ttct_df %>%
  dplyr::mutate(
    tokens_k        = total_tokens / 1000,
    creativity_per_k = TTCT_total / tokens_k
  )

ce_model <- lm(creativity_per_k ~ source + task, data = ttct_df)
saveRDS(ce_model, "model_cost_effectiveness.rds")

# ── 9  RANK‑STABILITY (Kendall’s W) ─────────────────────────────────────────-
rank_matrix <- ttct_df %>%
  dplyr::group_by(task, source) %>%
  dplyr::summarise(mean_orig = mean(originality), .groups = "drop") %>%
  dplyr::group_by(task) %>%
  dplyr::mutate(rank = rank(-mean_orig)) %>%
  dplyr::select(task, source, rank) %>%
  tidyr::pivot_wider(names_from = task, values_from = rank) %>%
  dplyr::select(-source) %>%
  as.matrix()

# irr::kendall expects RATERS in rows, SUBJECTS in columns
kw_rank <- irr::kendall(t(rank_matrix))
print(kw_rank)

# ── 10  EXPORT TIDY TABLES FOR THESIS ────────────────────────────────────────
# Mixed‑model ANOVA
broom::tidy(anova(lmm_ttct)) %>% readr::write_csv("tbl_anova_ttct.csv")
# Post‑hoc contrasts
as.data.frame(emm$contrasts)  %>% readr::write_csv("tbl_posthoc_ttct.csv")

# ── Breakpoints (SAFE version) ───────────────────────────────────────────────
saturation_results <- list()

for (m in unique(ideas_df$model)) {
  
  sat_df <- ideas_df %>%
    dplyr::filter(model == m) %>%
    dplyr::arrange(request_id, bullet_number) %>%
    dplyr::mutate(
      prompt_idx     = dplyr::row_number(),          # 1 … 100
      unique_cluster = !duplicated(cluster_id),
      cum_unique     = cumsum(unique_cluster)
    )
  
  # data‑driven starting point: midway (≈ 50)
  psi_start <- max(sat_df$prompt_idx) * 0.5
  
  seg_fit <- try(
    segmented::segmented(
      lm(cum_unique ~ prompt_idx, data = sat_df),
      seg.Z = ~prompt_idx,
      psi   = psi_start      # now inside the 1‑100 window
    ),
    silent = TRUE
  )
  
  if (!inherits(seg_fit, "try-error") &&
      !is.null(seg_fit$psi) &&
      nrow(seg_fit$psi) > 0) {
    saturation_results[[m]] <- seg_fit     # keep it
  } else {
    saturation_results[[m]] <- NULL        # mark as “no breakpoint”
  }
}

saveRDS(saturation_results,
        "thesis_results/saturation_breakpoints.rds")

##–– Begin breakpoint block ––––––––––––––––––––––––––––––––––––––––––––––––##
library(dplyr); library(tidyr); library(purrr); library(readr)

# 10.1.  Harvest *segmented* breakpoints  ──────────────────────────────────────
bp_seg <- imap_dfr(
  saturation_results,
  function(fit, m) {
    if (is.null(fit) || is.null(fit$psi) ||
        nrow(fit$psi) == 0 ||
        !"prompt_idx" %in% rownames(fit$psi)) return(NULL)
    tibble(model      = m,
           breakpoint = fit$psi["prompt_idx", "Est."],
           ci_low     = NA_real_,      # CIs optional
           ci_high    = NA_real_,
           method     = "segmented")
  }
)

# 10.2.  Fallback: <1 % gain over last 10 prompts  ─────────────────────────────
fallback_bp <- ideas_df %>%
  group_by(model) %>%
  arrange(request_id, bullet_number) %>%
  mutate(prompt_idx     = row_number(),
         unique_cluster = !duplicated(cluster_id),
         cum_unique     = cumsum(unique_cluster),
         delta10        = cum_unique - lag(cum_unique, 10)) %>%
  filter(is.na(delta10) | delta10 / cum_unique < 0.01) %>%   # <1 % rise
  summarise(breakpoint = min(prompt_idx), .groups = "drop") %>%
  mutate(method = "fallback")

# 10.3.  Combine: use segmented when available, fallback otherwise  ────────────
bp_tbl <- bind_rows(bp_seg, fallback_bp, .id = NULL) %>%
  group_by(model) %>%
  slice_min(order_by = method) %>%          # keep segmented if duplicate
  ungroup()

# 10.4.  Export to CSV  ────────────────────────────────────────────────────────
write_csv(bp_tbl, "thesis_results/tables/tbl_breakpoints.csv")
##–– End breakpoint block ––––––––––––––––––––––––––––––––––––––––––––––––––##


# Cluster diversity counts
readr::write_csv(unique_clusters_per_request, "tbl_unique_clusters_per_request.csv")
# Cost‑effectiveness model
broom::tidy(ce_model)          %>% readr::write_csv("tbl_cost_effectiveness.csv")

# ── 11  CLEAN‑UP ──────────────────────────────────────────────────────────────
dbDisconnect(con)
cat("\n✓  All analyses complete and tables exported.\n")
