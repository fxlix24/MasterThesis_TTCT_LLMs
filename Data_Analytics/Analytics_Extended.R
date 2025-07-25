################################################################################
#  Extended analyses for AUT creativity dataset
#  ─────────────────────────────────────────────────────────────────────────────
#  1. Models the 'Flexibility' score using a GLMM.
#  2. Calculates and models semantic diversity from idea embeddings.
#  3. Visualizes the conceptual space of ideas using UMAP.
#  ─────────────────────────────────────────────────────────────────────────────
#  Run once if you still need any packages:
#  to_install <- c("DBI", "RMariaDB", "dplyr", "dbplyr", "tidyr", "purrr",
#                  "glmmTMB", "lme4", "lmerTest", "jsonlite", "lsa",
#                  "uwot", "ggplot2", "readr")
#  install.packages(to_install)
################################################################################


# ── 0  SET-UP ──────────────────────────────────────────────────────────────────
# Data & Modeling
library(DBI);         library(RMariaDB)
library(dplyr, warn.conflicts = FALSE);   library(fixest)
library(dbplyr);      library(tidyr);     library(purrr)
library(glmmTMB);     library(lme4);      library(lmerTest)

# Specialized Tools
library(jsonlite);    # For parsing JSON embeddings
library(lsa);         # For cosine distance calculation
library(uwot);        # For UMAP dimensionality reduction
library(ggplot2);     # For plotting
library(readr);       # For saving CSVs


# EDIT ONLY THIS PART ----------------------------------------------------------
con <- dbConnect(
  RMariaDB::MariaDB(),
  host     = "localhost",
  port     = 3306,
  dbname   = "aut_data",
  user     = "automation",
  password = "automation"
)
# ------------------------------------------------------------------------------


# ── 1  LOAD & PRE-PROCESS DATA ────────────────────────────────────────────────
# Load base tables from the database
tbl_requests <- tbl(con, "requests")
tbl_evals    <- tbl(con, "evaluations")
tbl_ideas_1  <- tbl(con, "ideas_aut_1")
tbl_ideas_2  <- tbl(con, "ideas_aut_2")
tbl_ideas_3  <- tbl(con, "ideas_aut_3")

# Create request-level data (for modeling flexibility)
requests_df <- tbl_requests %>%
  dplyr::select(request_id = id, model, task = experiment_phase, prompt) %>%
  collect()
evals_df <- tbl_evals %>%
  dplyr::select(request_id, flexibility) %>%
  collect()
ttct_df <- left_join(evals_df, requests_df, by = "request_id") %>%
  mutate(source = factor(model),
         task = factor(task),
         prompt_id = factor(prompt))

# Create a small data frame to map request_id to experiment_phase
phase_map <- tbl_requests %>%
  dplyr::select(request_id = id, experiment_phase) %>%
  collect()

# Create idea-level data with embeddings and add the experiment_phase
message("Loading ideas and embeddings... (This may take a moment)")
ideas_df_with_embeddings <- bind_rows(
  tbl_ideas_1 %>% mutate(task = 1) %>% collect(),
  tbl_ideas_2 %>% mutate(task = 2) %>% collect(),
  tbl_ideas_3 %>% mutate(task = 3) %>% collect()
) %>%
  left_join(phase_map, by = "request_id") %>% # <-- This is the new line
  dplyr::select(request_id, model, experiment_phase, embedding) %>%
  mutate(embedding_vec = map(embedding, ~ fromJSON(.x)))

message("Data loading complete.")


# ── 2  ANALYSIS 1: MODEL FLEXIBILITY SCORE (GLMM) ────────────────────────────
message("\n── Running Analysis 1: Modeling Flexibility Score... ──")
nb_flexibility <- glmmTMB(
  flexibility ~ source * task + (1 | prompt_id),
  family  = nbinom2,
  data    = ttct_df
)

# Fallback to nbinom1 if convergence fails
if (!isTRUE(nb_flexibility$fit$convergence == 0)) {
  message("nbinom2 did not converge, retrying with nbinom1…")
  nb_flexibility <- glmmTMB(
    flexibility ~ source * task + (1 | prompt_id),
    family  = nbinom1,
    data    = ttct_df
  )
}

saveRDS(nb_flexibility, "model_nb_flexibility.rds")
message("Flexibility model saved to 'model_nb_flexibility.rds'")


# ── 3  ANALYSIS 2: SEMANTIC DIVERSITY FROM EMBEDDINGS ────────────────────────
# Group ideas by request and calculate dispersion for each
# Set a maximum number of ideas to compare per request.
# 500 is a good starting point.
SAMPLE_SIZE <- 500 

# Set a maximum number of ideas to compare per request.
SAMPLE_SIZE <- 500 

dispersion_df <- ideas_df_with_embeddings %>%
  group_by(request_id) %>%
  filter(n() > 1) %>%
  # Use group_modify() to conditionally sample each group
  group_modify(~ {
    if (nrow(.) > SAMPLE_SIZE) {
      sample_n(., SAMPLE_SIZE)
    } else {
      .
    }
  }) %>%
  # The data is now correctly sampled, so we can summarise.
  summarise(
    # This block now contains the FINAL optimization
    semantic_dispersion = {
      # Create matrix efficiently with each idea embedding as a row
      embedding_matrix <- do.call(rbind, embedding_vec)
      # lsa::cosine expects each idea as a column, so we transpose
      cosine_sim <- cosine(t(embedding_matrix))
      # Calculate mean distance and return it
      mean(as.dist(1 - cosine_sim))
    },
    # It's good practice to specify dropping the grouping after summarising
    .groups = "drop"
  )

# Join dispersion scores with model/task info for modeling
semantic_df <- ttct_df %>%
  inner_join(dispersion_df, by = "request_id")

# Model semantic dispersion using a linear mixed model
lmm_dispersion <- feols(
  semantic_dispersion ~ source * task | prompt_id,
  data = semantic_df
)

summary(lmm_dispersion)

saveRDS(lmm_dispersion, "model_lmm_semantic_dispersion.rds")
write_csv(semantic_df, "tbl_semantic_dispersion.csv")
message("Semantic dispersion model saved to 'model_lmm_semantic_dispersion.rds'")
message("Semantic dispersion scores saved to 'tbl_semantic_dispersion.csv'")

# ── 4  ANALYSIS 3: VISUALIZE CONCEPTUAL SPACE (UMAP) FOR EACH TASK ─────────

# Get the unique experiment phases from your data
phases <- unique(ideas_df_with_embeddings$experiment_phase)

# Loop through each phase to create a separate plot
for (phase in phases) {
  message(paste0("\n── Running UMAP for phase: ", phase, " ──"))
  
  # Filter the main dataframe for the current phase
  ideas_phase_data <- ideas_df_with_embeddings %>%
    filter(experiment_phase == phase)
  
  # Use a random sample of up to 20,000 ideas for performance
  set.seed(42) # for reproducibility
  n_sample <- min(nrow(ideas_phase_data), 20000)
  ideas_sample <- ideas_phase_data %>% sample_n(n_sample)
  
  # Extract the embedding vectors into a matrix using a fast method
  embedding_matrix <- do.call(rbind, ideas_sample$embedding_vec)
  
  # Run the UMAP algorithm
  umap_results <- uwot::umap(embedding_matrix, n_neighbors = 15, min_dist = 0.1)
  
  # Prepare data for plotting
  plot_df <- as_tibble(umap_results) %>%
    setNames(c("Dim1", "Dim2")) %>%
    mutate(model = ideas_sample$model)
  
  # Create the plot with a dynamic title
  umap_plot <- ggplot(plot_df, aes(x = Dim1, y = Dim2, color = model)) +
    geom_point(alpha = 0.5, size = 1) +
    guides(color = guide_legend(override.aes = list(alpha = 1, size = 3))) +
    labs(title = paste("Conceptual Space of Ideas for Task:", phase),
         subtitle = paste("UMAP projection of", n_sample, "idea embeddings"),
         x = "UMAP Dimension 1",
         y = "UMAP Dimension 2",
         color = "Model") +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  # Save the plot with a dynamic filename
  plot_filename <- paste0("plot_umap_conceptual_space_", phase, ".png")
  ggsave(plot_filename, umap_plot, width = 12, height = 9, dpi = 300)
  message(paste("UMAP plot saved to", plot_filename))
}

# ── 5  CLEAN-UP ───────────────────────────────────────────────────────────────
dbDisconnect(con)
message("\n✓  All extended analyses complete.")