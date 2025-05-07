# install.packages(c("DBI", "RMariaDB"))  # or "RMySQL" on old setups
library(DBI)

con <- dbConnect(
  RMariaDB::MariaDB(),
  dbname   = "aut_experiment",
  host     = "localhost",   # e.g. "localhost" or an IP / DNS name
  user     = "automation",
  password = rstudioapi::askForPassword("MySQL password"),  # keeps it off-disk
  port     = 3306
)

# list tables
dbListTables(con)

sql <-
"SELECT
    r.id AS request_id,
    r.prompt,
    r.experiment_phase,
    r.model,
    r.timestamp AS request_timestamp,
	r.total_tokens,
    res.id AS response_id,
    res.bullet_number,
    res.bullet_text,
    res.evaluation_label,
    res.timestamp AS response_timestamp
FROM requests r
JOIN responses res ON r.id = res.request_id"

df <- dbGetQuery(con, sql)

view(df)


df_summary <- df %>%                       # keep only the numbers you need
  group_by(model) %>%                      # one row per model
  summarise(avg_tokens = mean(total_tokens, na.rm = TRUE))

ggplot(df_summary, aes(model, avg_tokens, fill = model)) +
  geom_col(show.legend = FALSE) +          # geom_col = bar chart where height is already a number
  labs(
    x = "Model",
    y = "Average total tokens",
    title = "Average total-token count by model"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(
      angle = 90,        # rotate
      vjust = 0.5,       # vertically centre the text
      hjust = 1          # right-align so labels sit snug under the ticks
    )
)
dbClearResult(res)
dbDisconnect(con)
