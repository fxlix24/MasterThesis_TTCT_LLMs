library(tidyverse)
data()

view(mpg)
filter (mpg, cty >=20) 

mpg_metric <- mpg %>%
  mutate(cty_metric = 0.425144 * cty)

mpg %>% 
  group_by(class) %>% 
  summarise(mean(cty), median(cty))

ggplot(mpg, aes(x = cty)) + geom_histogram() + labs(x = "City Mileage")

ggplot(mpg, aes(x = cty,
                y = hwy,
                color = class)) +
  geom_point() +
  geom_smooth(method = "lm") +
  scale_color_brewer(palette = "Dark2")