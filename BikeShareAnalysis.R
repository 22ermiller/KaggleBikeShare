library(tidyverse)
library(vroom)
library(tidymodels)

## Load in Data

bike_train <- vroom("./train.csv") %>%
  select(-casual, -registered)

bike_test <- vroom("./test.csv")

## Prep Recipe

my_recipe <- recipe(count~., data = bike_train) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = as.factor(weather),
              season = as.factor(season),
              workingday = as.factor(workingday),
              holiday = as.factor(holiday)) %>%
  step_time(datetime, features= c("hour"))

prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, bike_test)


# Set up Linear Regression ------------------------------------------------


lin_mod <- linear_reg() %>%
  set_engine("lm") # linear regression model

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>% 
  add_model(lin_mod) %>%
  fit(data = bike_train)

# Make predictions

bike_predictions <- predict(bike_workflow,
                            new_data =bike_test) %>%
  mutate(preds = ifelse(.pred < 0, 0, .pred))

final_predictions <- tibble(datetime = bike_test$datetime, count = bike_predictions$preds)

final_predictions$datetime <- as.character(format(final_predictions$datetime))

vroom_write(final_predictions, "final_predictions.csv", delim = ",")



# Poisson Regression ------------------------------------------------------

library(poissonreg)

pois_mod <- poisson_reg() %>%
  set_engine("glm") # poisson regression model

bike_pois_workflow <- workflow() %>%
  add_recipe(my_recipe) %>% 
  add_model(pois_mod) %>%
  fit(data = bike_train)

bike_pois_predictions <- predict(bike_pois_workflow,
                                 new_data = bike_test)

final_pois_predictions <- tibble(datetime = bike_test$datetime, count = bike_pois_predictions$.pred)

final_pois_predictions$datetime <- as.character(format(final_pois_predictions$datetime))

vroom_write(final_pois_predictions, "final_pois_predictions.csv", delim = ",")
