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
  step_date(datetime, features=c("dow")) %>%
  step_time(datetime, features= c("hour", "am")) %>%
  step_poly(temp, degree=4) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, bike_test)


# Set up Linear Regression ------------------------------------------------


lin_mod <- linear_reg() %>%
  set_engine("lm") # linear regression model

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>% 
  add_model(lin_mod) %>%
  fit(data = bike_train_log)

# Make predictions

bike_predictions <- predict(bike_workflow,
                            new_data =bike_test) %>%
  mutate(preds = ifelse(.pred < 0, 0, .pred))

final_predictions <- tibble(datetime = bike_test$datetime, count = exp(bike_predictions$preds))

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

# Penalized Regression ----------------------------------------------------

# for log transformation, making training set response log(y)

bike_train_log <- bike_train %>%
  mutate(count = log(count))

# create new recipe

penalized_recipe <- recipe(count~., data = bike_train) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = as.factor(weather),
              season = as.factor(season),
              workingday = as.factor(workingday),
              holiday = as.factor(holiday)) %>%
  step_date(datetime, features="dow") %>%
  step_time(datetime, features= c("hour")) %>%
  step_poly(temp, degree=4) %>% 
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Penalized Regression Model
penalized_model <- linear_reg(penalty = 0, mixture = 0) %>% # Set model and Tuning
  set_engine("glmnet") # Function to fit in R

penalized_workflow <- workflow() %>%
  add_recipe(penalized_recipe) %>%
  add_model(penalized_model) %>%
  fit(data = bike_train_log)

predict(penalized_workflow, new_data = bike_test)

bike_penalized_predictions <- predict(penalized_workflow, 
                                 new_data = bike_test)

final_penalized_predictions <- tibble(datetime = bike_test$datetime, count = exp(bike_penalized_predictions$.pred))

final_penalized_predictions$datetime <- as.character(format(final_penalized_predictions$datetime))

vroom_write(final_penalized_predictions, "final_penalized_predictions.csv", delim = ",")


# Cross Validation Model --------------------------------------------------

preg_model <- linear_reg(penalty = tune(),
                         mixture = tune()) %>%
  set_engine("glmnet")

preg_wf <- workflow() %>%
  add_recipe(penalized_recipe) %>%
  add_model(preg_model)

# Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 8)

# Split data for cross validation
folds <- vfold_cv(bike_train_log, v = 10, repeats = 1)

# Run the cross validation
cv_results <- preg_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse,mae,rsq)) # here pick the metrics you are interested in

# Plot results
collect_metrics(cv_results) %>%
  filter(.metric == "rmse") %>%
  ggplot(data = ., aes(x = penalty, y = mean, color = factor(mixture))) +
  geom_line()

# find best tuning parameters

bestTune <- cv_results %>%
  select_best("rmse")

# finalize workflow

final_wf <- preg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = bike_train_log)

# predict
bike_penalized_predictions <- predict(final_wf, 
                                      new_data = bike_test)

final_penalized_predictions <- tibble(datetime = bike_test$datetime, count = exp(bike_penalized_predictions$.pred))

final_penalized_predictions$datetime <- as.character(format(final_penalized_predictions$datetime))

vroom_write(final_penalized_predictions, "final_penalized_predictions.csv", delim = ",")


# Regression Tree ---------------------------------------------------------

tree_recipe <- recipe(count~., data = bike_train) # set up recipe (for regression trees, not much data wrangling)

tree_model <- decision_tree(tree_depth = tune(),
                            cost_complexity = tune(),
                            min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

tree_wf <- workflow() %>%
  add_recipe(tree_recipe) %>%
  add_model(tree_model)

# Grid of values to tune over
tuning_grid <- grid_regular(tree_depth(),
                            cost_complexity(),
                            min_n(),
                            levels = 5)

# Split data for cross validation
folds <- vfold_cv(bike_train_log, v = 10, repeats = 1)

# Run the cross validation
cv_results <- tree_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse,mae,rsq)) # here pick the metrics you are interested in

# Plot results
collect_metrics(cv_results) %>%
  filter(.metric == "rmse") %>%
  ggplot(data = ., aes(x = tree_depth, y = mean, color = factor(min_n))) +
  geom_line()

# find best tuning parameters

bestTune <- cv_results %>%
  select_best("rmse")


# finalize workflow

final_wf <- tree_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = bike_train_log)

# predict
bike_tree_predicions <- predict(final_wf, 
                                      new_data = bike_test)

final_tree_predictions <- tibble(datetime = bike_test$datetime, count = exp(bike_tree_predicions$.pred))

final_tree_predictions$datetime <- as.character(format(final_tree_predictions$datetime))

vroom_write(final_tree_predictions, "final_tree_predictions.csv", delim = ",")


# Random Forests ----------------------------------------------------------

forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(forest_mod)

# Grid of values to tune over
tuning_grid <- grid_regular(mtry(range = c(1,10)),
                            min_n())

# Split data for cross validation
folds <- vfold_cv(bike_train_log, v = 10, repeats = 1)

# Run the cross validation
cv_results <- forest_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse,mae,rsq)) # here pick the metrics you are interested in

# Plot results
collect_metrics(cv_results) %>%
  filter(.metric == "rmse") %>%
  ggplot(data = ., aes(x = tree_depth, y = mean, color = factor(min_n))) +
  geom_line()

# find best tuning parameters

bestTune <- cv_results %>%
  select_best("rmse")


# finalize workflow

final_wf <- forest_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = bike_train_log)

# predict
bike_forest_predictions <- predict(final_wf, 
                                new_data = bike_test)

final_forest_predictions <- tibble(datetime = bike_test$datetime, count = exp(bike_forest_predictions$.pred))

final_forest_predictions$datetime <- as.character(format(final_forest_predictions$datetime))

vroom_write(final_forest_predictions, "final_forest_predictions.csv", delim = ",")













