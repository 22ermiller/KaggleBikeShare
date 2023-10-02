## THIS SCRIPT GOES THROUGH THE PROCESS OF DOING MODEL STACKING USING TIDYMODELS ##


library(tidyverse)
library(vroom)
library(tidymodels)
library(stacks)

## Load in Data

bike_train <- vroom("./train.csv") %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))

bike_test <- vroom("./test.csv")

## Prep Recipe

my_recipe <- recipe(count~., data = bike_train) %>%
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

# Split data for CV
folds <- vfold_cv(bike_train, v = 10, repeats = 1)

# Create a control Grid
untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples()

# Linear Model ------------------------------------------------------------

lin_mod <- linear_reg() %>%
  set_engine("lm") # linear regression model

linear_workflow <- workflow() %>%
  add_recipe(my_recipe) %>% 
  add_model(lin_mod)

linear_reg_model <- fit_resamples(
  linear_workflow, 
  resamples = folds,
  metrics = metric_set(rmse),
  control = tunedModel
)


# Penalized Regression Model ----------------------------------------------

penalized_model <- linear_reg(penalty = tune(), mixture = tune()) %>% # Set model and Tuning
  set_engine("glmnet") # Function to fit in R

penalized_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(penalized_model)

pen_tuning_grid <- grid_regular(penalty(),
                                mixture(),
                                levels = 8)

pen_reg_model <- penalized_workflow %>%
  tune_grid(resamples = folds,
            grid = pen_tuning_grid,
            metrics = metric_set(rmse, mae, rsq),
            control = untunedModel)


# Regression Tree ---------------------------------------------------------

tree_model <- decision_tree(tree_depth = tune(),
                            cost_complexity = tune(),
                            min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

tree_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(tree_model)

# Grid of values to tune over
tree_tuning_grid <- grid_regular(tree_depth(),
                            cost_complexity(),
                            min_n(),
                            levels = 5)

tree_reg_model <- tree_wf %>%
  tune_grid(resamples = folds,
            grid = tree_tuning_grid,
            metrics = metric_set(rmse, mae, rsq),
            control = untunedModel)


# Create Stacked Model ----------------------------------------------------

#Specify which models to include
my_stack <- stacks() %>%
  add_candidates(pen_reg_model) %>%
  add_candidates(linear_reg_model) %>%
  add_candidates(tree_reg_model)

# Fit stacked model
stack_mod <- my_stack %>%
  blend_predictions() %>% # LASSO penalized regression (default)
  fit_members()

stacking_predictions <- stack_mod %>% predict(new_data = bike_test)

final_stacking_predictions <- tibble(datetime = bike_test$datetime, count = exp(stacking_predictions$.pred))

final_stacking_predictions$datetime <- as.character(format(final_stacking_predictions$datetime))

vroom_write(final_stacking_predictions, "final_stacking_predictions.csv", delim = ",")
