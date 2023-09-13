## Bike Share EDA Code

## Libraries

library(tidyverse)
library(vroom)
library(skimr)
library(DataExplorer)
library(patchwork)

## Read in Data

bike <- vroom("./train.csv") %>%
  mutate(weather = as.factor(weather),
         season = as.factor(season))

glimpse(bike)

skim(bike)

# correlation plot
correlation_plot <- plot_correlation(bike)

# Explore the collinearity between the temperature and the temp it feels like

temp_graph <- ggplot(data = bike) +
  geom_point(mapping = aes(x = temp, y = atemp)) +
  labs(title = "Collinearity of Temp vs 'Feels like' Temp",
       x = "Temp",
       y = "'Feels like' Temp")

# How many days of each type of weather are there?

weather_counts <- ggplot(data = bike) +
  geom_bar(mapping = aes(weather, fill = weather)) +
  labs(title = "Counts of types of Weather",
       x = "Weather Type")

# Boxplot of bike counts based on weather type

weather_boxplot <- ggplot(data = bike) +
  geom_boxplot(mapping = aes(x = weather, y = count, fill = weather)) +
  labs(title = "Boxplot of Counts based on Weather")

# temp vs counts point and smoother

temp_counts <- ggplot(bike, mapping = aes(x = temp, y = count)) +
  geom_point() +
  geom_smooth() + 
  labs(title = "Number of bikes based on temperature",
       x = "Temperature",
       y = "Count")

(temp_counts + temp_graph) / (weather_boxplot + weather_counts)




