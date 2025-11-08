## Load libraries
library(tidyverse)
library(readxl)
library(sf)
library(dplyr)
library(zoo)
library(data.table)
library(lubridate)

## Load Banana dataset
Bananas <- read_excel("Bananas.xls")

## Load Climate mean dataset
Climate <- read_excel("Climate.xlsx")

## Clean banana data
na.omit(Bananas) |>
  mutate(areaKM = as.double(`Area (ha)`) * 0.01, 
    prod = as.double(Production), 
    region = Region, 
    year = as.double(Year), 
    id = ID)|>
  select(id, region, year, areaKM, prod) |>
  filter(year >= 2001) |>
  filter(year <= 2022) |>
  na.omit() -> brazilBAN

## Clean climate data
na.omit(Climate) |>
  mutate(
    region = name
  ) |>
  select(id, region, year, climate_Mean, climate_Max, climate_Min, climate_Humidity)-> brazilCLI

## Graph initial banana production
ggplot(brazilBAN, aes(x = year, y = prod, color = region)) +
  geom_point() +
  labs(title = "Brazil Regional Banana Production",
       x = "Year",
       y = "Banana Production") +
  theme_minimal()

## Graph initial climate 
ggplot(brazilCLI, aes(x = year, y = climate_Mean, color = region)) +
  geom_point() +
  labs(title = "Brazil Climate",
       x = "Year",
       y = "Climate") +
  theme_minimal()

## Add in diurnal range and approximate growing degree days 
mutate(
  brazilCLI,
  climate_Range = climate_Max - climate_Min, 
  climate_GDD = pmax(climate_Mean - 14, 0)
) -> brazilCLI

## Make master data table
brazilBAN |>
  left_join(brazilCLI, by = c("id", "region", "year")) -> brazilALL

## Plot banana production trends
ggplot(brazilALL, aes(year, prod, color = region)) + 
  geom_point() + 
  geom_line() +
  labs(
    title = "Brazil Banana Production (2001-2022)",
    x = "Year", 
    y = "Production (tons)"
  ) + 
  theme_minimal()

## Plot temperature trends
ggplot(brazilALL, aes(year, climate_Mean, color = region)) + 
  geom_point() + 
  geom_line() + 
  labs(
    title = "Mean Annual Temperature by Region", 
    x = "Year", 
    y = "Temperature (Degree Celcius)"
  ) + 
  theme_minimal()

## Aggregated data for all variables 
brazilTOTAL <- brazilALL |>
  group_by(year) |>
  summarise(
    total_prod = sum(prod, na.rm = TRUE),
    total_areaKM = sum(areaKM, na.rm = TRUE),
    total_mean = sum(climate_Mean * areaKM, na.rm = TRUE) / sum(areaKM, na.rm = TRUE), 
    total_max = sum(climate_Max * areaKM, na.rm = TRUE) / sum(areaKM, na.rm = TRUE), 
    total_min = sum(climate_Min * areaKM, na.rm = TRUE) / sum(areaKM, na.rm = TRUE), 
    total_humidity = sum(climate_Humidity * areaKM, na.rm = TRUE) / sum(areaKM, na.rm = TRUE), 
    total_GDD = sum(climate_GDD, na.rm = TRUE), 
    total_range = sum(climate_Range * areaKM, na.rm = TRUE) / sum(areaKM, na.rm = TRUE),
    total_prodperareaKM = sum(prod, na.rm = TRUE) / sum(areaKM, na.rm = TRUE)
  ) |>
  ungroup()

## Plot aggregate banana production trends
ggplot(brazilTOTAL, aes(year, total_prod)) + 
  geom_point() + 
  geom_line() +
  labs(
    title = "Aggregated Brazil Banana Production (2001-2022)",
    x = "Year", 
    y = "Production (tons)"
  ) + 
  theme_minimal()

## Plot aggregate farmed land trends
ggplot(brazilTOTAL, aes(year, total_areaKM)) + 
  geom_point() + 
  geom_line() +
  labs(
    title = "Aggregated Brazil Land Banana Farmed (2001-2022)",
    x = "Year", 
    y = "Land (km^2)"
  ) + 
  theme_minimal()

## Plot aggregate farmed land per banana produced trends
ggplot(brazilTOTAL, aes(year, total_prodperareaKM)) + 
  geom_point() + 
  geom_line() +
  labs(
    title = "Aggregated Brazil Land Banana Farmed Per Banana Production (2001-2022)",
    x = "Year", 
    y = "Banana per Land (tons per km^2)"
  ) + 
  theme_minimal()

## Plot aggregate mean temperature trends
ggplot(brazilTOTAL, aes(year, total_mean)) + 
  geom_point() + 
  geom_line() + 
  labs(
    title = "Aggregated Mean Annual Temperature", 
    x = "Year", 
    y = "Temperature (Degree Celcius)"
  ) + 
  theme_minimal()

## Plot aggregate max temperature trends
ggplot(brazilTOTAL, aes(year, total_max)) + 
  geom_point() + 
  geom_line() + 
  labs(
    title = "Aggregated Max Annual Temperature", 
    x = "Year", 
    y = "Temperature (Degree Celcius)"
  ) + 
  theme_minimal()

## Plot aggregate min temperature trends
ggplot(brazilTOTAL, aes(year, total_min)) + 
  geom_point() + 
  geom_line() + 
  labs(
    title = "Aggregated Min Annual Temperature", 
    x = "Year", 
    y = "Temperature (Degree Celcius)"
  ) + 
  theme_minimal()

## Plot aggregate temperature trends
ggplot(brazilTOTAL) + 
  geom_point(aes(year, total_mean)) + 
  geom_line(aes(year, total_mean)) +
  geom_point(aes(year, total_min)) + 
  geom_line(aes(year, total_min)) +
  geom_point(aes(year, total_max)) + 
  geom_line(aes(year, total_max)) +
  labs(
    title = "Aggregated Brazil Temperature Trends (2001-2022)",
    x = "Year", 
    y = "Temperature (Degree Celcius)"
  ) + 
  theme_minimal()

## Plot aggregate humidity trends
ggplot(brazilTOTAL, aes(year, total_humidity)) + 
  geom_point() + 
  geom_line() + 
  labs(
    title = "Aggregated Humidity", 
    x = "Year", 
    y = "Humidity"
  ) + 
  theme_minimal()

## Plot aggregate GDD trends
ggplot(brazilTOTAL, aes(year, total_GDD)) + 
  geom_point() + 
  geom_line() + 
  labs(
    title = "Aggregated GDD", 
    x = "Year", 
    y = "GDD"
  ) + 
  theme_minimal()

## Plot aggregate temperature range trends
ggplot(brazilTOTAL, aes(year, total_range)) + 
  geom_point() + 
  geom_line() + 
  labs(
    title = "Aggregated Annual Temperature Range", 
    x = "Year", 
    y = "Temperature Range (Degree Celcius)"
  ) + 
  theme_minimal()