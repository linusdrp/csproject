import numpy as np
import pandas as pd

# -----------------------------------------
# CONFIG
# -----------------------------------------
N = 10000  # number of synthetic samples
np.random.seed(42)

# -----------------------------------------
# FEATURE GENERATION
# -----------------------------------------

# Number of connectors (1–12 typical)
num_connectors = np.random.choice(
    [1, 2, 4, 6, 8, 12],
    size=N,
    p=[0.15, 0.35, 0.25, 0.15, 0.07, 0.03]
)

# Charger power in kW (AC 11–22, DC 50, 150, 350)
power_kw = np.random.choice(
    [11, 22, 50, 150, 350],
    size=N,
    p=[0.2, 0.2, 0.25, 0.25, 0.1]
)

# Charger type (AC=0, DC=1)
charger_type = (power_kw >= 50).astype(int)

# Time of day (0–23)
time_of_day = np.random.randint(0, 24, size=N)

# Day of week (0=Monday, 6=Sunday)
day_of_week = np.random.randint(0, 7, size=N)

# Traffic factor (1.0 = normal, >1.0 heavy traffic)
traffic_factor = np.random.normal(loc=1.15, scale=0.25, size=N).clip(0.7, 2.0)

# Population density near charger (0 = rural, 1 = urban)
population_density = np.random.beta(2, 5, size=N)

# Popularity score (1–5)
popularity = np.random.randint(1, 6, size=N)

# Historical occupancy rate (0–1)
occupancy = np.random.uniform(0, 1, size=N)

# Weather (0 = dry, 1 = rain)
weather = np.random.choice([0, 1], size=N, p=[0.7, 0.3])

# -----------------------------------------
# WAITING TIME GENERATION (AC vs DC)
# -----------------------------------------

base_time = np.where(power_kw >= 50, 1.5, 3.0)   # DC: low base, AC: higher base

waiting_time = (
    base_time
    + 3.0 * occupancy                             
    + 1.5 * population_density                    
    + 1.0 * ((16 <= time_of_day) & (time_of_day <= 20))  
    + 0.7 * (day_of_week >= 5)                    
    + 0.5 * weather
    + 0.4 * traffic_factor
    - 0.4 * np.log1p(num_connectors)
    - 0.8 * np.log1p(power_kw)
    + np.random.normal(0, 0.4, size=N)
)

# realistic bounds
waiting_time = np.where(power_kw >= 50,
                        waiting_time.clip(1, 4),      # HPC waits
                        waiting_time.clip(3, 7))      # AC waits

# -----------------------------------------
# BUILD DATAFRAME
# -----------------------------------------

df = pd.DataFrame({
    "num_connectors": num_connectors,
    "power_kw": power_kw,
    "charger_type": charger_type,
    "time_of_day": time_of_day,
    "day_of_week": day_of_week,
    "traffic_factor": traffic_factor,
    "population_density": population_density,
    "popularity": popularity,
    "occupancy_rate": occupancy,
    "weather_rain": weather,
    "waiting_time_minutes": waiting_time
})

# -----------------------------------------
# SAVE
# -----------------------------------------

df.to_csv("synthetic_waiting_data.csv", index=False)

print("Dataset generated: synthetic_waiting_data.csv")
print(df.head())
