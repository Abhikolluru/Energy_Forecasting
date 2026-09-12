import pandas as pd
import numpy as np

# Number of years
years = 4

# Create hourly dates for 4 years
dates = pd.date_range(
    start="2023-01-01",
    end="2026-12-31 23:00:00",
    freq="h"
)

# Random seed
np.random.seed(42)

energy = []

for date in dates:

    hour = date.hour
    day = date.dayofweek

    # Base consumption
    value = 3.0

    # Morning usage
    if 6 <= hour <= 9:
        value += 2.0

    # Afternoon usage
    if 10 <= hour <= 17:
        value += 2.5

    # Evening peak
    if 18 <= hour <= 22:
        value += 4.0

    # Weekend usage
    if day >= 5:
        value += 0.5

    # Add yearly variation
    value += (date.year - 2023) * 0.3

    # Random variation
    value += np.random.normal(0, 0.4)

    # Consumption cannot be negative
    value = max(value, 0)

    energy.append(round(value, 2))


# Create DataFrame
data = pd.DataFrame({
    "datetime": dates,
    "energy_consumption": energy
})


# Save CSV
data.to_csv(
    "energy_data.csv",
    index=False
)

print("Energy dataset created successfully!")
print("Total records:", len(data))
print("Years: 2023 - 2026")
print("Dataset saved as energy_data.csv")