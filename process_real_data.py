import pandas as pd

print("Reading real energy dataset...")

# Read only the columns we need
data = pd.read_csv(
    "household_power_consumption.txt",
    sep=";",
    usecols=["Date", "Time", "Global_active_power"],
    na_values=["?"],
    low_memory=False
)

print("Dataset loaded successfully!")
print("Total records:", len(data))


# Combine Date and Time
data["datetime"] = pd.to_datetime(
    data["Date"] + " " + data["Time"],
    dayfirst=True
)


# Convert energy column to numbers
data["Global_active_power"] = pd.to_numeric(
    data["Global_active_power"],
    errors="coerce"
)


# Remove missing values
data = data.dropna(
    subset=["datetime", "Global_active_power"]
)


# Sort by datetime
data = data.sort_values(
    "datetime"
)


# Set datetime as index
data = data.set_index(
    "datetime"
)


# Convert minute-level power data
# into hourly energy consumption (kWh)
hourly_data = (
    data["Global_active_power"]
    .resample("1h")
    .sum()
    / 60
)


# Convert to DataFrame
energy_data = hourly_data.reset_index()


# Rename columns
energy_data.columns = [
    "datetime",
    "energy_consumption"
]


# Round values
energy_data["energy_consumption"] = (
    energy_data["energy_consumption"]
    .round(3)
)


# Remove empty rows
energy_data = energy_data.dropna()


# Save new dataset
energy_data.to_csv(
    "energy_data.csv",
    index=False
)


print()
print("===================================")
print("REAL ENERGY DATA CREATED!")
print("===================================")

print("Total hourly records:",
      len(energy_data))

print("Start:",
      energy_data["datetime"].min())

print("End:",
      energy_data["datetime"].max())

print()
print("Saved as:")
print("energy_data.csv")