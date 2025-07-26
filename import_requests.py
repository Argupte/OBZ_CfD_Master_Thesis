import xarray as xr
import pandas as pd
import numpy as np
import cfgrib

print("cfgrib and xarray are installed correctly!")

# Load GRIB file
grib_file = "E:/TU Delft/Thesis/OBZ_CfD/data.grib"
ds = xr.open_dataset(grib_file, engine="cfgrib")

# Print latitude and longitude values
print("Available Latitude:", ds["latitude"].values)
print("Available Longitudes:", ds["longitude"].values)

# Select the closest latitude and longitude
target_lat = 54.49
target_lon = 5.41

# Extract wind components at the specified location for all timesteps
u10_data = ds["u10"].sel(latitude=target_lat, longitude=target_lon, method="nearest")
v10_data = ds["v10"].sel(latitude=target_lat, longitude=target_lon, method="nearest")

# Create a DataFrame
df = pd.DataFrame({
    "Time": ds["time"].values,
    "U10": u10_data.values,
    "V10": v10_data.values
})

# Save to CSV
df.to_csv("wind_data.csv", index=False)

print("Wind speed data saved to 'wind_data.csv'")
