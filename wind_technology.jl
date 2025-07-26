using Pkg
Pkg.add("Optim")
Pkg.add("Interpolations")
Pkg.add("Dates")
Pkg.add("Dierckx")
using CSV, DataFrames, Statistics, Optim, Plots, Interpolations, Dates, Dierckx
using Interpolations: Flat
using Random

#------ Wind speed calculation ------#
# Read the CSV file into a DataFrame
df = CSV.read("wind_data.csv", DataFrame)

# Rename columns for clarity
rename!(df, [:Time, :U10, :V10]) 

# Replace Time with a range from 1 to 8760 (non-leap year 2024)
df.Time = 1:size(df, 1)

# Create a DateTime column assuming data starts from Jan 1st, 2024
start_date = DateTime(2024, 1, 1, 0, 0)
df.DateTime = [start_date + Hour(t-1) for t in df.Time]

# Extract Month and Hour
df.Month = month.(df.DateTime)
df.Hour = hour.(df.DateTime)

# Compute wind speed
df.WindSpeed = sqrt.(df.U10 .^ 2 .+ df.V10 .^ 2)

# Compute wind direction in degrees
df.WindDirection = atan.(-df.U10, -df.V10) .* (180 / π) .+ 180

# Define objective function: negative total wind exposure (since Optim.jl minimizes by default)
function total_wind_exposure(theta)
    return -sum(df.WindSpeed .* cosd.(df.WindDirection .- theta))
end

# Find the optimal orientation using optimization (bounded between 0° and 360°)
result = optimize(total_wind_exposure, 0, 360)

# Extract the optimal orientation
optimal_orientation = Optim.minimizer(result)

println("Optimal wind farm orientation: ", optimal_orientation, "°")

# Define wind farm orientation (example: 60°)
farm_orientation = 60

# Compute wind exposure based on wind direction
df.WindExposure = df.WindSpeed .* cosd.(df.WindDirection .- farm_orientation)

# Define parameters
z0 = 0.0002  # Surface roughness length for offshore
alpha = 0.14 # Wind shear exponent for offshore

# Compute wind speed at 60m using Logarithmic Law
df.WindSpeed_60m = df.WindSpeed .* log(60 / z0) ./ log(10 / z0)

#------ Compress Data to 744 Timesteps (Monthly Hourly Averages) ------#
orig_time = 1:size(df, 1)  # 8760 original hours
new_time = range(1, stop=8760, length=744)  # New 744-hour scale

# Create Akima spline interpolation using Dierckx
spline = Spline1D(orig_time, df.WindSpeed_60m, k=3, bc="nearest")  # Use 60m wind speed

# Apply interpolation to get 744-point curve
wind_speed_744 = spline(new_time)

# Create new DataFrame
df_compressed = DataFrame(Time=1:744, WindSpeed_60m=wind_speed_744)


#------ Wind farm power output calculation ------#

# --- Turbine dictionary ---- #

turbines = Dict(
    "Vestas 164 8.0 MW" => (140.0, 164.0, 8.0, DataFrame(
        WindSpeed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                     21, 22, 23, 24, 25],  
        PowerOutput = [0.0, 0.0, 0.0, 0.1, 0.650, 1.150, 1.850, 2.900, 
                       4.150, 5.600, 7.100, 7.800, 8.00, 8.00, 8.00, 8.00, 
                       8.00, 8.00, 8.00, 8.00, 8.00, 8.00, 8.00, 8.00, 8.00]  # Power in MW
    )),

    "Siemens SWT 3.6 130" => (165.0, 130.0, 3.6, DataFrame(
        WindSpeed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                     21, 22, 23, 24, 25],  
        PowerOutput = [0.0, 0.0, 0.043, 0.184, 0.421, 0.94, 1.459, 1.978, 
                       2.497, 3.016, 3.535, 3.593, 3.60, 3.60, 3.60, 3.60, 
                       3.60, 3.60, 3.60, 3.60, 3.60, 3.60, 3.60, 3.60, 3.60]  # Power in MW
    )),

    "Vestas V80 - 180" => (80.0, 80.0, 1.8, DataFrame(
        WindSpeed = [1, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 
                     7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 
                     12.5, 13, 13.5, 14, 14.5, 15, 16, 17, 18, 19, 
                     20, 21, 22, 23, 24, 25],  # Wind speeds
        PowerOutput = [0.0, 0.0, 0.0, 0.0114, 0.06022, 0.109, 0.181, 0.253, 
                       0.344, 0.443, 0.5625, 0.682, 0.825, 0.968, 1.13391, 1.286, 
                       1.434, 1.582, 1.665, 1.748, 1.77148, 1.793, 1.796, 1.799, 
                       1.79952, 1.800, 1.800, 1.800, 1.800, 1.800, 
                       1.800, 1.800, 1.800, 1.800, 1.800, 1.800]  # Power in MW
    )),

    "Vestas V90" => (105.0, 90.0, 2.03, DataFrame(
        WindSpeed = [1, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 
                     7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 
                     12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 
                     17.5, 18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22, 
                     22.5, 23, 23.5, 24, 24.5, 25],  # 46 Wind speeds
        PowerOutput = [0.0, 0.0, 0.0, 0.075, 0.128, 0.190, 0.265, 0.354, 
                       0.459, 0.582, 0.723, 0.883, 1.058, 1.240, 1.427, 1.604, 
                       1.762, 1.893, 1.968, 2.005, 2.021, 2.027, 2.029, 2.030, 
                       2.030, 2.030, 2.030, 2.030, 2.030, 2.030, 
                       2.030, 2.030, 2.030, 2.030, 2.030, 2.030,
                       2.030, 2.030, 2.030, 2.030, 2.030, 2.030, 2.030, 2.030, 2.030, 2.030]  # Power in MW
    )),

    "Vestas V112 - 3.075 MW" => (119.0, 112.0, 3.075, DataFrame(
        WindSpeed = [1, 2, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 10.5, 
                     11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 
                     16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5, 
                     21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5, 25],  
        PowerOutput = [0.0, 0.0, 0.023, 0.068, 0.130, 0.206, 0.301, 0.757, 
                       1.213, 1.669, 2.125, 2.581, 2.809, 2.988, 3.017, 3.046, 
                       3.055, 3.065, 3.069, 3.073, 3.074, 3.075, 3.075, 3.075, 
                       3.075, 3.075, 3.075, 3.075, 3.075, 3.075, 3.075, 3.075, 
                       3.075, 3.075, 3.075, 3.075, 3.075, 3.075, 3.075, 3.075, 
                       3.075, 3.075] # Fixed length to match WindSpeed
    ))
)


# Select turbine type (Change as needed)
selected_turbine = "Vestas V112 - 3.075 MW"

# Check if selected turbine exists in dictionary
if haskey(turbines, selected_turbine)
    hub_height, rotor_diameter, rated_power, power_curve = turbines[selected_turbine]

    # Compute wind speed at hub height using Power Law (on compressed data)
    df_compressed.WindSpeed_Hub = df_compressed.WindSpeed_60m .* (hub_height / 60) .^ alpha

    # Interpolate power curve with flat extrapolation
    itp = interpolate((power_curve.WindSpeed,), power_curve.PowerOutput, Gridded(Linear()))
    ext_itp = extrapolate(itp, Flat())

    # Compute power output per turbine
    df_compressed.PowerPerTurbine = [ext_itp(ws) for ws in df_compressed.WindSpeed_Hub]

    # Scale up to wind farm level (assuming farm capacity of 1500 MW)
    wind_farm_capacity = 1500.0
    df_compressed.WindFarmPower = (df_compressed.PowerPerTurbine ./ rated_power) #.* wind_farm_capacity

    # Introduce random loss fraction (10% to 30%) for each timestep
    n_timesteps = size(df_compressed, 1)  # 744 timesteps
    Random.seed!(1234)  # Optional: Set seed for reproducibility
    loss_fractions = 0.05 .+ rand(n_timesteps) .* 0.2  # Random values between 0.1 and 0.3 (10% to 30%)

    # Apply random loss to wind farm power
    df_compressed.WindFarmPower_WithLoss = df_compressed.WindFarmPower .* (1 .- loss_fractions)

    # Save the results
    CSV.write("owf_vestas_v112.csv", df_compressed)

    # Plot wind farm power output over time (with random losses)
    p = plot(df_compressed.Time, df_compressed.WindFarmPower_WithLoss, 
             xlabel="Time (hours)", ylabel="Power Output (MW)",
             title="Wind Farm Power Output Over Time (744 Timesteps, 10-30% Random Loss)", 
             legend=false)

    println("Wind farm power output with random losses computed and saved as 'owf_vestas_v112.csv'.")
    display(p)
else
    error("Turbine type not found in dictionary: $selected_turbine")
end