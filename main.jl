# ========================
# main.jl
# ========================

using Pkg
# Activate the environment in the directory where Project.toml and Manifest.toml are located

cd("E:/TU Delft/Thesis/Code")
Pkg.activate(".")  # Activate the project environment

# Precompile (Optional, depending on your needs)
# Pkg.precompile()

# Load required packages
using CSV, XLSX, DataFrames, YAML, Printf, JuMP, Gurobi, Cbc, DelimitedFiles, Profile

# Define model preferences
a = Model(Gurobi.Optimizer)

# Adjust loops to preferences


a.ext[:loops] = Dict()
read_data = a.ext[:loops][:read_data] = "Reference_Case"  # Change this for different test cases
select_model = a.ext[:loops][:select_model] = "GSK_AHC"
redispatch = a.ext[:loops][:redispatch] = "yes"
cap_calc = a.ext[:loops][:cap_calc] = "yes"
hydrogen = a.ext[:loops][:hydrogen] = "yes"

# Set up base folders for results and input data


include("setup_paths.jl")  # This sets `base_folder_data` and `base_folder_results` based on the chosen model

# Load the data
include("data.jl")  # This assumes the file 'data.jl' handles loading your input data from `base_folder_data`

# Include optimization files based on hydrogen
if hydrogen == "yes"
    include("function_zonalclearing_GSK_AHC_H2.jl")
    include("redispatch_H2.jl")
else
    include("function_zonalclearing_GSK_AHC.jl")
    include("redispatch.jl")
end


# Convert to DataFrame
df = DataFrame(lambda_opt = initial_λ_opt)

# Define the file path
file_path = "E:/TU Delft/Thesis/Code/Results/initial_lambda_opt.csv"

# Save as CSV file
CSV.write(file_path, df)
println("File saved at: $file_path")
dispatch = JuMP.value.(a.ext[:variables][:v])[[83,84,88],:]
curtailment = Array(value.(a.ext[:variables][:curt])[[119,120,124],:])

# Include capacity calculation if enabled
if cap_calc == "yes"

    include("max_net_position.jl")
end

# Process the results
include("process_results.jl")

# Include optional model functions if required
# include("function_zonalclearing_exact_FBonly.jl")
# include("function_zonalclearing_GSK_FBonly.jl")
# include("function_nodalclearing.jl")
# include("function_zonalclearing_GSK_SHC.jl")

owf_production = RES_prod[:,[98,99,103]]
owf_curt = permutedims(Matrix(curt_DA[[119,120,124],:]))
net_owf_production = owf_production .- owf_curt

dispatch_ow = v_DA[[83],:]

node = 3
owf_capacity = owf_production[:,node]
owf_production = net_owf_production[:,node]
   
plot(T, owf_capacity, label="Initial Volume", color=:blue, xlabel="Time Step", ylabel="Volume", title="Volume Comparison for Zone")
plot!(T, owf_production, label="Final Volume", color=:red, linestyle=:dash)