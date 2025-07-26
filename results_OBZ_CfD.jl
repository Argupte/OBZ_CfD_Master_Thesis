using DataFrames, CSV

# Define scenario number (you can change this easily now)
scenario_num = 5  # Use 1 for case_study, 2 for scenario_2, etc.
sub_scenario_num = 1  # Sub-scenario number (if needed)

# Generate folder name and base directory
scenario_folder = scenario_num == 1 ? "case_study" : "scenario_$(scenario_num)_$(sub_scenario_num)"  #_$(sub_scenario_num)
# Generate folder name and base directory

base_dir = "E:/TU Delft/Thesis/Code/Results/$(scenario_folder)"

# All case directories and file prefixes
case_dict = Dict(
    "c1_one_sided" => ("$base_dir/c1_one_sided", "c1"),
    "c2_two_sided" => ("$base_dir/c2_two_sided", "c2"),
    "c2b_two_sided_yearly_avg" => ("$base_dir/c2_two_sided_yearly", "c2b"),
    "c3_cap_and_floor" => ("$base_dir/c3_cap_floor", "c3"),
    "c4_financial_avg" => ("$base_dir/c4_financial_avg", "c4a"),
    "c4b_financial_avg" => ("$base_dir/c4_financial_avg_dynamic", "c4b"),
    "c5_financial_avg2" => ("$base_dir/c5_financial_avg2", "c5"),
    "c6_financial_avg3" => ("$base_dir/c6_financial_avg3", "c6"),
    "c7_financial_node98" => ("$base_dir/c7_financial_node98", "c7"),
    "c7b_financial_node99" => ("$base_dir/c7b_financial_node99", "c7b"),
    "c7c_financial_node103" => ("$base_dir/c7c_financial_node103", "c7c"),
    "c8_financial_5_2" => ("$base_dir/financial_scenario5_2", "c8"),
    "c9_financial_sp5000" => ("$base_dir/c9_financial_sp5000", "c9"),
    "c10_financial_sp1920"  => ("$base_dir/c10_financial_sp1920", "c10"),
    "c11_financial_based_q3_noCfD" => ("$base_dir/c11_financial_based_q3_noCfD", "c11"),
    "c12_financial_based_q1_noCfD" => ("$base_dir/c12_financial_based_q1_noCfD", "c12"),
    "c13_financial_based_q2_noCfD" => ("$base_dir/c13_financial_based_q2_noCfD", "c13"),
    "c14_financial_based_q1_CfD" => ("$base_dir/c14_financial_based_q1_CfD", "c14"),
    "c15_financial_based_q2_CfD" => ("$base_dir/c15_financial_based_q2_CfD", "c15"),
    "c16_financial_based_q3_CfD" => ("$base_dir/c16_financial_based_q3_CfD", "c16"),
    "c17_financial_sp10000" => ("$base_dir/c17_financial_sp10000", "c17")
)

# Select the case
case_name = "c4_financial_avg"  # Change this to any valid key from case_dict

maxposOBZ = Matrix(CSV.read("E:/TU Delft/Thesis/Code/Results/case_study/p_domain.csv", DataFrame))[4,:]
RES_bc = RES[:,[98,99,103]]
initial_curt_DA = Matrix(CSV.read("E:/TU Delft/Thesis/Code/Results/$(scenario_folder)/bs1_curtailment.csv", DataFrame))

RES_bc_q1 = RES_bc[:,1] - initial_curt_DA[1,:]
RES_bc_q2 = RES_bc[:,2] - initial_curt_DA[2,:]
RES_bc_q3 = RES_bc[:,3] - initial_curt_DA[3,:]

#Average res : one of the reference values
res_avg = zeros(Float64, length(T))
for t in 1:length(T)
    res_avg[t] = (RES_bc_q1[t] + RES_bc_q2[t] + RES_bc_q3[t]) / 3
end

res_batch_avg = zeros(Float64, div(length(T), 30)) # Dividing 744 by 30 gives 24 batches

# Loop through the data in chunks of 30
for i in 1:length(res_batch_avg)
    start_idx = (i - 1) * 30 + 1  # Starting index of the chunk
    end_idx = min(i * 30, length(T))  # Ending index of the chunk
    res_batch_avg[i] = mean(res_avg[start_idx:end_idx])  # Calculate the mean for this batch
end

# Get path and prefix
if haskey(case_dict, case_name)
    dir_path, prefix = case_dict[case_name]
else
    error("Invalid case name: $case_name")
end

# Load data into unified variable names
price         = Matrix(CSV.read("$dir_path/$(prefix)_lambda_opt.csv", DataFrame))
prices        = Matrix(CSV.read("$dir_path/$(prefix)_prices.csv", DataFrame))
curtailment   = Matrix(CSV.read("$dir_path/$(prefix)_curtailment.csv", DataFrame))[1:744, :]
available_power = Matrix(CSV.read("$dir_path/$(prefix)_available_power.csv", DataFrame))[1:744, 1:3]
DC_flow       = Matrix(CSV.read("$dir_path/$(prefix)_DC_flow.csv", DataFrame))
electrolyser  = Matrix(CSV.read("$dir_path/$(prefix)_electrolyser.csv", DataFrame))
position      = Matrix(CSV.read("$dir_path/$(prefix)_position.csv", DataFrame))

#-------------------------------------------------------------------------------------------------------#

mean_available_power = mean(available_power, dims=2)[:]

sum_wf1 = sum(available_power[:, 1])
sum_wf2 = sum(available_power[:, 2])
sum_wf3 = sum(available_power[:, 3])

curtailment_wf1 = sum(curtailment[:, 1])
curtailment_wf2 = sum(curtailment[:, 2])
curtailment_wf3 = sum(curtailment[:, 3])

#--------------------------------------------------#

# Create folder to save results
results_dir = joinpath(base_dir, "results_summary_$(case_name)")
isdir(results_dir) || mkpath(results_dir)



results_dict = Dict{String, Dict{String, Any}}()

time_steps = 1:744
nodes = 1:3

revenue_wp = zeros(744)
total_revenue_nocfd = zeros(3)

#for n in nodes
#    for t in time_steps
#    revenue_wp[t] = bs1_available_power[n, t] .* (initial_λ_opt[t]) #revenue without CfD
#    total_revenue_nocfd[n] += revenue_wp[t]
#    end
#end
total_revenue_nocfd


export_zero = zeros(744, 3)  # Store exported power at each timestep for each node
zero_price_hours = 0  # Initialize counter for zero price hours

for t in 1:744  # Loop over time steps
    if price[t] <= 1e-3  # Check if price is zero
        zero_price_hours += 1  # Increment counter
        for n in 1:3  # Loop over nodes
            export_zero[t, n] = available_power[t, n]  # Store exported power
        end
    end
end
zero_price_hours  # Total number of hours with zero price
export_zero

no_non_zero = zeros(3)  # Initialize vector to store number of hours with export during non zero prices
sum_power_non_zero = zeros(3)  # Initialize vector to store sum of exported power when price is zero
for t in time_steps
    for n in nodes  # Loop over nodes
        if export_zero[t, n] != 0  # Check if exported power is not zero
            no_non_zero[n] += 1 
            sum_power_non_zero[n] += export_zero[t, n]  # Sum exported power
        end # Count number of non-zero prices
    end
end
no_non_zero #hours
sum_power_non_zero #MWh

# another code for noting congestion when the price is 0.

# Price risks#
# flat price risk

function compute_price_risk_1(prices::Matrix{Float64}, OBZ_zone::Int, base_folder_path::String)
    # Intra-zone metrics
    mean_price = mean(prices, dims=2)[:]
    std_dev_price = std(prices, dims=2)[:]
    coefficient_of_variation = std_dev_price ./ mean_price

    # Format Zone labels
    zone_labels = [i == OBZ_zone ? "OBZ" : string(i) for i in 1:size(prices, 1)]

    intra_z_df = DataFrame(
    "Zone" => zone_labels,
    "Average Price" => mean_price,
    "Std Deviation" => std_dev_price,
    "Coefficient of Variation" => coefficient_of_variation
    )

    # Save intra-zone result
    intra_z_path = base_folder_path * "/pr1_intra_z_indicator_pr1.csv"
    CSV.write(intra_z_path, intra_z_df)

    # Inter-zone metrics
    inter_zone_mean = mean(mean_price)
    inter_zone_std = std(mean_price)
    inter_zone_cov = inter_zone_std / inter_zone_mean

    inter_z_df = DataFrame(
        Metric = ["Inter-Zone Mean Price", "Inter-Zone Std Deviation", "Inter-Zone Coefficient of Variation"],
        Value = [inter_zone_mean, inter_zone_std, inter_zone_cov]
    )

    inter_z_path = base_folder_path * "/pr1_inter_z_indicator.csv"
    CSV.write(inter_z_path, inter_z_df)

    return intra_z_df, inter_z_df
end

OBZ_zone = 4  # Zone 4 is OBZ
intra, inter = compute_price_risk_1(prices, OBZ_zone, base_dir)

count_low_price = count(x -> x < 1e-4, price)
println("Total count of near-zero prices in the matrix: ", count_low_price)

function compute_price_risk_2(prices::Matrix{Float64}, available_power::Matrix{Float64},
    OWF_location::Vector{Int}, OBZ_zone::Int, WTP::Float64,
    base_folder_path::String)

    # Init counters
    positive_hours = 0
    negative_hours = 0
    positive_mwh = 0.0
    negative_mwh = 0.0
    positive_effect = 0.0
    negative_effect = 0.0

    for t in 1:size(prices, 2)  # Loop over hours
        price_obz = round(prices[OBZ_zone, t]; digits=4)
        other_prices = round.(prices[setdiff(1:end, OBZ_zone), t]; digits=4)

        # Skip hydrogen WTP hours
        if price_obz == WTP
            continue
        end

        # Only process if OBZ price ≠ other zones AND not zero
        if all(p -> p != price_obz, other_prices) && abs(price_obz) > 1e-6
            net_production = sum(available_power[t, owf] for owf in OWF_location)

            if abs(price_obz) > minimum(other_prices)
                positive_hours += 1
                positive_mwh += net_production
                positive_effect += net_production * abs(price_obz)
            elseif abs(price_obz) < minimum(other_prices)
                negative_hours += 1
                negative_mwh += net_production
                negative_effect += net_production * abs(price_obz)
            end
        end
    end

    net_effect = positive_effect - negative_effect

    # Build result DataFrame
    results_df = DataFrame(
        "Metrics" => [
            "Number of Positive Hours",
            "Number of Negative Hours",
            "Positive Hours [MWh]",
            "Negative Hours [MWh]",
            "Positive Effect [€]",
            "Negative Effect [€]",
            "Net Effect [€]"
        ],
        "Values" => [
            positive_hours,
            negative_hours,
            positive_mwh,
            negative_mwh,
            positive_effect,
            negative_effect,
            net_effect
        ]
    )

    # Save to CSV
    CSV.write(base_folder_path * "/pr2_results.csv", results_df)

    return results_df
end

OBZ_zone = 4
WTP = 41.7  # Or whatever value marks hydrogen willingness-to-pay hours
OWF_location = [1, 2, 3]

compute_price_risk_2(
    prices, available_power, OWF_location,
    OBZ_zone, WTP, base_dir
)

nonobz_price = prices[[1,2,3],:]
obz_price = permutedims(price)

# Results storage
zone4_lowest = Vector{Tuple{Int, Vector{Int}}}()
zone4_middle = Vector{Tuple{Int, Vector{Int}}}()
zone4_highest = Vector{Tuple{Int, Vector{Int}}}()
zone4_different = Int[]

# Iterate over timesteps
for t in T
    # Sort prices while keeping original zone indices
    prices_at_t = nonobz_price[:, t]
    sorted_prices = sort(collect(enumerate(prices_at_t)), by=x -> x[2])  # [(Zone, Price), ...]

    # Extract sorted zones and values
    sorted_zones, sorted_values = first.(sorted_prices), last.(sorted_prices)

    # Find unique prices
    unique_values = unique(sorted_values)
    min_zones = []
    max_zones = []
    middle_zones = []

    # Classify Min, Max, and Middle
    if length(unique_values) == 1
        min_zones = sorted_zones
        max_zones = sorted_zones
    elseif length(unique_values) == 2
        min_zones = [sorted_zones[i] for i in findall(x -> x == unique_values[1], sorted_values)]
        max_zones = [sorted_zones[i] for i in findall(x -> x == unique_values[2], sorted_values)]
    else
        min_zones = [sorted_zones[i] for i in findall(x -> x == unique_values[1], sorted_values)]
        max_zones = [sorted_zones[i] for i in findall(x -> x == unique_values[end], sorted_values)]
        middle_zones = [sorted_zones[i] for i in findall(x -> x == unique_values[2], sorted_values)]
    end

    # Compare with OBZ price
    price_obz = obz_price[t]

    if price_obz in unique_values[1]
        push!(zone4_lowest, (t, min_zones))
    elseif length(unique_values) ≥ 3 && price_obz in unique_values[2]
        push!(zone4_middle, (t, middle_zones))
    elseif price_obz in unique_values[end]
        push!(zone4_highest, (t, max_zones))
    else
        push!(zone4_different, t)
    end
end

numberof = length(zone4_different) 

initial_λ = CSV.read("E:/TU Delft/Thesis/Code/Results/$(scenario_folder)/bs1_lambda.csv", DataFrame)
initial_λ = Matrix(initial_λ)'

price_delta = initial_λ' .- price
counter = 0
for i in 1:744
    if price_delta[i] != 0
        counter += 1
    end
end
counter

function save_zone4_price_cats(filename::String, data::Vector{Tuple{Int, Vector{Int}}})
    df = DataFrame(Timestep = Int[], MatchingZones = String[])
    for (t, zones) in data
        push!(df, (t, join(zones, ",")))
    end
    CSV.write(joinpath(results_dir, filename), df)
end

save_zone4_price_cats("zone4_lowest.csv", zone4_lowest)
save_zone4_price_cats("zone4_middle.csv", zone4_middle)
save_zone4_price_cats("zone4_highest.csv", zone4_highest)

# Save zone4_different
CSV.write(joinpath(results_dir, "zone4_different.csv"), DataFrame(Timestep = zone4_different))

# Save price delta values
CSV.write(joinpath(results_dir, "price_delta.csv"), DataFrame(
    Timestep = 1:length(price_delta[:]),
    PriceDelta = price_delta[:]  # ensure it's a flat Vector
))

# Initialize a dictionary to store congested DC lines indexed by timestamps
congested_dc_lines = Dict{Int, Vector{Tuple{Int, String}}}()  # Dict{Timestep, [(l_dc, Direction)]}

# Extract prices and DC line parameters
zone_prices = prices  # Zonal prices
obz_zone = 4  # Zone 4 (Offshore Bidding Zone)

F_DC_DA = DC_flow  # DC power flows [l_dc, t]
TC_DC = a.ext[:parameters][:TC_DC]  # DC capacities [l_dc]
T = a.ext[:sets][:T]  # Timestamps

# Identify time periods where Zone 4 price is zero
zero_price_times = [t for t in T if abs(zone_prices[obz_zone, t]) < 1e-6]

# Check for congestion in DC lines
for t in zero_price_times
    congested_lines_at_t = []

    for l_dc in eachindex(TC_DC)
        flow = F_DC_DA[l_dc, t]
        capacity = TC_DC[l_dc]

        # Check if line is congested
        if abs(flow) >= capacity
            direction = if flow > 0 "+1" else "-1" end
            push!(congested_lines_at_t, (l_dc, direction))
        end
    end

    # Store results only if congestion was detected at time t
    if !isempty(congested_lines_at_t)
        congested_dc_lines[t] = congested_lines_at_t
    end
end

# Print results indexed by timestamps
println("Congested DC lines by timestamps:")
for (t, lines) in congested_dc_lines
    println("Timestep $t: $lines")
end
congested_dc_lines

# Create a dictionary to map DC line index (row number) to (FromBus, ToBus)
dc_line_mapping = Dict{Int, Tuple{Int, Int}}()

# Iterate over each row and use row index as DC line number
for (i, row) in enumerate(eachrow(df_DC))  # i is the row index (1-based)
    from_bus = row[:FromBus]
    to_bus = row[:ToBus]
    
    dc_line_mapping[i] = (from_bus, to_bus)  # Store mapping
end

# Print stored mappings for verification
println("Stored DC Line Mappings: ", dc_line_mapping)

# Step 1: Count occurrences of each congested DC line
congested_counts = Dict{Int, Int}()  # Dictionary to store {DC Line Index => Count}

# Iterate over congested timesteps and count occurrences of each line
for (_, lines) in congested_dc_lines
    for (l_dc, _) in lines  # Extract line number based on row index
        congested_counts[l_dc] = get(congested_counts, l_dc, 0) + 1
    end
end

# Step 2: Compute congestion percentages
total_timesteps_with_congestion = length(congested_dc_lines)

congestion_percentage = Dict{Int, Float64}()  # Store {DC Line Index => Percentage}

for (l_dc, count) in congested_counts
    congestion_percentage[l_dc] = (count / total_timesteps_with_congestion) * 100
end

# Step 3: Print results including FromBus and ToBus
println("Percentage of time each DC line is congested (with bus info):")
for (l_dc, pct) in sort(collect(congestion_percentage), by=x->x[1])  # Sort by DC line number
    if haskey(dc_line_mapping, l_dc)
        from_bus, to_bus = dc_line_mapping[l_dc]  # Fetch FromBus and ToBus
    else
        from_bus, to_bus = "MISSING", "MISSING"  # Debugging fallback
    end
    println("DC Line $l_dc: From Bus $from_bus → To Bus $to_bus | Congested $(round(pct, digits=2))% of the time")
end

# --- Save congested DC lines by timestamp ---
congested_df = DataFrame(Timestep = Int[], CongestedLines = String[])

for (t, lines) in congested_dc_lines
    # Format each line-direction pair as "Line:Dir"
    line_strs = ["$(l_dc):$dir" for (l_dc, dir) in lines]
    combined_str = join(line_strs, ", ")  # Join into one string

    push!(congested_df, (t, combined_str))
end

# Save to CSV
CSV.write(joinpath(results_dir, "congested_dc_lines.csv"), congested_df)

# --- Save DC line mapping (index → FromBus, ToBus) ---
mapping_df = DataFrame(DC_Line = Int[], FromBus = Int[], ToBus = Int[])
for (i, (from_bus, to_bus)) in dc_line_mapping
    push!(mapping_df, (i, from_bus, to_bus))
end
CSV.write(joinpath(results_dir, "dc_line_mapping.csv"), mapping_df)

# --- Save congestion percentage per DC line ---
congestion_df = DataFrame(DC_Line = Int[], FromBus = String[], ToBus = String[], CongestedPercent = Float64[])
for (l_dc, pct) in sort(collect(congestion_percentage), by=x->x[1])
    from_bus, to_bus = get(dc_line_mapping, l_dc, ("MISSING", "MISSING"))
    push!(congestion_df, (l_dc, string(from_bus), string(to_bus), round(pct, digits=2)))
end
CSV.write(joinpath(results_dir, "dc_congestion_percent.csv"), congestion_df)

# Final results dictionary
results = Dict(
    "revenue_metrics" => Dict(
        "revenue_matrix" => revenue,
        "total_revenue_per_farm" => total_revenue,
        "revenue_mean_per_farm" => [mean(revenue[:, n]) for n in 1:3],
        "revenue_std_per_farm" => [std(revenue[:, n]) for n in 1:3],
        "revenue_var_per_farm" => [var(revenue[:, n]) for n in 1:3],
        "revenue_cov_per_farm" => [std(revenue[:, n]) / mean(revenue[:, n]) for n in 1:3]
    ),

    "export_zero_metrics" => Dict(
        "export_zero_matrix" => export_zero,
        "non_zero_price_hours_per_farm" => no_non_zero,
        "sum_exported_power_when_price_zero" => sum_power_non_zero
    ),

    "price_risk_1" => Dict(
        "intra_zone_dataframe" => intra,
        "inter_zone_dataframe" => inter
    ),

    "price_risk_2" => Dict(
        "results_dataframe" => compute_price_risk_2(
            prices, available_power, OWF_location,
            OBZ_zone, WTP, base_dir
        )
    )
)

CSV.write(joinpath(results_dir, "revenue_stats.csv"), DataFrame(
    Farm = 1:3,
    Mean = results["revenue_metrics"]["revenue_mean_per_farm"],
    Std = results["revenue_metrics"]["revenue_std_per_farm"],
    Variance = results["revenue_metrics"]["revenue_var_per_farm"],
    CoV = results["revenue_metrics"]["revenue_cov_per_farm"]
))

# Save export_zero metrics
CSV.write(joinpath(results_dir, "export_zero_matrix.csv"), DataFrame(export_zero, :auto))
CSV.write(joinpath(results_dir, "non_zero_price_hours.csv"),
    DataFrame(Farm=1:3, NonZeroHours=results["export_zero_metrics"]["non_zero_price_hours_per_farm"]))
CSV.write(joinpath(results_dir, "sum_power_when_price_zero.csv"),
    DataFrame(Farm=1:3, ExportedPower=results["export_zero_metrics"]["sum_exported_power_when_price_zero"]))

# Save price risk 1 (already dataframes)
CSV.write(joinpath(results_dir, "pr1_intra_zone.csv"), results["price_risk_1"]["intra_zone_dataframe"])

CSV.write(joinpath(results_dir, "pr1_inter_zone.csv"), results["price_risk_1"]["inter_zone_dataframe"])

# Save price risk 2
CSV.write(joinpath(results_dir, "pr2_price_congestion_effects.csv"), results["price_risk_2"]["results_dataframe"])

#------ volume effects ------#

potential_wind = RES[:, [98,99,103]]

function compute_volume_risk_metrics(
    topology,
    available_power,
    curtailment,
    potential,
    maxposOBZ,
    prices,
    owf_location
)
    # Derived values
    total_production = sum(available_power, dims=2)[:]
    total_curtailment = sum(curtailment, dims=2)[:]
    total_potential = sum(potential, dims=2)[:]
    price_zone1 = prices[1, :]
    price_zone2 = prices[2, :]
    price_zone3 = prices[3, :]

    # Output containers
    volume_risk_severity = Vector{Dict{String, Any}}()
    volume_risk_frequency = Dict("total" => 0, "Capacity Calculation" => 0, "Capacity Allocation" => 0, "both" => 0)

    # Loop over time
    for t in 1:744
        actual_RES = total_production[t]
        potential_RES = total_potential[t]
        curtail = total_curtailment[t]
        maxpOBZ = maxposOBZ[t]

        if any(curtailment[t, owf] != 0 for owf in owf_location)
            curtailment_per_owf = [curtailment[t, owf] for owf in owf_location]

            if topology == "Reference_Case"
                if maxpOBZ < potential_RES && maxpOBZ >= actual_RES
                    V3 = potential_RES - maxpOBZ
                    V4 = curtail - V3

                    if V3 > 0 && V4 == 0
                        volume_risk_frequency["Capacity Calculation"] += 1
                        volume_risk_frequency["total"] += 1
                        push!(volume_risk_severity, Dict(
                            "Hour" => t,
                            "Capacity Calculation [MWh]" => V3,
                            "Capacity Allocation [MWh]" => 0.0,
                            "Total volume risk [MWh]" => V3,
                            "λ OBZ [€/MWh]" => maxpOBZ,
                            "λ zone 1 [€/MWh]" => price_zone1[t],
                            "λ zone 2 [€/MWh]" => price_zone2[t],
                            "λ zone 3 [€/MWh]" => price_zone3[t],
                            "Curtailment OWF 119 (z1) [MWh]" => curtailment_per_owf[1],
                            "Curtailment OWF 120 (z3) [MWh]" => curtailment_per_owf[2],
                            "Curtailment OWF 124 (z2) [MWh]" => curtailment_per_owf[3]
                        ))
                    elseif V3 > 0 && V4 > 0
                        volume_risk_frequency["both"] += 1
                        volume_risk_frequency["Capacity Calculation"] += 1
                        volume_risk_frequency["Capacity Allocation"] += 1
                        volume_risk_frequency["total"] += 1
                        push!(volume_risk_severity, Dict(
                            "Hour" => t,
                            "Capacity Calculation [MWh]" => V3,
                            "Capacity Allocation [MWh]" => V4,
                            "Total volume risk [MWh]" => V3 + V4,
                            "λ OBZ [€/MWh]" => maxpOBZ,
                            "λ zone 1 [€/MWh]" => price_zone1[t],
                            "λ zone 2 [€/MWh]" => price_zone2[t],
                            "λ zone 3 [€/MWh]" => price_zone3[t],
                            "Curtailment OWF 119 (z1) [MWh]" => curtailment_per_owf[1],
                            "Curtailment OWF 120 (z3) [MWh]" => curtailment_per_owf[2],
                            "Curtailment OWF 124 (z2) [MWh]" => curtailment_per_owf[3]
                        ))
                    end
                elseif maxpOBZ >= potential_RES && maxpOBZ >= actual_RES
                    V4 = curtail
                    volume_risk_frequency["Capacity Allocation"] += 1
                    volume_risk_frequency["total"] += 1
                    push!(volume_risk_severity, Dict(
                        "Hour" => t,
                        "Capacity Calculation [MWh]" => 0.0,
                        "Capacity Allocation [MWh]" => V4,
                        "Total volume risk [MWh]" => V4,
                        "λ OBZ [€/MWh]" => maxpOBZ,
                        "λ zone 1 [€/MWh]" => price_zone1[t],
                        "λ zone 2 [€/MWh]" => price_zone2[t],
                        "λ zone 3 [€/MWh]" => price_zone3[t],
                        "Curtailment OWF 119 (z1) [MWh]" => curtailment_per_owf[1],
                        "Curtailment OWF 120 (z3) [MWh]" => curtailment_per_owf[2],
                        "Curtailment OWF 124 (z2) [MWh]" => curtailment_per_owf[3]
                    ))
                end
            end
        end
    end

    return volume_risk_frequency, volume_risk_severity
end

OWF_location = [1, 2, 3]  # Your wind farm indices

# For Case 1
volume_risk_frequency, volume_risk_severity = compute_volume_risk_metrics(
    "Reference_Case",
    available_power,
    curtailment,
    potential_wind,
    maxposOBZ,
    prices,
    OWF_location
)

# Convert volume_risk_frequency dictionary to DataFrame and save
vrf_df = DataFrame(
    Category = keys(volume_risk_frequency),
    Count = values(volume_risk_frequency)
)
CSV.write(joinpath(results_dir, "volume_risk_frequency.csv"), vrf_df)

# Convert vector of dictionaries (volume_risk_severity) to DataFrame and save
vrs_df = DataFrame(volume_risk_severity)
CSV.write(joinpath(results_dir, "volume_risk_severity.csv"), vrs_df)

total_revenue = zeros(3)
revenue  = zeros(744,3)
SP = 8.0
Fix_SP = 1920.0
floor_price = 6.0
cap_price = 20.0

res_avg
avg_p = mean(prices, dims=2)[4]
clawback = zeros(744,3)

if case_name == "c1_one_sided"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            revenue[t, farm] = power * max(λ, SP)  # Revenue for each farm
            total_revenue[farm] += power * max(λ, SP)
        end
    end

elseif case_name == "c2_two_sided"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            revenue[t, farm] = power * λ + power * (SP - λ)  # Revenue for each farm
            total_revenue[farm] += power * λ + power * (SP - λ)
            clawback[t, farm] = power * (max(λ - SP, 0))  # Clawback for each farm
        end
    end

elseif case_name == "c2b_two_sided_yearly_avg"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            revenue[t, farm] = power * avg_p + power * (SP - avg_p)  # Revenue for each farm
            total_revenue[farm] += power * avg_p + power * (SP - avg_p)
            clawback[t, farm] = power * (max(SP - avg_p, 0))  # Clawback for each farm
        end
    end

elseif case_name == "c3_cap_and_floor"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            remuneration = max(floor_price - λ, 0)
            cb = max(λ - cap_price, 0)
            clawback[t,farm] = power * (max(λ - cap_price, 0))
            revenue[t, farm] = power * (λ + remuneration - cb)
            total_revenue[farm] += power * (λ + remuneration - cb)
        end
    end

elseif case_name == "c4_financial_avg"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            batch_cost = res_avg[t]
            clawback[t,farm] = batch_cost * λ
            revenue[t,farm] = (power - batch_cost) * λ + Fix_SP
            total_revenue[farm] += (power - batch_cost) * λ + Fix_SP
        end
    end

elseif case_name == "c4b_financial_avg"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            batch_cost = mean_available_power[t]
            clawback[t,farm] = batch_cost * λ
            revenue[t,farm] = (power - batch_cost) * λ + Fix_SP
            total_revenue[farm] += (power - batch_cost) * λ + Fix_SP
        end
    end

elseif case_name == "c5_financial_avg2"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            δ_t = (power > 0 && λ > 0) ? 1 : 0
            batch_cost = res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))]
            clawback[t,farm] = batch_cost * λ
            revenue[t, farm] = (power - batch_cost) * λ + δ_t * Fix_SP
            total_revenue[farm] += revenue[t, farm]
        end
    end

elseif case_name == "c6_financial_avg3"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            clawback[t,farm] = mean_available_power[t] * λ
            revenue[t,farm] = (power - res_avg[t]) * λ + Fix_SP
            total_revenue[farm] += (power - res_avg[t]) * λ + Fix_SP
        end
    end

elseif case_name == "c7_financial_node98"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            revenue[t,farm] = (power - available_power[t,1]) * λ + Fix_SP
            clawback[t,farm] = available_power[t,1] * λ
            total_revenue[farm] += (power - available_power[t,1]) * λ + Fix_SP
        end
    end

elseif case_name == "c7b_financial_node99"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            revenue[t,farm] = (power - available_power[t,2]) * λ + Fix_SP
            clawback[t,farm] = available_power[t,2] * λ
            total_revenue[farm] += (power - available_power[t,2]) * λ + Fix_SP
        end
    end

elseif case_name == "c7c_financial_node103"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            revenue[t,farm] = (power - available_power[t,3]) * λ + Fix_SP
            clawback[t,farm] = available_power[t,3] * λ
            total_revenue[farm] += (power - available_power[t,3]) * λ + Fix_SP
        end
    end

elseif case_name == "c8_financial_5_2"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            batch_cost = mean_available_power[t]
            clawback[t,farm] = batch_cost * λ
            revenue[t,farm] = (power - batch_cost) * λ + Fix_SP
            total_revenue[farm] += (power - batch_cost) * λ + Fix_SP
        end
    end

elseif case_name == "c9_financial_sp5000"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            batch_cost = res_avg[t]
            clawback[t,farm] = batch_cost * λ
            revenue[t,farm] = (power - batch_cost) * λ + Fix_SP
            total_revenue[farm] += (power - batch_cost) * λ + Fix_SP
        end
    end

elseif case_name == "c10_financial_sp1920"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            batch_cost = res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))]
            clawback[t,farm] = batch_cost * λ
            revenue[t,farm] = (power - batch_cost) * λ + Fix_SP
            total_revenue[farm] += (power - batch_cost) * λ + Fix_SP
        end
    end

elseif case_name == "c11_financial_based_q3_noCfD"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            batch_cost = res_avg[t]

            if farm == 3
                clawback[t, farm] = 0.0  # assuming no clawback for Q3
                revenue[t, farm] = power * λ
                total_revenue[farm] += power * λ
            else
                clawback[t, farm] = batch_cost * λ
                revenue[t, farm] = (power - batch_cost) * λ + Fix_SP
                total_revenue[farm] += (power - batch_cost) * λ + Fix_SP
            end
        end
    end
    elseif case_name == "c12_financial_based_q1_noCfD"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            batch_cost = res_avg[t]

            if farm == 1
                clawback[t, farm] = 0.0  # assuming no clawback for Q3
                revenue[t, farm] = power * λ
                total_revenue[farm] += power * λ
            else
                clawback[t, farm] = batch_cost * λ
                revenue[t, farm] = (power - batch_cost) * λ + Fix_SP
                total_revenue[farm] += (power - batch_cost) * λ + Fix_SP
            end
        end
    end
    elseif case_name == "c13_financial_based_q2_noCfD"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            batch_cost = res_avg[t]

            if farm == 2
                clawback[t, farm] = 0.0  # assuming no clawback for Q3
                revenue[t, farm] = power * λ
                total_revenue[farm] += power * λ
            else
                clawback[t, farm] = batch_cost * λ
                revenue[t, farm] = (power - batch_cost) * λ + Fix_SP
                total_revenue[farm] += (power - batch_cost) * λ + Fix_SP
            end
        end
    end
    elseif case_name == "c14_financial_based_q1_CfD"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            batch_cost = RES_bc_q1[t]

            if farm != 1
                clawback[t, farm] = 0.0  # assuming no clawback for Q3
                revenue[t, farm] = power * λ
                total_revenue[farm] += power * λ
            else
                clawback[t, farm] = batch_cost * λ
                revenue[t, farm] = (power - batch_cost) * λ + Fix_SP
                total_revenue[farm] += (power - batch_cost) * λ + Fix_SP
            end
        end
    end
    elseif case_name == "c15_financial_based_q2_CfD"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            batch_cost = res_avg[t]

            if farm != 2
                clawback[t, farm] = 0.0  # assuming no clawback for Q3
                revenue[t, farm] = power * λ
                total_revenue[farm] += power * λ
            else
                clawback[t, farm] = batch_cost * λ
                revenue[t, farm] = (power - batch_cost) * λ + Fix_SP
                total_revenue[farm] += (power - batch_cost) * λ + Fix_SP
            end
        end
    end
    elseif case_name == "c16_financial_based_q3_CfD"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            batch_cost = res_avg[t]

            if farm != 3
                clawback[t, farm] = 0.0  # assuming no clawback for Q3
                revenue[t, farm] = power * λ
                total_revenue[farm] += power * λ
            else
                clawback[t, farm] = batch_cost * λ
                revenue[t, farm] = (power - batch_cost) * λ + Fix_SP
                total_revenue[farm] += (power - batch_cost) * λ + Fix_SP
            end
        end
    end
    elseif case_name == "c17_financial_sp10000"
    for t in time_steps
        for farm in 1:3
            λ = price[t]
            power = available_power[t, farm]
            batch_cost = mean_available_power[t]
            clawback[t,farm] = batch_cost * λ
            revenue[t,farm] = (power - batch_cost) * λ + Fix_SP
            total_revenue[farm] += (power - batch_cost) * λ + Fix_SP
        end
    end
end


# Compute and print mean and standard deviation after revenue is populated
mean_revenue = [mean(revenue[:, farm]) for farm in 1:3]
std_revenue = [std(revenue[:, farm]) for farm in 1:3]
var_revenue = [var(revenue[:, farm]) for farm in 1:3]

clawback_wf1 = sum(clawback[:, 1])
clawback_wf2 = sum(clawback[:, 2])
clawback_wf3 = sum(clawback[:, 3])

revenue_df = DataFrame(Timestep = 1:744)
for farm in 1:3
    revenue_df[!, "Farm_$(farm)_Revenue"] = revenue[:, farm]
end
CSV.write(joinpath("E:/TU Delft/Thesis/Code/Results/$(scenario_folder)/results_summary_$(case_name)/revenue_hourly_$(case_name).csv"), revenue_df)

total_revenue  

summary_df = DataFrame(
    Farm = ["Farm 1", "Farm 2", "Farm 3"],
    Total_Revenue = total_revenue,
    Mean_Revenue = mean_revenue,
    Std_Revenue = std_revenue,
    Var_Revenue = var_revenue
)

CSV.write(joinpath("E:/TU Delft/Thesis/Code/Results/$(scenario_folder)/results_summary_$(case_name)/revenue_summary_$(case_name).csv"), summary_df)


