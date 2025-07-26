using JuMP, DataFrames, Plots, CSV

# ------------------------------------------------------------------
# STEP 1: Identify Congested DC Lines at Zero-Price Timesteps
# ------------------------------------------------------------------

# Read price and power flow results
# Read the CSV as DataFrame (all columns as strings if mixed content)
df = CSV.read("E:/TU Delft/Thesis/Code/Results/scenario_2/c2_two_sided/c2_prices.csv", DataFrame)

# Extract numeric portion: rows 1:4 and columns 1:744
# First, select the numeric region from the DataFrame
df_numeric = df[1:4, 1:744]

# Convert to Float64 matrix
zone_prices = Matrix{Float64}(undef, 4, 744)

for i in 1:4, j in 1:744
    zone_prices[i, j] = parse(Float64, df_numeric[i, j])
end
F_DC_DA     = Matrix(CSV.read("E:/TU Delft/Thesis/Code/Results/scenario_2/bs1_DC_flow.csv", DataFrame))  # DC flow per line and time

# Load model sets and parameters
obz_zone = 4  # Zone 4 = Offshore Bidding Zone
TC_DC    = a.ext[:parameters][:TC_DC]  # Transmission capacity for each DC line
T        = a.ext[:sets][:T]  # Timestamps

# Identify zero-price hours in OBZ
zero_price_times = [t for t in T if abs(zone_prices[obz_zone, t]) < 1e-6]

# Track congested lines: Dict{Timestep => Vector of (LineIndex, Direction)}
congested_dc_lines = Dict{Int, Vector{Tuple{Int, String}}}()

for t in zero_price_times
    congested_lines_at_t = Tuple{Int, String}[]  # Reset for each time

    for l_dc in eachindex(TC_DC)
        flow = F_DC_DA[l_dc, t]
        capacity = TC_DC[l_dc]

        if abs(flow) ≥ capacity  # Congestion condition
            direction = flow > 0 ? "+1" : "-1"
            push!(congested_lines_at_t, (l_dc, direction))
        end
    end

    if !isempty(congested_lines_at_t)
        congested_dc_lines[t] = congested_lines_at_t
    end
end

# Print detected congestions
println("Congested DC lines by timestamps:")
for (t, lines) in congested_dc_lines
    println("Timestep $t: $lines")
end


# ------------------------------------------------------------------
# STEP 2: Map DC Line Index to (FromBus, ToBus)
# ------------------------------------------------------------------

dc_line_mapping = Dict{Int, Tuple{Int, Int}}()
for (i, row) in enumerate(eachrow(df_DC))
    dc_line_mapping[i] = (row[:FromBus], row[:ToBus])
end
println("Stored DC Line Mappings: ", dc_line_mapping)


# ------------------------------------------------------------------
# STEP 3: Count Congestion Frequency and Compute Percentages
# ------------------------------------------------------------------

congested_counts = Dict{Int, Int}()
for (_, lines) in congested_dc_lines
    for (l_dc, _) in lines
        congested_counts[l_dc] = get(congested_counts, l_dc, 0) + 1
    end
end

total_congestion_timesteps = length(congested_dc_lines)
congestion_percentage = Dict{Int, Float64}()
for (l_dc, count) in congested_counts
    congestion_percentage[l_dc] = (count / total_congestion_timesteps) * 100
end

# Print percentage congestion with bus info
println("Percentage of time each DC line is congested (with bus info):")
for (l_dc, pct) in sort(collect(congestion_percentage), by = x -> x[1])
    from_bus, to_bus = get(dc_line_mapping, l_dc, ("MISSING", "MISSING"))
    println("DC Line $l_dc: From Bus $from_bus → To Bus $to_bus | Congested $(round(pct, digits=2))% of the time")
end


# ------------------------------------------------------------------
# STEP 4: Analyse OBZ Price Relative to Other Zones
# ------------------------------------------------------------------

nonobz_price = zone_prices[[1, 2, 3], :]
obz_price    = zone_prices[obz_zone, :]

zone4_lowest   = Vector{Tuple{Int, Vector{Int}}}()
zone4_middle   = Vector{Tuple{Int, Vector{Int}}}()
zone4_highest  = Vector{Tuple{Int, Vector{Int}}}()
zone4_different = Int[]

for t in T
    prices_at_t = nonobz_price[:, t]
    sorted_prices = sort(collect(enumerate(prices_at_t)), by = x -> x[2])
    sorted_zones, sorted_values = first.(sorted_prices), last.(sorted_prices)
    unique_values = unique(sorted_values)

    if obz_price[t] ≈ unique_values[1]
        min_zones = [sorted_zones[i] for i in findall(x -> x == unique_values[1], sorted_values)]
        push!(zone4_lowest, (t, min_zones))
    elseif length(unique_values) ≥ 3 && obz_price[t] ≈ unique_values[2]
        mid_zones = [sorted_zones[i] for i in findall(x -> x == unique_values[2], sorted_values)]
        push!(zone4_middle, (t, mid_zones))
    elseif obz_price[t] ≈ unique_values[end]
        max_zones = [sorted_zones[i] for i in findall(x -> x == unique_values[end], sorted_values)]
        push!(zone4_highest, (t, max_zones))
    else
        push!(zone4_different, t)
    end
end

println("Zone 4 price categories by timestamps and matching zones:")
println("Lowest price timestamps: ", zone4_lowest)
println("Middle price timestamps: ", zone4_middle)
println("Highest price timestamps: ", zone4_highest)
println("Completely different timestamps: ", zone4_different)


# ------------------------------------------------------------------
# STEP 5: Compute OBZ Rank in Timestamps with Unique Prices
# ------------------------------------------------------------------

zone4_rankings = Dict{Int, Int}()
price = zone_prices

price_diff = zeros(4, length(zone4_different))
obz_diff   = zeros(length(zone4_different))
obz_rank   = zeros(Int, length(zone4_different))

for (i, diff_t) in enumerate(zone4_different)
    price_diff[:, i] = price[:, diff_t]
    obz_diff[i] = price[obz_zone, diff_t]

    sorted_prices = sort(price_diff[:, i], rev = true)
    obz_rank[i] = findfirst(x -> isapprox(x, obz_diff[i]; atol = 1e-8), sorted_prices)
end

println("OBZ ranks at specified timesteps: ", obz_rank)


# ------------------------------------------------------------------
# STEP 6: Plot Price Duration Curve
# ------------------------------------------------------------------

zone1_prices = zone_prices[1, :]
zone2_prices = zone_prices[2, :]
zone3_prices = zone_prices[3, :]
zone4_prices = zone_prices[4, :]

price_dur1 = sort(zone1_prices, rev = true)
price_dur2 = sort(zone2_prices, rev = true)
price_dur3 = sort(zone3_prices, rev = true)
price_dur4 = sort(zone4_prices, rev = true)

num_hours = length(price_dur1)
hours = 1:num_hours

default(
    fontfamily = "sans-serif",
    linewidth = 3,
    size = (1400, 800),
    framestyle = :box,
    gridalpha = 0.5,
    left_margin = 10Plots.mm,
    bottom_margin = 10Plots.mm,
    legend = :topright
)

p = plot(hours, price_dur1, label = "Zone 1", linewidth = 2.5, color = :blue)
plot!(p, hours, price_dur2, label = "Zone 2", linewidth = 2.5, color = :red)
plot!(p, hours, price_dur3, label = "Zone 3", linewidth = 2.5, color = :green)
plot!(p, hours, price_dur4, label = "Zone 4", linewidth = 4.0, color = :purple)

xlabel!(p, "Hours", fontsize = 14)
ylabel!(p, "Electricity Price (€/MWh)", fontsize = 16)
title!(p, "Price Duration Curve", fontsize = 16)

savefig(p, "E:/TU Delft/Thesis/Code/Results/Pics/price_duration_curve_s5_1.png")