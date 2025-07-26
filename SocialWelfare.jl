import Pkg
Pkg.add("Plots")  #
Pkg.add("CSV")
Pkg.add("DataFrames")

using CSV, DataFrames, Statistics, Plots, Measures

# SCENARIO AND CASE SELECTION
scenario_num = 1  # 1: case_study, 2: scenario_2, ...
alt_case = "c4a"  # options: "c4a", "c11"
base_case = "c2"  # always fixed to base (two-sided)

case_pair = (alt_case, base_case)

function get_case_folder(case::String, scenario_num::Int)
    base = scenario_num == 1 ? "case_study" : "scenario_$(scenario_num)"
    case_map = Dict(
        "c2"  => "c2_two_sided",
        "c4a" => "c4_financial_avg",
        "c11" => "c11_financial_based_q3_noCfD"
    )
    prefix_map = Dict(
        "c2"  => "c2",
        "c4a" => "c4a",
        "c11" => "c11"
    )
    folder = case_map[case]
    prefix = prefix_map[case]
    return "E:/TU Delft/Thesis/Code/Results/$(base)/$(folder)", prefix
end

function parse_price_matrix(path::String)
    df = CSV.read(path, DataFrame)
    return [parse(Float64, df[i, j]) for i in 1:4, j in 1:744]
end

function run_case_comparison(alt_case::String, base_case::String, scenario_num::Int)

    alt_dir, alt_prefix = get_case_folder(alt_case, scenario_num)
    base_dir, base_prefix = get_case_folder(base_case, scenario_num)

    # Load data
    prices_alt = parse_price_matrix("$(alt_dir)/$(alt_prefix)_prices.csv")
    prices_base = parse_price_matrix("$(base_dir)/$(base_prefix)_prices.csv")
    position_alt = Matrix(CSV.read("$(alt_dir)/$(alt_prefix)_position.csv", DataFrame))
    position_base = Matrix(CSV.read("$(base_dir)/$(base_prefix)_position.csv", DataFrame))
    available_alt = Matrix(CSV.read("$(alt_dir)/$(alt_prefix)_available_power.csv", DataFrame))[1:744, 1:3]
    available_base = Matrix(CSV.read("$(base_dir)/$(base_prefix)_available_power.csv", DataFrame))[1:744, 1:3]

    # Wind cleared difference
    wind_cleared_alt = sum(available_alt, dims=2)
    wind_cleared_base = sum(available_base, dims=2)
    delta = wind_cleared_alt - wind_cleared_base
    decline_timesteps = findall(delta .< 0)

    # Delta exports per zone
    delta_zones = Dict(z => position_alt[z, :] - position_base[z, :] for z in 1:3)
    extra_cost = 0.0
    for t in decline_timesteps
        for z in 1:3
            d = delta_zones[z][t]
            if d > 0
                extra_cost += d * prices_alt[z, t]
            end
        end
    end

    # Price-only effect
    price_effect = 0.0
    for t in decline_timesteps
        for z in 1:3
            if delta_zones[z][t] ≈ 0.0 && (prices_alt[z, t] - prices_base[z, t]) > 0
                base_gen = position_base[z, t]
                price_effect += base_gen * (prices_alt[z, t] - prices_base[z, t])
            end
        end
    end

    # Revenue calculation
    revenue = zeros(744, 3)
    clawback = zeros(744, 3)
    total_revenue = zeros(3)
    mean_power = [mean(available_alt[t, :]) for t in 1:744]
    Fix_SP = 1920.0

    for t in 1:744
        for f in 1:3
            λ = prices_alt[4, t]
            power = available_alt[t, f]
            batch = mean_power[t]

            if alt_case == "c11" && f == 3
                revenue[t, f] = power * λ
                clawback[t, f] = 0.0
            else
                revenue[t, f] = (power - batch) * λ + Fix_SP
                clawback[t, f] = batch * λ
            end

            total_revenue[f] += revenue[t, f]
        end
    end

    # Base case revenue
    base_revenue = sum(sum(available_base[t, :] .* prices_base[4, t]) for t in 1:744)
    wind_revenue = sum(total_revenue)
    Δproducer_profit = wind_revenue - base_revenue
    num_supported_farms = alt_case == "c11" ? 2 : 3
    total_gov_payout = Fix_SP * 744 * num_supported_farms
    Δsocial_welfare = Δproducer_profit - extra_cost - price_effect

    return Dict(
        "Δproducer_profit" => Δproducer_profit,
        "extra_generation_cost" => extra_cost,
        "price_effect_cost" => price_effect,
        "total_revenue" => wind_revenue,
        "base_revenue" => base_revenue,
        "gov_payout" => total_gov_payout,
        "Δsocial_welfare" => Δsocial_welfare
    )
end

results = Dict()
for scen in 1:5
    results["Scenario $scen - c4a vs c2"] = run_case_comparison("c4a", "c2", scen)
    results["Scenario $scen - c11 vs c2"] = run_case_comparison("c11", "c2", scen)
end

for scen in 1:5
    println("========== Scenario $(scen == 1 ? "case_study" : "scenario_$scen") ==========")

    for alt_case in ["c4a", "c11"]
        key = "Scenario $scen - $alt_case vs c2"
        result = results[key]
        println("$alt_case vs c2:")
        println("  Δproducer_profit     = $(round(result["Δproducer_profit"], digits=2))")
        println("  Extra generation cost = $(round(result["extra_generation_cost"], digits=2))")
        println("  Price effect cost     = $(round(result["price_effect_cost"], digits=2))")
        println("  Total revenue         = $(round(result["total_revenue"], digits=2))")
        println("  Base revenue          = $(round(result["base_revenue"], digits=2))")
        println("  Government payout     = $(round(result["gov_payout"], digits=2))")
        println("  Δsocial_welfare       = $(round(result["Δsocial_welfare"], digits=2))")
        println()
    end
end

delta
sum(delta)

default(
    dpi = 600,
    size = (1600, 1000),
    left_margin = 20mm,              
    legendfontsize = 11,
    guidefontsize = 13,
    tickfontsize = 11,
    titlefontsize = 14,
    lw = 0.6
)

for scen in 1:5
    for alt_case in ["c4a", "c11"]
        alt_dir, alt_prefix = get_case_folder(alt_case, scen)
        base_dir, base_prefix = get_case_folder("c2", scen)

        available_alt = Matrix(CSV.read("$(alt_dir)/$(alt_prefix)_available_power.csv", DataFrame))[1:744, 1:3]
        available_base = Matrix(CSV.read("$(base_dir)/$(base_prefix)_available_power.csv", DataFrame))[1:744, 1:3]
        prices_alt = parse_price_matrix("$(alt_dir)/$(alt_prefix)_prices.csv")
        obz_price = prices_alt[4, :]

        total_base = sum(available_base, dims=2)[:]
        total_alt = sum(available_alt, dims=2)[:]
        time = 1:744

        # Primary axis: Cleared power
        p1 = plot(
            time, total_base,
            label = "Base Case (c2)", color = :blue, lw = 1,
            xlabel = "Time (h)", ylabel = "Energy Dispatched (MWh)",
            legend = :topright,
            framestyle = :box,
            grid = :on
        )
        plot!(p1, time, total_alt, label = "CfD Case ($alt_case)", color = :darkred, lw = 2)

        # Secondary axis: OBZ price
        p2 = twinx()
        plot!(p2, time, obz_price, label = "OBZ Price (€/MWh)", color = :yellow, lw = 1.0,
              ylabel = "OBZ Price (€/MWh)", ylims = (0, maximum(obz_price) * 1.1))

        # Save high-resolution figure
        savefig("E:/TU Delft/Thesis/Code/Results/Pics/cleared_power_price_scenario$(scen)_$(alt_case).png")
    end
end