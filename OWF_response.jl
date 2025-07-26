# --- Load Required Packages --- #
using JuMP, Gurobi, DataFrames, CSV, Plots, Statistics, StatsBase

# --- Load Initial Data --- #
include("data.jl")

# --- Define Technology Data Paths --- #
technology_dir = "E:/TU Delft/Thesis/OBZ_CfD/"
wind_farm_cap = 1500.0  # MW

# --- Load Wind Technology Production Profiles (normalised to capacity) --- #
RES_v164 = CSV.read(technology_dir * "owf_vestas_v164.csv", DataFrame)[:,7] * wind_farm_cap
RES_v80  = CSV.read(technology_dir * "owf_vestas_v80.csv", DataFrame)[:,7]  * wind_farm_cap
RES_v112 = CSV.read(technology_dir * "owf_vestas_v112.csv", DataFrame)[:,7] * wind_farm_cap

# --- Assign RES Profiles to Nodes (Editable for Different Scenarios) --- #
RES[:, 98] = RES_v80
RES[:, 99] = RES_v80
RES[:,103] = RES_v80  # Scenario 1 (base case)

# Other scenarios: just overwrite these 3 lines when switching scenarios

# --- Define Shifted RES Profiles --- #
RES_q1 = RES[:,98]
RES_q2 = RES[:,99]
RES_q3 = RES[:,103]

shifted_RES_q1 = vcat(RES_q1[end-9:end], RES_q1[1:end-10])
shifted_RES_q2 = vcat(RES_q2[end-9:end], RES_q2[1:end-10])
shifted_RES_q3 = vcat(RES_q3[21:end], RES_q3[1:20])

# --- Visualise First 100 Hours of RES Profiles --- #
times = 1:100
plot(times, RES_v164[1:100], label="Vestas V164", linewidth=2)
plot!(times, RES_v80[1:100], label="Vestas V80", linewidth=2)
plot!(times, RES_v112[1:100], label="Vestas V112", linewidth=2)

# --- Plot Shifted RES Profiles (Full Horizon) --- #
timee = 1:744
plot(timee, RES_q1, label="Wind farm 1", linewidth=2)
plot!(timee, RES_q2, label="Wind farm 2", linewidth=2)
plot!(timee, RES_q3, label="Wind farm 3", linewidth=2)

# --- Moving Average Smoother --- #
function moving_average(x, window_size)
    result = similar(x)
    half = div(window_size, 2)
    for i in 1:length(x)
        result[i] = mean(x[max(1, i-half):min(end, i+half)])
    end
    return result
end

# Apply smoothing
smooth_RES_98  = moving_average(RES_v80, 24)
smooth_RES_99  = moving_average(shifted_RES_q2, 24)
smooth_RES_103 = moving_average(shifted_RES_q3, 24)

# Plot smoothed output
gr()
default(legend = :topright, xlabel = "Time (hours)", ylabel = "Power Output (MW)",
        title = "Smoothed Power Output from Wind Farms", lw = 2,
        size = (1000, 600), dpi = 300, grid = true, framestyle = :box)

p = plot(timee, smooth_RES_98,  label="Wind Farm 1", color=:black)
plot!(p, timee, smooth_RES_99,  label="Wind Farm 2", color=:red)
plot!(p, timee, smooth_RES_103, label="Wind Farm 3", color=:green)
savefig("E:/TU Delft/Thesis/Code/Results/Pics/scenario_2_OWF.png")

# --- Average RES Computation --- #
res_avg = (RES_q1 .+ RES_q2 .+ RES_q3) ./ 3
res_batch_avg = [mean(res_avg[(i-1)*30+1:min(i*30, end)]) for i in 1:div(length(T),30)]

# --- Run Main Model --- #
include("function_zonalclearing_GSK_AHC_H2.jl")

# --- Extract Base Case Results --- #
initial_λ_opt = Array(JuMP.dual.(a.ext[:constraints][:con1])[4, :])
available_power_bc = Matrix(a.ext[:parameters][:RES])[:, [98,99,103]] -
                     Matrix(a.ext[:parameters][:curt_DA])[[98,99,103], :]'

avg_available_power_bc = mean(available_power_bc, dims=2)
available_power_bc_node98 = available_power_bc[:,1]
available_power_bc_node99 = available_power_bc[:,2]
available_power_bc_node103 = available_power_bc[:,3]

# Save batch-wise RES averages
res_batch_avg = [mean(avg_available_power_bc[(i-1)*30+1:min(i*30,end)]) for i in 1:div(length(T),30)]

# --- Extract Model Outputs --- #
bs1_DC_flow        = Matrix(value.(a.ext[:variables][:F_DC]))
bs1_electrolysers  = Matrix(value.(a.ext[:variables][:e]))
bs1_posiition      = Matrix(value.(a.ext[:variables][:p]))
bs1_curtailment    = Matrix(value.(a.ext[:variables][:curt]))[[98,99,103], :]
bs1_available_power = permutedims(RES[:,[98,99,103]]) .- bs1_curtailment
bs1_prices         = Matrix(dual.(a.ext[:constraints][:con1]))

# --- Save Outputs (Set Scenario Directory) --- #
result_dir = "E:/TU Delft/Thesis/Code/Results/scenario_5_2/"
CSV.write(result_dir * "bs1_lambda.csv",        DataFrame(bs1 = initial_λ_opt))
CSV.write(result_dir * "bs1_DC_flow.csv",       DataFrame(bs1_DC_flow, :auto))
CSV.write(result_dir * "bs1_electrolysers.csv", DataFrame(bs1_electrolysers, :auto))
CSV.write(result_dir * "bs1_position.csv",      DataFrame(bs1_posiition, :auto))
CSV.write(result_dir * "bs1_curtailment.csv",   DataFrame(bs1_curtailment, :auto))
CSV.write(result_dir * "bs1_available_power.csv", DataFrame(bs1_available_power, :auto))
CSV.write(result_dir * "bs1_prices.csv",        DataFrame(bs1_prices, :auto))

# --- Load and Modify Reference RES Matrix --- #
updated_RES = a.ext[:parameters][:RES] = Matrix(CSV.read("E:/TU Delft/Thesis/Code/grid_topology/Reference_Case/RES.csv", DataFrame))

# Overwrite reference with current RES for scenario
updated_RES[:,98] = RES_q1
updated_RES[:,99] = RES_q2
updated_RES[:,103] = RES_q3

# --- Extract Offshore Generation for Zone 4 --- #
offshore_gen_zone_4 = zeros(Float64, length(T), 3)
zone_4_node_indices = findall(df_node.ZoneRes .== 4)
for i = 1:3
    offshore_gen_zone_4[:,i] = RES[:,zone_4_node_indices[i]]
end

# --- CfD Case Notes --- #
# case 1  → Simple CfD (test only)
# case 2  → Two-sided CfD
# case 3  → Cap and floor CfD
# case 4a → Financial CfD with fixed compensation
# case 4b → Financial CfD with compensation when no production
# case 5  → Financial CfD with different reference generators



function optimize_revenue(zone_4_node_indices, T, updated_RES::Matrix{Float64}, inner_initial_λ_opt, SP::Float64, CfD_type=:simple, q_prev::Union{Nothing, Matrix{Float64}} = nothing, step_limit::Float64 = 0.2)

    rev_node98 = Model(Gurobi.Optimizer)
    set_optimizer_attribute(rev_node98, "OutputFlag", 1)
    set_optimizer_attribute(rev_node98, "InfUnbdInfo", 1)  # Enable detailed Gurobi output
    
    rev_node99 = Model(Gurobi.Optimizer)
    set_optimizer_attribute(rev_node99, "InfUnbdInfo", 1)
    set_optimizer_attribute(rev_node99, "OutputFlag", 1)
    
    rev_node103 = Model(Gurobi.Optimizer)
    set_optimizer_attribute(rev_node103, "OutputFlag", 1)
    set_optimizer_attribute(rev_node103, "InfUnbdInfo", 1)

    yearly_avg = mean(inner_initial_λ_opt)

    # Initialize variables and constraints for tracking
    rev_node98.ext = Dict(:variables => Dict(), :constraints => Dict(), :objective => Dict())
    rev_node99.ext = Dict(:variables => Dict(), :constraints => Dict(), :objective => Dict())
    rev_node103.ext = Dict(:variables => Dict(), :constraints => Dict(), :objective => Dict())

    # Decision variables for each node
    q1 = rev_node98.ext[:variables][:q1] = @variable(rev_node98, [t in T], lower_bound = 0, base_name = "q_node1")
    q2 = rev_node99.ext[:variables][:q2] = @variable(rev_node99, [t in T], lower_bound = 0, base_name = "q_node2")
    q3 = rev_node103.ext[:variables][:q3] = @variable(rev_node103, [t in T], lower_bound = 0, base_name = "q_node3")

    # Constraints for each node
    const1 = rev_node98.ext[:constraints][:const1] = @constraint(rev_node98, [t in T], q1[t] <= offshore_gen_zone_4[t, 1])  # Node 1
    const2 = rev_node99.ext[:constraints][:const2] = @constraint(rev_node99, [t in T], q2[t] <= offshore_gen_zone_4[t, 2])  # Node 2
    const3 =rev_node103.ext[:constraints][:const3] = @constraint(rev_node103, [t in T], q3[t] <= offshore_gen_zone_4[t, 3])  # Node 3

    #if q_prev !== nothing
    #    q1_prev = q_prev[1, :]
    #    q2_prev = q_prev[2, :]
    #    q3_prev = q_prev[3, :]

#        for t_idx in eachindex(T)
#            t = T[t_idx]
#            base1 = max(abs(q1_prev[t_idx]), 1e-3)
#            base2 = max(abs(q2_prev[t_idx]), 1e-3)
#            base3 = max(abs(q3_prev[t_idx]), 1e-3)

#            @constraint(rev_node98, q1[t] <= q1_prev[t_idx] + step_limit * base1)
#            @constraint(rev_node98, q1[t] >= q1_prev[t_idx] - step_limit * base1)

#            @constraint(rev_node99, q2[t] <= q2_prev[t_idx] + step_limit * base2)
#            @constraint(rev_node99, q2[t] >= q2_prev[t_idx] - step_limit * base2)

#            @constraint(rev_node103, q3[t] <= q3_prev[t_idx] + step_limit * base3)
#            @constraint(rev_node103, q3[t] >= q3_prev[t_idx] - step_limit * base3)
#        end    
#    end


    # Objective based on CfD type
    if CfD_type == :simple
        # Objective 1: Maximise revenue for each node
        rev_node98.ext[:objective][:rev98_simple] = @objective(rev_node98, Max, sum(
            max(SP, inner_initial_λ_opt[t]) * q1[t] 
            for t in T)
        )

        rev_node99.ext[:objective][:rev99_simple] = @objective(rev_node99, Max, sum(
            max(SP, inner_initial_λ_opt[t]) * q2[t] 
            for t in T)
        )

        rev_node103.ext[:objective][:rev103_simple] = @objective(rev_node103, Max, sum(
            max(SP, inner_initial_λ_opt[t]) * q3[t] 
            for t in T)
        )

    elseif CfD_type == :two_sided
        rev_node98.ext[:objective][:rev98_two_sided] = @objective(rev_node98, Max, sum(
            (q1[t] * inner_initial_λ_opt[t]) + (q1[t] * (SP - inner_initial_λ_opt[t]))
            for t in T)
        )
        rev_node99.ext[:objective][:rev99_two_sided] = @objective(rev_node99, Max, sum(
            (q2[t] * inner_initial_λ_opt[t]) + (q2[t] * (SP - inner_initial_λ_opt[t]))
            for t in T)
        )
        rev_node103.ext[:objective][:rev103_two_sided] = @objective(rev_node103, Max, sum(
            (q3[t] * inner_initial_λ_opt[t]) + (q3[t] * (SP - inner_initial_λ_opt[t]))
            for t in T)
        )
    
    elseif CfD_type == :two_sided_yearly_avg
        rev_node98.ext[:objective][:rev98_two_sided] = @objective(rev_node98, Max, sum(
            (q1[t] * inner_initial_λ_opt[t]) + (q1[t] * (SP - yearly_avg))
            for t in T)
        )
        rev_node99.ext[:objective][:rev99_two_sided] = @objective(rev_node99, Max, sum(
            (q2[t] * inner_initial_λ_opt[t]) + (q2[t] * (SP - yearly_avg))
            for t in T)
        )
        rev_node103.ext[:objective][:rev103_two_sided] = @objective(rev_node103, Max, sum(
            (q3[t] * inner_initial_λ_opt[t]) + (q3[t] * (SP - yearly_avg))
            for t in T)
        )
    
    elseif CfD_type == :cap_floor
        cap = 20.0
        floor = 8.0
        rev_node98.ext[:objective][:rev98_cap_floor] = @objective(rev_node98, Max, sum(
        q1[t] * (
        inner_initial_λ_opt[t] +
        max(floor - inner_initial_λ_opt[t], 0) -  # Compensation if below floor
        max(inner_initial_λ_opt[t] - cap, 0) ) for t in T))    # Payback if above cap 

        rev_node99.ext[:objective][:rev99_cap_floor] = @objective(rev_node99, Max, sum(
        q2[t] * (
        inner_initial_λ_opt[t] +
        max(floor - inner_initial_λ_opt[t], 0) -  # Compensation if below floor
        max(inner_initial_λ_opt[t] - cap, 0) ) for t in T))     # Payback if above cap
        
        rev_node103.ext[:objective][:rev103_cap_floor] = @objective(rev_node103, Max, sum(
        q3[t] * (
        inner_initial_λ_opt[t] +
        max(floor - inner_initial_λ_opt[t], 0) -  # Compensation if below floor
        max(inner_initial_λ_opt[t] - cap, 0)   ) for t in T))   # Payback if above cap


    elseif CfD_type == :capability_based_avg
        penalty_weight = 1.0  # Tune this value to control how much you penalise jumps
    


        if q_prev === nothing

            # First iteration: no penalty, only normal objective
            rev_node98.ext[:objective][:rev98_simple] = @objective(rev_node98, Max, sum(
                SP +
                q1[t] * inner_initial_λ_opt[t] -
                inner_initial_λ_opt[t] * avg_available_power_bc[1]
                for t in T))
            
            rev_node99.ext[:objective][:rev99_simple] = @objective(rev_node99, Max, sum(
                SP +
                q2[t] * inner_initial_λ_opt[t] -
                inner_initial_λ_opt[t] * avg_available_power_bc[1]
                for t in T))
        
            rev_node103.ext[:objective][:rev103_simple] = @objective(rev_node103, Max, sum(
                SP +
                q3[t] * inner_initial_λ_opt[t] -
                inner_initial_λ_opt[t] * avg_available_power_bc[1]
                for t in T))
        
        else
            # From second iteration: apply penalty
            q1_prev = q_prev[1, :]
            q2_prev = q_prev[2, :]
            q3_prev = q_prev[3, :]
        
            penalty_weight = 1.0  # (or whatever you prefer)
        
            rev_node98.ext[:objective][:rev98_penalty] = @objective(rev_node98, Max, sum(
                SP +
                q1[t] * inner_initial_λ_opt[t] -
                inner_initial_λ_opt[t] * (q1_prev[t]+ q2_prev[t] + q3_prev[t]) / 3.0 -
                penalty_weight * (q1[t] - q1_prev[t])^2
                for t in T))
        
            rev_node99.ext[:objective][:rev99_penalty] = @objective(rev_node99, Max, sum(
                SP +
                q2[t] * inner_initial_λ_opt[t] -
                inner_initial_λ_opt[t] * (q1_prev[t]+ q2_prev[t] + q3_prev[t]) / 3.0 -
                penalty_weight * (q2[t] - q2_prev[t])^2
                for t in T))
        
            rev_node103.ext[:objective][:rev103_penalty] = @objective(rev_node103, Max, sum(
                SP +
                q3[t] * inner_initial_λ_opt[t] -
                inner_initial_λ_opt[t] * (q1_prev[t]+ q2_prev[t] + q3_prev[t]) / 3.0 -
                penalty_weight * (q3[t] - q3_prev[t])^2
                for t in T))
        end

    elseif CfD_type == :capability_based_avg2
        penalty_weight = 1.0  # You can tune this
        
        # Binary variable δ[t] to indicate whether q[t] > 0
        δ1 = @variable(rev_node98, [t in T], Bin, base_name = "delta_node1")
        δ2 = @variable(rev_node99, [t in T], Bin, base_name = "delta_node2")
        δ3 = @variable(rev_node103, [t in T], Bin, base_name = "delta_node3")
        
        # Constraints to link δ[t] with q[t]
        M = 1e6  # A sufficiently large number
        @constraint(rev_node98, [t in T], q1[t] <= M * δ1[t])
        @constraint(rev_node99, [t in T], q2[t] <= M * δ2[t])
        @constraint(rev_node103, [t in T], q3[t] <= M * δ3[t])
        
        if q_prev === nothing
            # No penalties in first iteration
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                ((q1[t] - res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))]) * inner_initial_λ_opt[t]) +
                δ1[t] * SP
                for t in T))
        
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                ((q2[t] - res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))]) * inner_initial_λ_opt[t]) +
                δ2[t] * SP
                for t in T))
        
            rev_node103.ext[:objective][:rev103_fixed] = @objective(rev_node103, Max, sum(
                ((q3[t] - res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))]) * inner_initial_λ_opt[t]) +
                δ3[t] * SP
                for t in T))
        
        else
            # Apply penalty terms in second and further iterations
            q1_prev = q_prev[1, :]
            q2_prev = q_prev[2, :]
            q3_prev = q_prev[3, :]
        
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                ((q1[t] - res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))]) * inner_initial_λ_opt[t]) +
                δ1[t] * SP -
                penalty_weight * (q1[t] - q1_prev[t])^2
                for t in T))
        
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                ((q2[t] - res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))]) * inner_initial_λ_opt[t]) +
                δ2[t] * SP -
                penalty_weight * (q2[t] - q2_prev[t])^2
                for t in T))
        
            rev_node103.ext[:objective][:rev103_fixed] = @objective(rev_node103, Max, sum(
                ((q3[t] - res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))]) * inner_initial_λ_opt[t]) +
                δ3[t] * SP -
                penalty_weight * (q3[t] - q3_prev[t])^2
                for t in T))
        end

    elseif CfD_type == :capability_based_thresh
        threshold = 50.0
        M = 1e6
        penalty_weight = 1.0  # Tune as needed
    
        # Binary variables δ[t] for SP activation
        δ1 = @variable(rev_node98, [t in T], Bin, base_name = "delta_node1_thresh")
        δ2 = @variable(rev_node99, [t in T], Bin, base_name = "delta_node2_thresh")
        δ3 = @variable(rev_node103, [t in T], Bin, base_name = "delta_node3_thresh")
    
        # Enforce: if δ[t] = 1, then q[t] ≥ threshold
        @constraint(rev_node98, [t in T], q1[t] >= threshold * δ1[t])
        @constraint(rev_node99, [t in T], q2[t] >= threshold * δ2[t])
        @constraint(rev_node103, [t in T], q3[t] >= threshold * δ3[t])
    
        # Standard big-M upper bounds
        @constraint(rev_node98, [t in T], q1[t] <= M * δ1[t])
        @constraint(rev_node99, [t in T], q2[t] <= M * δ2[t])
        @constraint(rev_node103, [t in T], q3[t] <= M * δ3[t])
    
        if q_prev === nothing
            # First iteration – no penalty
            rev_node98.ext[:objective][:rev98_thresh] = @objective(rev_node98, Max, sum(
                q1[t] * inner_initial_λ_opt[t] +
                δ1[t] * SP -
                inner_initial_λ_opt[t] * res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))]
                for t in T))
    
            rev_node99.ext[:objective][:rev99_thresh] = @objective(rev_node99, Max, sum(
                q2[t] * inner_initial_λ_opt[t] +
                δ2[t] * SP -
                inner_initial_λ_opt[t] * res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))]
                for t in T))
    
            rev_node103.ext[:objective][:rev103_thresh] = @objective(rev_node103, Max, sum(
                q3[t] * inner_initial_λ_opt[t] +
                δ3[t] * SP -
                inner_initial_λ_opt[t] * res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))]
                for t in T))
        else
            # Later iterations – apply penalty
            q1_prev = q_prev[1, :]
            q2_prev = q_prev[2, :]
            q3_prev = q_prev[3, :]
    
            rev_node98.ext[:objective][:rev98_thresh] = @objective(rev_node98, Max, sum(
                q1[t] * inner_initial_λ_opt[t] +
                δ1[t] * SP -
                inner_initial_λ_opt[t] * res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))] -
                penalty_weight * (q1[t] - q1_prev[t])^2
                for t in T))
    
            rev_node99.ext[:objective][:rev99_thresh] = @objective(rev_node99, Max, sum(
                q2[t] * inner_initial_λ_opt[t] +
                δ2[t] * SP -
                inner_initial_λ_opt[t] * res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))] -
                penalty_weight * (q2[t] - q2_prev[t])^2
                for t in T))
    
            rev_node103.ext[:objective][:rev103_thresh] = @objective(rev_node103, Max, sum(
                q3[t] * inner_initial_λ_opt[t] +
                δ3[t] * SP -
                inner_initial_λ_opt[t] * res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))] -
                penalty_weight * (q3[t] - q3_prev[t])^2
                for t in T))
        end

    elseif CfD_type == :capability_based_avg3
        penalty_factor = 1.0  # or tune as needed
            
        if q_prev === nothing
            # First iteration – no penalty
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
            ((q1[t] - res_avg[t]) * inner_initial_λ_opt[t]) + SP
            for t in T))
            
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
            ((q2[t] - res_avg[t]) * inner_initial_λ_opt[t]) + SP
            for t in T))
            
            rev_node103.ext[:objective][:rev103_fixed] = @objective(rev_node103, Max, sum(
            ((q3[t] - res_avg[t]) * inner_initial_λ_opt[t]) + SP
            for t in T))
            
        else
            # Subsequent iterations – apply penalty on deviation from previous q
            q1_prev = q_prev[1, :]
            q2_prev = q_prev[2, :]
            q3_prev = q_prev[3, :]
            
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
            ((q1[t] - res_avg[t]) * inner_initial_λ_opt[t]) + SP -
            penalty_factor * (q1[t] - q1_prev[t])^2
            for t in T))
            
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
            ((q2[t] - res_avg[t]) * inner_initial_λ_opt[t]) + SP -
            penalty_factor * (q2[t] - q2_prev[t])^2
            for t in T))
            
            rev_node103.ext[:objective][:rev103_fixed] = @objective(rev_node103, Max, sum(
            ((q3[t] - res_avg[t]) * inner_initial_λ_opt[t]) + SP -
            penalty_factor * (q3[t] - q3_prev[t])^2
            for t in T))
        end
    
    elseif CfD_type == :capability_based_individualq1
        penalty_weight = 1.0  # Tune as needed
    
        if q_prev === nothing
            # First iteration – no penalty
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                q1[t] * inner_initial_λ_opt[t] + 
                SP -
                inner_initial_λ_opt[t] * available_power_bc_node98[t]
                for t in T))
    
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                q2[t] * inner_initial_λ_opt[t] + 
                SP -
                inner_initial_λ_opt[t] * available_power_bc_node98[t]
                for t in T))
    
            rev_node103.ext[:objective][:rev103_fixed] = @objective(rev_node103, Max, sum(
                q3[t] * inner_initial_λ_opt[t] + 
                SP - 
                inner_initial_λ_opt[t] * available_power_bc_node98[t]
                for t in T))
    
        else
            # Later iterations – reward matching previous + penalise deviation
            q1_prev = q_prev[1, :]
            q2_prev = q_prev[2, :]
            q3_prev = q_prev[3, :]
    
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                (q1[t]  * inner_initial_λ_opt[t]) + 
                SP -
                inner_initial_λ_opt[t] * available_power_bc_node98[t] -
                penalty_weight * (q1[t] - q1_prev[t])^2
                for t in T))
    
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                (q2[t]  * inner_initial_λ_opt[t]) + 
                SP -
                inner_initial_λ_opt[t] * available_power_bc_node98[t] -
                penalty_weight * (q2[t] - q2_prev[t])^2
                for t in T))
    
            rev_node103.ext[:objective][:rev103_fixed] = @objective(rev_node103, Max, sum(
                (q3[t]  * inner_initial_λ_opt[t]) + 
                SP -
                inner_initial_λ_opt[t] * available_power_bc_node98[t] -
                penalty_weight * (q3[t] - q3_prev[t])^2
                for t in T))
        end

    elseif CfD_type == :capability_based_individualq2
        penalty_weight = 1.0  # adjust as needed
    
        if q_prev === nothing
            # First iteration – no penalty
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                (q1[t] * inner_initial_λ_opt[t]) + 
                SP - 
                inner_initial_λ_opt[t] * available_power_bc_node99[t]
                for t in T))
    
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                (q2[t]  * inner_initial_λ_opt[t]) + 
                SP -
                inner_initial_λ_opt[t] * available_power_bc_node99[t]
                for t in T))
    
            rev_node103.ext[:objective][:rev103_fixed] = @objective(rev_node103, Max, sum(
                (q3[t] * inner_initial_λ_opt[t]) + 
                SP -
                inner_initial_λ_opt[t] * available_power_bc_node99[t]
                for t in T))
        else
            # Later iterations – apply penalty on deviation from q_prev
            q1_prev = q_prev[1, :]
            q2_prev = q_prev[2, :]
            q3_prev = q_prev[3, :]
    
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                ((q1[t] * inner_initial_λ_opt[t]) + 
                SP -
                inner_initial_λ_opt[t] * available_power_bc_node99[t]) -
                penalty_weight * (q1[t] - q1_prev[t])^2
                for t in T))
    
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                ((q2[t] * inner_initial_λ_opt[t]) + 
                SP -
                inner_initial_λ_opt[t] * available_power_bc_node99[t]) -
                penalty_weight * (q2[t] - q2_prev[t])^2
                for t in T))
    
            rev_node103.ext[:objective][:rev103_fixed] = @objective(rev_node103, Max, sum(
                (q3[t] * inner_initial_λ_opt[t]) + 
                SP -
                inner_initial_λ_opt[t] * available_power_bc_node99[t] -
                penalty_weight * (q3[t] - q3_prev[t])^2
                for t in T))
        end

    elseif CfD_type == :capability_based_individualq3
        penalty_factor = 1.0
    
        if q_prev === nothing
            # First iteration – no penalty
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                (q1[t] * inner_initial_λ_opt[t]) + 
                SP -    
                inner_initial_λ_opt[t] * available_power_bc_node103[t]
                for t in T))
    
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                (q2[t] * inner_initial_λ_opt[t]) + 
                SP -
                inner_initial_λ_opt[t] * available_power_bc_node103[t]
                for t in T))
    
            rev_node103.ext[:objective][:rev103_fixed] = @objective(rev_node103, Max, sum(
                (q3[t] * inner_initial_λ_opt[t]) + 
                SP -
                inner_initial_λ_opt[t] * available_power_bc_node103[t]
                for t in T))
        else
            # Later iterations – include penalty on deviation from previous q
            q1_prev = q_prev[1, :]
            q2_prev = q_prev[2, :]
            q3_prev = q_prev[3, :]
    
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                (q1[t] * inner_initial_λ_opt[t]) + 
                SP -
                inner_initial_λ_opt[t] * available_power_bc_node103[t] -
                penalty_factor * (q1[t] - q1_prev[t])^2
                for t in T))
    
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                (q2[t] * inner_initial_λ_opt[t]) + 
                SP -
                inner_initial_λ_opt[t] * available_power_bc_node103[t] -
                penalty_factor * (q2[t] - q2_prev[t])^2
                for t in T))
    
            rev_node103.ext[:objective][:rev103_fixed] = @objective(rev_node103, Max, sum(
                (q3[t] * inner_initial_λ_opt[t]) + 
                SP -
                inner_initial_λ_opt[t] * available_power_bc_node103[t] -
                penalty_factor * (q3[t] - q3_prev[t])^2
                for t in T))
        end

    elseif CfD_type == :capability_based_q1_noCfD
        
        penalty_weight = 1.0
    
        if q_prev === nothing
            # First iteration – no penalty
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                (q1[t]  * inner_initial_λ_opt[t]) 
                for t in T))
    
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                (q2[t] * inner_initial_λ_opt[t]) + 
                SP -
                inner_initial_λ_opt[t] * avg_available_power_bc[t]
                for t in T))
    
            rev_node103.ext[:objective][:rev103_two_sided] = @objective(rev_node103, Max, sum(
                q3[t] * inner_initial_λ_opt[t] +
                SP -
                inner_initial_λ_opt[t] * avg_available_power_bc[t]
                for t in T))
        else
            # Later iterations – apply penalty for deviation from previous q
            q1_prev = q_prev[1, :]
            q2_prev = q_prev[2, :]
            q3_prev = q_prev[3, :]
    
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                (q1[t] * inner_initial_λ_opt[t]) -
                penalty_weight * (q1[t] - q1_prev[t])^2
                for t in T))
    
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                ((q2[t] * inner_initial_λ_opt[t]) + 
                SP -
                inner_initial_λ_opt[t] * avg_available_power_bc[t]) -
                penalty_weight * (q2[t] - q2_prev[t])^2
                for t in T))
    
            rev_node103.ext[:objective][:rev103_two_sided] = @objective(rev_node103, Max, sum(
                q3[t] * inner_initial_λ_opt[t] +
                SP -
                inner_initial_λ_opt[t] * avg_available_power_bc[t] -
                penalty_weight * (q3[t] - q3_prev[t])^2
                for t in T))
        end

    elseif CfD_type == :capability_based_q2_noCfD
        
        penalty_weight = 1.0
    
        if q_prev === nothing
            # First iteration – no penalty
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                (q1[t]  * inner_initial_λ_opt[t]) +
                SP -
                inner_initial_λ_opt[t] * avg_available_power_bc[t]
                for t in T))
    
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                (q2[t] * inner_initial_λ_opt[t]) 
                for t in T))
    
            rev_node103.ext[:objective][:rev103_two_sided] = @objective(rev_node103, Max, sum(
                q3[t] * inner_initial_λ_opt[t] +
                SP -
                inner_initial_λ_opt[t] * avg_available_power_bc[t]
                for t in T))
        else
            # Later iterations – apply penalty for deviation from previous q
            q1_prev = q_prev[1, :]
            q2_prev = q_prev[2, :]
            q3_prev = q_prev[3, :]
    
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                (q1[t] * inner_initial_λ_opt[t]) +
                SP -
                inner_initial_λ_opt[t] * avg_available_power_bc[t] -
                penalty_weight * (q1[t] - q1_prev[t])^2
                for t in T))
    
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                (q2[t] * inner_initial_λ_opt[t]) -
                penalty_weight * (q2[t] - q2_prev[t])^2
                for t in T))
    
            rev_node103.ext[:objective][:rev103_two_sided] = @objective(rev_node103, Max, sum(
                q3[t] * inner_initial_λ_opt[t] +
                SP -
                inner_initial_λ_opt[t] * avg_available_power_bc[t] -
                penalty_weight * (q3[t] - q3_prev[t])^2
                for t in T))
        end

    elseif CfD_type == :capability_based_q3_noCfD
        penalty_weight = 1.0
    
        if q_prev === nothing
            # First iteration – no penalty
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                ((q1[t] - avg_available_power_bc[1]) + SP)
                for t in T))
    
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                ((q2[t] - avg_available_power_bc[1]) + SP)
                for t in T))
    
            rev_node103.ext[:objective][:rev103_two_sided] = @objective(rev_node103, Max, sum(
                q3[t] * inner_initial_λ_opt[t] 
                for t in T))
        else
            # Later iterations – apply penalty for deviation from previous q
            q1_prev = q_prev[1, :]
            q2_prev = q_prev[2, :]
            q3_prev = q_prev[3, :]
    
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                ((q1[t] - (q1_prev[t]+q2_prev[t])/2.0) * inner_initial_λ_opt[t]) + SP -
                penalty_weight * (q1[t] - q1_prev[t])^2
                for t in T))
    
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                ((q2[t] - (q1_prev[t]+q2_prev[t])/2.0) * inner_initial_λ_opt[t]) + SP -
                penalty_weight * (q2[t] - q2_prev[t])^2
                for t in T))
    
            rev_node103.ext[:objective][:rev103_two_sided] = @objective(rev_node103, Max, sum(
                q3[t] * inner_initial_λ_opt[t] -
                penalty_weight * (q3[t] - q3_prev[t])^2
                for t in T))
        end

    elseif CfD_type == :capability_based_q1_CfD
        
        penalty_weight = 1.0
    
        if q_prev === nothing
            # First iteration – no penalty
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                (q1[t]  * inner_initial_λ_opt[t]) +
                SP -
                inner_initial_λ_opt[t] * avg_available_power_bc[t]
                for t in T))
    
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                (q2[t] * inner_initial_λ_opt[t]) 
                for t in T))
    
            rev_node103.ext[:objective][:rev103_two_sided] = @objective(rev_node103, Max, sum(
                q3[t] * inner_initial_λ_opt[t] 
                for t in T))
        else
            # Later iterations – apply penalty for deviation from previous q
            q1_prev = q_prev[1, :]
            q2_prev = q_prev[2, :]
            q3_prev = q_prev[3, :]
    
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                (q1[t] * inner_initial_λ_opt[t]) +
                SP - 
                inner_initial_λ_opt[t] * avg_available_power_bc[t] -
                penalty_weight * (q1[t] - q1_prev[t])^2
                for t in T))
    
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                (q2[t] * inner_initial_λ_opt[t]) -
                penalty_weight * (q2[t] - q2_prev[t])^2
                for t in T))
    
            rev_node103.ext[:objective][:rev103_two_sided] = @objective(rev_node103, Max, sum(
                q3[t] * inner_initial_λ_opt[t] -
                penalty_weight * (q3[t] - q3_prev[t])^2
                for t in T))
        end    

    elseif CfD_type == :capability_based_q2_CfD
        
        penalty_weight = 1.0
    
        if q_prev === nothing
            # First iteration – no penalty
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                (q1[t]  * inner_initial_λ_opt[t])
                for t in T))
    
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                (q2[t] * inner_initial_λ_opt[t]) +
                SP -
                inner_initial_λ_opt[t] * avg_available_power_bc[t]
                for t in T))
    
            rev_node103.ext[:objective][:rev103_two_sided] = @objective(rev_node103, Max, sum(
                q3[t] * inner_initial_λ_opt[t] 
                for t in T))
        else
            # Later iterations – apply penalty for deviation from previous q
            q1_prev = q_prev[1, :]
            q2_prev = q_prev[2, :]
            q3_prev = q_prev[3, :]
    
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                (q1[t] * inner_initial_λ_opt[t]) -
                penalty_weight * (q1[t] - q1_prev[t])^2
                for t in T))
    
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                (q2[t] * inner_initial_λ_opt[t]) +
                SP -
                inner_initial_λ_opt[t] * avg_available_power_bc[t] -
                penalty_weight * (q2[t] - q2_prev[t])^2
                for t in T))
    
            rev_node103.ext[:objective][:rev103_two_sided] = @objective(rev_node103, Max, sum(
                q3[t] * inner_initial_λ_opt[t] -
                penalty_weight * (q3[t] - q3_prev[t])^2
                for t in T))
        end 

    elseif CfD_type == :capability_based_q3_CfD
        
        penalty_weight = 1.0
    
        if q_prev === nothing
            # First iteration – no penalty
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                (q1[t]  * inner_initial_λ_opt[t])
                for t in T))
    
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                (q2[t] * inner_initial_λ_opt[t]) 
                for t in T))
    
            rev_node103.ext[:objective][:rev103_two_sided] = @objective(rev_node103, Max, sum(
                q3[t] * inner_initial_λ_opt[t] +
                SP -
                inner_initial_λ_opt[t] * avg_available_power_bc[t]
                for t in T))
        else
            # Later iterations – apply penalty for deviation from previous q
            q1_prev = q_prev[1, :]
            q2_prev = q_prev[2, :]
            q3_prev = q_prev[3, :]
    
            rev_node98.ext[:objective][:rev98_fixed] = @objective(rev_node98, Max, sum(
                (q1[t] * inner_initial_λ_opt[t]) -
                penalty_weight * (q1[t] - q1_prev[t])^2
                for t in T))
    
            rev_node99.ext[:objective][:rev99_fixed] = @objective(rev_node99, Max, sum(
                (q2[t] * inner_initial_λ_opt[t]) -
                penalty_weight * (q2[t] - q2_prev[t])^2
                for t in T))
    
            rev_node103.ext[:objective][:rev103_two_sided] = @objective(rev_node103, Max, sum(
                q3[t] * inner_initial_λ_opt[t] +
                SP -
                inner_initial_λ_opt[t] * avg_available_power_bc[t] -
                penalty_weight * (q3[t] - q3_prev[t])^2
                for t in T))
        end 

    elseif CfD_type == :capability_based_actual
        penalty_weight = 0.01
    
        if q_prev === nothing
            # First iteration – no penalty
            rev_node98.ext[:objective][:ref_cost] = @objective(rev_node98, Max, sum(
                q1[t] * inner_initial_λ_opt[t] +
                (SP - inner_initial_λ_opt[t]) * res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))]
                for t in T))
    
            rev_node99.ext[:objective][:ref_cost] = @objective(rev_node99, Max, sum(
                q2[t] * inner_initial_λ_opt[t] +
                (SP - inner_initial_λ_opt[t]) * res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))]
                for t in T))
    
            rev_node103.ext[:objective][:ref_cost] = @objective(rev_node103, Max, sum(
                q3[t] * inner_initial_λ_opt[t] +
                (SP - inner_initial_λ_opt[t]) * res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))]
                for t in T))
        else
            # Later iterations – include penalty on q deviation
            q1_prev = q_prev[1, :]
            q2_prev = q_prev[2, :]
            q3_prev = q_prev[3, :]
    
            rev_node98.ext[:objective][:ref_cost] = @objective(rev_node98, Max, sum(
                q1[t] * inner_initial_λ_opt[t] +
                (SP - inner_initial_λ_opt[t]) * res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))] -
                penalty_weight * (q1[t] - q1_prev[t])^2
                for t in T))
    
            rev_node99.ext[:objective][:ref_cost] = @objective(rev_node99, Max, sum(
                q2[t] * inner_initial_λ_opt[t] +
                (SP - inner_initial_λ_opt[t]) * res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))] -
                penalty_weight * (q2[t] - q2_prev[t])^2
                for t in T))
    
            rev_node103.ext[:objective][:ref_cost] = @objective(rev_node103, Max, sum(
                q3[t] * inner_initial_λ_opt[t] +
                (SP - inner_initial_λ_opt[t]) * res_batch_avg[min(ceil(Int, t / 30), length(res_batch_avg))] -
                penalty_weight * (q3[t] - q3_prev[t])^2
                for t in T))
        end

    else
        error("Unsupported CfD type.")
    end

    # Optimize each model
    optimize!(rev_node98)
    optimize!(rev_node99)
    optimize!(rev_node103)

    # Combine results into a (3, length(T)) matrix
    q1_values = [value(q1[t]) for t in T]
    q2_values = [value(q2[t]) for t in T]
    q3_values = [value(q3[t]) for t in T]
    q_values = hcat(q1_values, q2_values, q3_values)'
    return q_values
end

function iterative_market_clearing!(a_iter::Model, updated_RES::Matrix{Float64})

    read_data = a.ext[:loops][:read_data]
    # Initialize custom fields for a_iter
    a_iter.ext[:variables] = Dict()
    a_iter.ext[:expressions] = Dict()
    a_iter.ext[:constraints] = Dict()
    a_iter.ext[:objective] = Dict()

    # Extract sets
    N = a.ext[:sets][:N]
    G = a.ext[:sets][:G]
    Z = a.ext[:sets][:Z]
    L = a.ext[:sets][:L]
    L_DC = a.ext[:sets][:L_DC]
    T = a.ext[:sets][:T]
    H = a.ext[:sets][:H]

    # Extract parameters
    GENCAP = a.ext[:parameters][:GENCAP]
    RES = updated_RES
    MC = a.ext[:parameters][:MC]
    DEM = a.ext[:parameters][:DEM]
    NPTDF = a.ext[:parameters][:NPTDF]
    n_in_z = a.ext[:parameters][:n_in_z]
    g_in_n = a.ext[:parameters][:g_in_n]
    TC = a.ext[:parameters][:TC]
    TC_DC = a.ext[:parameters][:TC_DC]
    v_bc = a.ext[:parameters][:v_bc]
    p_bc = a.ext[:parameters][:p_bc]
    F_bc = a.ext[:parameters][:F_bc]
    GSK = a.ext[:parameters][:GSK]
    inc_dc = a.ext[:parameters][:inc_dc]
    i = a.ext[:parameters][:i]
    i_DC = a.ext[:parameters][:i_DC]
    border_ACDC = a.ext[:parameters][:border_ACDC] 
    CC = a.ext[:parameters][:CC]
    ZPTDF = a.ext[:parameters][:ZPTDF_bc] 
    RAM_pos = a.ext[:parameters][:RAM_pos_bc] 
    RAM_neg = a.ext[:parameters][:RAM_neg_bc] 
    WTP = a.ext[:parameters][:WTP]
    H2CAP = a.ext[:parameters][:H2CAP]
    h2_in_n = a.ext[:parameters][:h2_in_n]
    h2_in_z = a.ext[:parameters][:h2_in_z]
 
    if read_data == "5-node_OBZ" || read_data == "5-node_HM"
        NTC_35 = a.ext[:parameters][:NTC_35]
        NTC_45 = a.ext[:parameters][:NTC_45]

    elseif read_data == "Schonheit_HM" || read_data == "Schonheit_OBZ" || read_data == "Schonheit_OBZ_adjusted"
        NTC_12, NTC_23, NTC_34, NTC_15, NTC_25, NTC_35, NTC_45 = a.ext[:parameters][:NTC_12], a.ext[:parameters][:NTC_23], a.ext[:parameters][:NTC_34], a.ext[:parameters][:NTC_15], a.ext[:parameters][:NTC_25], a.ext[:parameters][:NTC_35], a.ext[:parameters][:NTC_45]    

    elseif read_data == "Reference_Case"
        NTC_12 = a.ext[:parameters][:NTC_12]
        NTC_23 = a.ext[:parameters][:NTC_23]
        NTC_13 = a.ext[:parameters][:NTC_13]
        NTC_14 = a.ext[:parameters][:NTC_14]
        NTC_24 = a.ext[:parameters][:NTC_24]
        NTC_34 = a.ext[:parameters][:NTC_34]   
        
    elseif read_data == "Simple_Hybrid"
        NTC_12 = a.ext[:parameters][:NTC_12]
        NTC_23 = a.ext[:parameters][:NTC_23]
        NTC_13 = a.ext[:parameters][:NTC_13]
        NTC_14 = a.ext[:parameters][:NTC_14]
        NTC_34 = a.ext[:parameters][:NTC_34]
    end


    # Create variables
    v = a_iter.ext[:variables][:v] = @variable(a_iter, [g in G, t in T], lower_bound = 0, upper_bound = 1, base_name = "dispatch")
    curt = a_iter.ext[:variables][:curt] = @variable(a_iter, [n in N, t in T], lower_bound = 0, base_name = "curtailment")
    p = a_iter.ext[:variables][:p] = @variable(a_iter, [z in Z, t in T], base_name = "total position")
    p_FB = a_iter.ext[:variables][:p_FB] = @variable(a_iter, [z in Z, t in T], base_name = "flow-based position")
    F_FBMC = a_iter.ext[:variables][:F_FBMC] = @variable(a_iter, [l in L, t in T], base_name = "commercial flow")
    F_DC = a_iter.ext[:variables][:F_DC] = @variable(a_iter, [l_dc in L_DC, t in T], base_name = "commercial DC flow")
    e = a_iter.ext[:variables][:e] = @variable(a_iter, [h in H, t in T], lower_bound = 0, upper_bound = 1, base_name = "dispatch electrolyser")

    # Create expressions
    F = a_iter.ext[:expressions][:F] = @expression(a_iter, [l in L, t in T], sum(NPTDF[findfirst(N .== n),l]*(sum(GENCAP[g]*(v[g,t]) for g in g_in_n[n]) + (RES[t,findfirst(N .== n)]-curt[n,t])- DEM[t,findfirst(N .== n)] - sum(H2CAP[h]*e[h,t] for h in h2_in_n[n]) - sum(F_DC[l_dc,t]*inc_dc[l_dc,findfirst(N .== n)] for l_dc in L_DC)) for n in N ))
    
    Flow_FBMC = a_iter.ext[:expressions][:Flow_FBMC] = @expression(a_iter, [l in L, t in T], sum(ZPTDF[z,l]*p_FB[z,t] for z in Z) + sum(F_DC[l_dc,t] * (- NPTDF[findfirst(N .== df_DC[l_dc,:FromBus]),l]*df_DC[l_dc,:FromDirection] - NPTDF[findfirst(N .== df_DC[l_dc,:ToBus]),l]*df_DC[l_dc,:ToDirection]) for l_dc in L_DC))   #Middle part of constraint 4b (Kenis, 2023) --> AC flow restriction

    # GC_per_time_and_zone = a.ext[:expressions][:GC_per_zone] = @expression(a, [z in Z, t in T], sum(sum(MC[g] * GENCAP[g] * v[g, t] for g in g_in_n[n]) + CC * curt[n, t] - sum(WTP[h] * H2CAP[h] * e[h, t] for h in h2_in_z[z]) for n in n_in_z[z]))
    GC_per_time_and_zone = a_iter.ext[:expressions][:GC_per_zone] = @expression(a_iter, [z in Z, t in T], sum(sum(MC[g]*GENCAP[g]*v[g,t] for g in g_in_n[n]) + CC*curt[n,t] - sum(WTP[h]*H2CAP[h]*e[h,t] for h in h2_in_n[n]) for n in n_in_z[z])) #update with h2_in_n
    # GC_per_time_and_zone = a.ext[:expressions][:GC_per_zone] = @expression(a, [z in Z, t in T], sum(sum(MC[g]*GENCAP[g]*v[g,t] for g in g_in_n[n]) + CC*curt[n,t] for n in n_in_z[z])) #Old
   


    # ZPTDF = a.ext[:expressions][:ZPTDF] = @expression(a, [z in Z, l in L], sum(NPTDF[findfirst(N .== n),l]*GSK[findfirst(N .== n),z] for n in N ))
    # RAM_pos = a.ext[:expressions][:RAM_pos] = @expression(a, [l in L, t in T], TC[l] - (F_bc[l,t] - sum(ZPTDF[z,l] * p_bc[z,t] for z in Z )) )
    # RAM_neg = a.ext[:expressions][:RAM_neg] = @expression(a, [l in L, t in T], TC[l] + (F_bc[l,t] - sum(ZPTDF[z,l] * p_bc[z,t] for z in Z )) )
    
    # Objective
    GC = a_iter.ext[:objective][:GC] = @objective(a_iter, Min, sum(sum(MC[g]*GENCAP[g]*v[g,t] for g in g_in_n[n]) + CC*curt[n,t] -  sum(WTP[h]*H2CAP[h]*e[h,t] for h in h2_in_n[n]) for t in T for n in N)) #Objective function 1a (Kenis, 2023)

    a_iter.ext[:constraints][:conn1] = @constraint(a_iter, [z in Z, t in T], sum(sum(GENCAP[g]*v[g,t] for g in g_in_n[n]) + (RES[t,findfirst(N .== n)] - curt[n, t]) - DEM[t,findfirst(N .== n)] - sum(H2CAP[h]*e[h,t] for h in h2_in_n[n]) for n in n_in_z[z]) - p[z,t]  == 0)#Constraint 1d: Power balance (Kenis, 2023)

    a_iter.ext[:constraints][:conn2] = @constraint(a_iter, [z in Z, t in T], sum(F_FBMC[l,t]*i[l,z] for l in L) + sum(F_DC[l_dc,t]*i_DC[l_dc,z] for l_dc in L_DC) - p[z,t] == 0 ) #Constraint 4a (Kenis, 2023)
    a_iter.ext[:constraints][:conn3] = @constraint(a_iter, [z in Z, t in T], p_FB[z,t] == p[z,t] - sum(F_DC[l_dc,t]*i_DC[l_dc,z] for l_dc in L_DC)) #Constraint 4a (Kenis, 2023)
    a_iter.ext[:constraints][:conn4] = @constraint(a_iter, [l in L, t in T], -RAM_neg[l,t] <= Flow_FBMC[l,t] <= RAM_pos[l,t] ) #Completion of constraint 4b (Kenis, 2023) --> AC flow restriction
    a_iter.ext[:constraints][:conn5] = @constraint(a_iter, [n in N, t in T], curt[n,t] <= RES[t,findfirst(N .== n)] )
    #Constraint 1c upper bound (Kenis, 2023)

    if read_data == "5-node_HM" || read_data == "5-node_OBZ"
        #enter laws of Kirchoff ---- NEW ---    
       a_iter.ext[:constraints][:conn6] = @constraint(a_iter, [t in T], -F_DC[1,t]-(RES[t,findfirst(N .== 5)]-curt[5,t]) + (H2CAP[1]*e[1,t]) - F_DC[2,t] == 0)
       a_iter.ext[:constraints][:conn7] = @constraint(a_iter, [t in T], -NTC_35 <= F_DC[1,t] <= NTC_35) #limit flow on cross-border line
       a_iter.ext[:constraints][:conn8] = @constraint(a_iter, [t in T], -NTC_45 <= F_DC[2,t] <= NTC_45) #limit flow on cross-border line

   elseif read_data == "Schonheit_HM" || read_data == "Schonheit_OBZ" || read_data == "Schonheit_OBZ_adjusted"
       #enter laws of Kirchoff --- OLD ---    Constraint 4c (Kenis, 2023)
       #Radially connected wind farm
       a_iter.ext[:constraints][:conn6] = @constraint(a_iter, [t in T], -F_DC[1,t]-(RES[t,findfirst(N .== 122)]-curt[122,t]) == 0)

       #Hybrids with triangle setup
       a_iter.ext[:constraints][:conn7] = @constraint(a_iter, [t in T], -F_DC[2,t]-(RES[t,findfirst(N .== 119)]-curt[119,t]) 
    #    + (H2CAP[1]*e[1,t]) 
       +F_DC[8,t] +F_DC[7,t] == 0)

       a_iter.ext[:constraints][:conn8] = @constraint(a_iter, [t in T], -F_DC[5,t]-(RES[t,findfirst(N .== 120)]-curt[120,t]) 
    #    + (H2CAP[2]*e[2,t]) 
       -F_DC[7,t] +F_DC[9,t] == 0)

       a_iter.ext[:constraints][:conn9] = @constraint(a_iter, [t in T], -F_DC[6,t]-(RES[t,findfirst(N .== 124)]-curt[124,t]) 
    #    + (H2CAP[3]*e[3,t]) 
       -F_DC[8,t] -F_DC[9,t] == 0)

       #hybrid wind farm 2 OWFs 1 interco
       a_iter.ext[:constraints][:conn10] = @constraint(a_iter, [t in T], -F_DC[3,t]-(RES[t,findfirst(N .== 121)]-curt[121,t]) +F_DC[10,t] == 0)
       a_iter.ext[:constraints][:conn11] = @constraint(a_iter, [t in T], -F_DC[4,t]-(RES[t,findfirst(N .== 123)]-curt[123,t]) -F_DC[10,t] == 0)

       # #    NTC values -- OLD ---
       a_iter.ext[:constraints][:conn12] = @constraint(a_iter, [t in T], -NTC_12 <= F_DC[11,t] <= NTC_12) 
       a_iter.ext[:constraints][:conn13] = @constraint(a_iter, [t in T], -NTC_23 <= -F_DC[12,t] <= NTC_23)
       a_iter.ext[:constraints][:conn14] = @constraint(a_iter, [t in T], -NTC_34 <= F_DC[13,t] <= NTC_34)
       a_iter.ext[:constraints][:conn15] = @constraint(a_iter, [t in T], -NTC_15 <= F_DC[2,t] <= NTC_15) 
       a_iter.ext[:constraints][:conn16] = @constraint(a_iter, [t in T], -NTC_25 <= F_DC[1,t]+F_DC[6,t] <= NTC_25) 
       a_iter.ext[:constraints][:conn17] = @constraint(a_iter, [t in T], -NTC_35 <= F_DC[4,t]+F_DC[5,t] <= NTC_35) 
       a_iter.ext[:constraints][:conn18] = @constraint(a_iter, [t in T], -NTC_45 <= F_DC[3,t] <= NTC_45) 

   elseif read_data =="Reference_Case"
       #enter laws of Kirchoff --- OLD ---    Constraints [2i] 
       #Radially connected wind farms
       a_iter.ext[:constraints][:conn6] = @constraint(a_iter, [t in T], -F_DC[1,t]-(RES[t,findfirst(N .== 122)]-curt[122,t]) == 0)
       a_iter.ext[:constraints][:conn7] = @constraint(a_iter, [t in T], -F_DC[3,t]-(RES[t,findfirst(N .== 121)]-curt[121,t]) == 0)
       a_iter.ext[:constraints][:conn8] = @constraint(a_iter, [t in T], -F_DC[4,t]-(RES[t,findfirst(N .== 123)]-curt[123,t]) == 0)

       #Hybrid wind farms with triangle setup
       a_iter.ext[:constraints][:conn9] = @constraint(a_iter, [t in T], -F_DC[2,t]-(RES[t,findfirst(N .== 119)]-curt[119,t]) 
    #    + (H2CAP[1]*e[1,t]) 
       +F_DC[8,t] +F_DC[7,t] == 0)

       a_iter.ext[:constraints][:conn10] = @constraint(a_iter, [t in T], -F_DC[5,t]-(RES[t,findfirst(N .== 120)]-curt[120,t]) 
    #    + (H2CAP[1]*e[1,t]) 
       -F_DC[7,t] +F_DC[9,t] == 0)

       a_iter.ext[:constraints][:conn11] = @constraint(a_iter, [t in T], -F_DC[6,t]-(RES[t,findfirst(N .== 124)]-curt[124,t]) 
    #    + (H2CAP[1]*e[1,t]) 
       -F_DC[8,t] -F_DC[9,t] == 0)
       
       #Constraints [2f]: limit flow on cross-border line
       a_iter.ext[:constraints][:conn12] = @constraint(a_iter, [t in T], -NTC_12 <= F_DC[10,t] <= NTC_12) 
       a_iter.ext[:constraints][:conn13] = @constraint(a_iter, [t in T], -NTC_23 <= -F_DC[11,t] <= NTC_23)
       a_iter.ext[:constraints][:conn14] = @constraint(a_iter, [t in T], -NTC_13 <= F_DC[12,t] <= NTC_13) 
       a_iter.ext[:constraints][:conn15] = @constraint(a_iter, [t in T], -NTC_14 <= F_DC[2,t] <= NTC_14) 
       a_iter.ext[:constraints][:conn16] = @constraint(a_iter, [t in T], -NTC_24 <= F_DC[6,t] <= NTC_24) 
       a_iter.ext[:constraints][:conn17] = @constraint(a_iter, [t in T], -NTC_34 <= F_DC[5,t] <= NTC_34)

       #Constraints to limit flow on internal OBZ lines to stay within limits of TC
       a_iter.ext[:constraints][:conn18] = @constraint(a_iter, [t in T], -TC[7] <= F_DC[7,t] <= TC[7])
       a_iter.ext[:constraints][:conn19] = @constraint(a_iter, [t in T], -TC[8] <= F_DC[8,t] <= TC[8])
       a_iter.ext[:constraints][:conn20] = @constraint(a_iter, [t in T], -TC[9] <= F_DC[9,t] <= TC[9])
       
    elseif read_data == "Simple_Hybrid"     #no H2 included in the constraints!
        #enter laws of Kirchoff Constraints [2i] 
        #Radially connected wind farms
        a_iter.ext[:constraints][:conn6] = @constraint(a_iter, [t in T], -F_DC[1,t]-(RES[t,findfirst(N .== 122)]-curt[122,t]) == 0)
        a_iter.ext[:constraints][:conn7] = @constraint(a_iter, [t in T], -F_DC[3,t]-(RES[t,findfirst(N .== 121)]-curt[121,t]) == 0)
        a_iter.ext[:constraints][:conn8] = @constraint(a_iter, [t in T], -F_DC[4,t]-(RES[t,findfirst(N .== 123)]-curt[123,t]) == 0)

        #hybrid wind farm 2 OWFs 1 interco
        a_iter.ext[:constraints][:conn9] = @constraint(a_iter, [t in T], -F_DC[2,t]-(RES[t,findfirst(N .== 119)]-curt[119,t]) +F_DC[6,t] == 0)
        a_iter.ext[:constraints][:conn10] = @constraint(a_iter, [t in T], -F_DC[5,t]-(RES[t,findfirst(N .== 120)]-curt[120,t]) +F_DC[6,t] == 0)

        #Constraints [2f]: limit flow on cross-border line
        a_iter.ext[:constraints][:conn11] = @constraint(a_iter, [t in T], -NTC_12 <= F_DC[7,t] <= NTC_12) 
        a_iter.ext[:constraints][:conn12] = @constraint(a_iter, [t in T], -NTC_23 <= -F_DC[8,t] <= NTC_23)
        a_iter.ext[:constraints][:conn13] = @constraint(a_iter, [t in T], -NTC_13 <= F_DC[9,t] <= NTC_13) 
        a_iter.ext[:constraints][:conn14] = @constraint(a_iter, [t in T], -NTC_14 <= F_DC[2,t] <= NTC_14) 
        a_iter.ext[:constraints][:conn15] = @constraint(a_iter, [t in T], -NTC_34 <= F_DC[5,t] <= NTC_34) 

        #Constraints to limit flow on internal OBZ lines to stay within limits of TC
        a_iter.ext[:constraints][:conn16] = @constraint(a_iter, [t in T], -TC[6] <= F_DC[6,t] <= TC[6])
    end
    return a_iter
end

# Step 3: Reinitialise a_iter
a_iter = Model(Gurobi.Optimizer)
set_optimizer_attribute(a_iter, "OutputFlag", 1)
set_optimizer_attribute(a_iter, "InfUnbdInfo", 1)


function main_optimization_loop(a_iter::Model, T::Vector{Int64}, zone_4_node_indices::Vector{Int64}, initial_λ_opt::Vector{Float64}, offshore_gen_zone_4::Matrix{Float64};
    SP::Float64, CfD_type::Symbol)

    convergence_tolerance = 0.1
    max_iterations = 60  # Combined maximum iterations
    iteration = 0
    converged = false
    max_price_diff = Inf
    final_q_values = DataFrame()

    # Initialize lists to store intermediate values
    q_values_list = []  # To store intermediate q values (volumes)
    lambda_opt_list = []  # To store intermediate λ_opt values (prices)

    # Start with the initial λ_opt
    current_λ_opt = initial_λ_opt
    updated_RES = copy(a.ext[:parameters][:RES])  # Copy initial RES to modify

    curt_DA_aiter = nothing
    while !converged && iteration < max_iterations
        iteration += 1
        println("Iteration: $iteration")
    
        # Only use step control from iteration 2 onwards
        q_prev = iteration > 1 ? Matrix(q_values_list[end]) : nothing

        # Step 1: Revenue optimisation
        final_q_values = optimize_revenue(zone_4_node_indices, T, updated_RES, current_λ_opt, SP, CfD_type, q_prev, 0.2) #20% step size limit
        push!(q_values_list, deepcopy(final_q_values))
        
        # Step 2: Update RES
        for t in 1:length(T)
            for i in 1:length(zone_4_node_indices)
                updated_RES[t, zone_4_node_indices[i]] = final_q_values[i, t]
            end
        end
        # Step 2: Update RES
        #jump_limit = 0.5  # Maximum allowed 50% change
        #smooth_factor = 0.8  # Gradual adjustment factor

        #for t in 1:length(T)
        #    for i in 1:length(zone_4_node_indices)
        #        old_value = updated_RES[t, zone_4_node_indices[i]]
        #        new_value = final_q_values[i, t]
        #
        #        if abs(new_value - old_value) > jump_limit * old_value  # If change > 50%
        #            if new_value > old_value  # If increasing
        #                global updated_RES[t, zone_4_node_indices[i]] = old_value + jump_limit * old_value  # Limit increase
        #            else  # If decreasing
        #                global updated_RES[t, zone_4_node_indices[i]] = old_value * smooth_factor  # Gradual drop
        #            end
        #        else
        #            global updated_RES[t, zone_4_node_indices[i]] = new_value  # Accept normal update
        #       end
        #    end
        #end

        # Step 3: Reinitialise a_iter
        a_iter = Model(Gurobi.Optimizer)
        set_optimizer_attribute(a_iter, "OutputFlag", 1)
        set_optimizer_attribute(a_iter, "InfUnbdInfo", 1)
    
        # Step 4: Rebuild and optimise a_iter
        iterative_market_clearing!(a_iter, updated_RES)
        #println("Keys in a_iter.ext[:constraints]: ", keys(a_iter.ext[:constraints]))
    
        optimize!(a_iter)
        if termination_status(a_iter) != MOI.OPTIMAL
            error("Market clearing optimisation failed.")
        end
    
        # Step 5: Update λ_opt
        if haskey(a_iter.ext[:constraints], :conn1)
            updated_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
            push!(lambda_opt_list, deepcopy(updated_λ_opt))
            max_price_diff = maximum(abs.(current_λ_opt .- updated_λ_opt))
            println("  Max price difference: $max_price_diff")
        else
            error(":conn1 not found in a_iter.ext[:constraints].")
        end
        curt_DA_aiter = value.(a_iter.ext[:variables][:curt])

        if max_price_diff < convergence_tolerance
            converged = true
            println("Converged after $iteration iterations.")
        else
            current_λ_opt = updated_λ_opt
        end
    end
    return final_q_values, iteration, q_values_list, lambda_opt_list, curt_DA_aiter
end

# Main execution

df_offshore_gen_zone_4 = DataFrame(offshore_gen_zone_4, :auto)
max_price_diff = Inf

# Initialize lists in Main scope to capture intermediate values

global q_values_list = []
global lambda_opt_list = []

# Assuming PPA and CfD_type are keyword arguments:
#final_q_values, total_outer_iterations, total_inner_iterations, q_values_list, lambda_opt_list = main_optimization_loop(a, T, zone_4_node_indices, initial_λ_opt, df_offshore_gen_zone_4; PPA=20.0, CfD_type=:two_sided)

# Pass PPA_intervals and interval_size to the main optimization loop

final_q_values, iterations, q_values_list, lambda_opt_list, curt_DA_aiter = main_optimization_loop(
    a_iter, T, zone_4_node_indices, initial_λ_opt, offshore_gen_zone_4, SP = 1920.0, CfD_type=:capability_based_q3_noCfD)

println("Final Q Values: ", final_q_values)
println("Total Iterations: ", iterations)
println("Q Values List: ", q_values_list)
println("Lambda Opt List: ", lambda_opt_list)

q_values_list
q_trends = [q[node_index, timestep_index] for q in q_values_list]
q1_trends = [q[1, :] for q in q_values_list]
q2_trends = [q[2, :] for q in q_values_list]
q3_trends = [q[3, :] for q in q_values_list]
plot(T, q1_trends, label="Node 1", xlabel="Time Step", ylabel="quantity (€/MWh)", title="quantity for Different Nodes")
plot(T, q2_trends, label="Node 2")
plot(T, q3_trends, label="Node 3")

node_index = 1        # node number (1, 2, or 3)
zone_4_node_indices = [98, 99, 103]  # Mapping from local node 1–3 to RES columns
timestep_index = 359  # timestep (1 to 744)

# Extract data avoiding name conflicts
node_values = [q_values_list[i][node_index, timestep_index] for i in 1:length(q_values_list)]
#price_values = [lambda_opt_list[i][node_index] for i in 1:length(lambda_opt_list)]

plot(title="Bid Output at Timestep $timestep_index ",
     xlabel="Scenario Index",
     ylabel="Quantity (MW)")

for node_index in 1:3
    res_node = zone_4_node_indices[node_index]

    # RES value at this timestep
    res_value = RES[timestep_index, res_node]

    # Extract bid values for this node and timestep across all scenarios
    bid_series = [q_values_list[i][node_index, timestep_index] for i in 1:length(q_values_list)]

    # Prepend RES value
    full_series = vcat(res_value, bid_series)

    # Plot: x goes from 0 (RES) to number of scenarios
    plot!(0:length(bid_series), full_series,
          label="Node $res_node",
          marker=:circle,
          linewidth=2)
end

savefig("E:/TU Delft/Thesis/Code/Results/Pics/test_q3nocfd.png")

q_values_list  
plot(1:length(q_values_list), node_values,
     xlabel="Scenario Index",
     ylabel="Value",
     title="Quantities and Prices at Node $node_index, Timestep $timestep_index",
     label="Quantity (MW)",
     marker=:circle,
     linewidth=2)

plot!(1:length(lambda_opt_list), price_values,
      label="Price (€/MWh)",
      marker=:star,
      linestyle=:dash,
      linewidth=2)

num_nodes = 3
num_timesteps = 744
num_scenarios = length(q_values_list)

println("Checking for variation in values across 100 scenarios...\n")

for node_index in 1:num_nodes
    for timestep_index in 1:num_timesteps
        # Extract values for this node and timestep across all 100 scenarios
        node_values = [q_values_list[i][node_index, timestep_index] for i in 1:num_scenarios]
        
        # Check if there's any variation
        if length(unique(node_values)) > 1
            println("⚠️  Value changes at Node $node_index, Timestep $timestep_index")
        end
    end
end

println("\n✅ Done checking all node-timestep combinations.")


updated_RES[:,98] .= final_q_values[1,:]
updated_RES[:,99] .= final_q_values[2,:]
updated_RES[:,103] .= final_q_values[3,:]


c1_RES = updated_RES[:,[98,99,103]]
c2_RES = updated_RES[:,[98,99,103]]
c2b_RES = updated_RES[:,[98,99,103]]
c3_RES = updated_RES[:,[98,99,103]]
c4a_RES = updated_RES[:,[98,99,103]]
c4b_RES = updated_RES[:,[98,99,103]]
c5_RES = updated_RES[:,[98,99,103]]
c6_RES = updated_RES[:,[98,99,103]]
c7_RES = updated_RES[:,[98,99,103]]
c7b_RES = updated_RES[:,[98,99,103]]
c7c_RES = updated_RES[:,[98,99,103]]
c8_RES = updated_RES[:,[98,99,103]]
c9_RES = updated_RES[:,[98,99,103]]
c10_RES = updated_RES[:,[98,99,103]]
c11_RES = updated_RES[:,[98,99,103]]
c12_RES = updated_RES[:,[98,99,103]]
c13_RES = updated_RES[:,[98,99,103]]
c14_RES = updated_RES[:,[98,99,103]]
c15_RES = updated_RES[:,[98,99,103]]
c16_RES = updated_RES[:,[98,99,103]]
c17_RES = updated_RES[:,[98,99,103]]
c18_RES = updated_RES[:,[98,99,103]]
c19_RES = updated_RES[:,[98,99,103]]
c20_RES = updated_RES[:,[98,99,103]]
c21_RES = updated_RES[:,[98,99,103]]
c22_RES = updated_RES[:,[98,99,103]]
c23_RES = updated_RES[:,[98,99,103]]
c24_RES = updated_RES[:,[98,99,103]]

difff = c11_RES .- c13_RES
maximum(difff)

a_iter = Model(Gurobi.Optimizer)
set_optimizer_attribute(a_iter, "OutputFlag", 1)
set_optimizer_attribute(a_iter, "InfUnbdInfo", 1)
iterative_market_clearing!(a_iter, updated_RES)
status = optimize!(a_iter)

CfD_type = :capability_based_sp10000 # Set the CfD type for the current iteration

if CfD_type == :simple
    c1_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c1_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c1_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c1_available_power = c1_RES .- c1_curtailment
    c1_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c1_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c1_position = Matrix(value.(a_iter.ext[:variables][:p]))
    c1_bid = c1_RES

elseif CfD_type == :two_sided
    c2_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c2_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c2_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c2_available_power = c2_RES .- c2_curtailment
    c2_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c2_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c2_position = Matrix(value.(a_iter.ext[:variables][:p]))

elseif CfD_type == :two_sided_yearly_avg
    c2b_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c2b_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c2b_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c2b_available_power = c2b_RES .- c2b_curtailment
    c2b_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c2b_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c2b_position = Matrix(value.(a_iter.ext[:variables][:p]))

elseif CfD_type == :cap_floor
    c3_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c3_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c3_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c3_available_power = c3_RES .- c3_curtailment
    c3_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c3_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c3_position = Matrix(value.(a_iter.ext[:variables][:p]))

elseif CfD_type == :capability_based_avg
    c4a_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c4a_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c4a_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c4a_available_power = c4a_RES .- c4a_curtailment
    c4a_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c4a_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c4a_position = Matrix(value.(a_iter.ext[:variables][:p]))

elseif CfD_type == :capability_based_avgb #This is where each res value is the same. since it is in the same area
    c4b_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c4b_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c4b_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c4b_available_power = c4b_RES .- c4b_curtailment
    c4b_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c4b_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c4b_position = Matrix(value.(a_iter.ext[:variables][:p]))

elseif CfD_type == :capability_based_avg2 
    c5_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c5_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c5_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c5_available_power = c5_RES .- c5_curtailment
    c5_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c5_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c5_position = Matrix(value.(a_iter.ext[:variables][:p]))

elseif CfD_type == :capability_based_avg3
    c6_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c6_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c6_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c6_available_power = c6_RES .- c6_curtailment
    c6_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c6_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c6_position = Matrix(value.(a_iter.ext[:variables][:p]))

elseif CfD_type == :capability_based_node98
    c7_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c7_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c7_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c7_available_power = c7_RES .- c7_curtailment
    c7_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c7_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c7_position = Matrix(value.(a_iter.ext[:variables][:p]))

elseif CfD_type == :capability_based_node99
    c7b_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c7b_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c7b_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c7b_available_power = c7b_RES .- c7b_curtailment
    c7b_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c7b_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c7b_position = Matrix(value.(a_iter.ext[:variables][:p]))

elseif CfD_type == :capability_based_node103
    c7c_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c7c_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c7c_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c7c_available_power = c7c_RES .- c7c_curtailment
    c7c_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c7c_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c7c_position = Matrix(value.(a_iter.ext[:variables][:p]))

elseif CfD_type == :capability_based_scenario5_2
    c8_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c8_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c8_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c8_available_power = c8_RES .- c8_curtailment
    c8_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c8_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c8_position = Matrix(value.(a_iter.ext[:variables][:p]))

elseif CfD_type == :capability_based_sp5000
    c9_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c9_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c9_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c9_available_power = c9_RES .- c9_curtailment
    c9_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c9_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c9_position = Matrix(value.(a_iter.ext[:variables][:p]))

elseif CfD_type == :capability_based_sp1920
    c10_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c10_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c10_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c10_available_power = c12_RES .- c10_curtailment
    c10_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c10_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c10_position = Matrix(value.(a_iter.ext[:variables][:p]))

elseif CfD_type == :capability_based_q3_noCfD
    c11_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c11_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c11_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c11_available_power = c11_RES .- c11_curtailment
    c11_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c11_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c11_position = Matrix(value.(a_iter.ext[:variables][:p]))
    #c11_bids = Matrix(value.(a_iter.ext[:variables][:bids]))

elseif CfD_type == :capability_based_q1_noCfD
    c12_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c12_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c12_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c12_available_power = c12_RES .- c12_curtailment
    c12_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c12_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c12_position = Matrix(value.(a_iter.ext[:variables][:p]))

elseif CfD_type == :capability_based_q2_noCfD
    c13_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c13_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c13_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c13_available_power = c13_RES .- c13_curtailment
    c13_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c13_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c13_position = Matrix(value.(a_iter.ext[:variables][:p]))

elseif CfD_type == :capability_based_q1_CfD
    c14_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c14_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c14_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c14_available_power = c14_RES .- c14_curtailment
    c14_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c14_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c14_position = Matrix(value.(a_iter.ext[:variables][:p]))

elseif CfD_type == :capability_based_q2_CfD
    c15_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c15_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c15_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c15_available_power = c15_RES .- c15_curtailment
    c15_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c15_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c15_position = Matrix(value.(a_iter.ext[:variables][:p]))

elseif CfD_type == :capability_based_q3_CfD
    c16_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c16_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c16_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c16_available_power = c16_RES .- c16_curtailment
    c16_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c16_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c16_position = Matrix(value.(a_iter.ext[:variables][:p]))

    elseif CfD_type == :capability_based_sp10000
    c17_λ_opt = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1])[4, :])
    c17_prices = Array(JuMP.dual.(a_iter.ext[:constraints][:conn1]))
    c17_curtailment = permutedims(Matrix(value.(a_iter.ext[:variables][:curt])[[119,120,124],:]))
    c17_available_power = c17_RES .- c17_curtailment
    c17_electrolyser = Matrix(value.(a_iter.ext[:variables][:e]))
    c17_DC_flow = Matrix(value.(a_iter.ext[:variables][:F_DC]))
    c17_position = Matrix(value.(a_iter.ext[:variables][:p]))

end

maximum(c4a_λ_opt .- 41.7)

timee = 1:744
plot(timee, c4a_λ_opt, label="Node 1", xlabel="Time Step", ylabel="quantity (€/MWh)", title="quantity for Different Nodes")

#-------------------------------------------------------------------------------#
