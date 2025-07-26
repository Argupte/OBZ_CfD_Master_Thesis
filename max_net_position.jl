using JuMP
using Gurobi

x = Model(Gurobi.Optimizer)

OBZ_zone = 4      #change this number to indicate which zone is the OBZ zone (1-indexed!)

base_folder_results = a.ext[:loops][:base_folder_results]
read_data = a.ext[:loops][:read_data]

function capacity_calculation(a::Model)
    
    x.ext[:variables] = Dict()
    x.ext[:expressions] = Dict()
    x.ext[:constraints] = Dict()
    x.ext[:objective] = Dict()

    # Extract sets
    N = a.ext[:sets][:N]        
    Z = a.ext[:sets][:Z]
    L = a.ext[:sets][:L]
    L_DC = a.ext[:sets][:L_DC]
    T = a.ext[:sets][:T]

    # Extract parameters
    RAM_pos = a.ext[:parameters][:RAM_pos_bc]    
    RAM_neg = a.ext[:parameters][:RAM_neg_bc]
    ZPTDF = a.ext[:parameters][:ZPTDF_bc]
    NPTDF = a.ext[:parameters][:NPTDF]
    i = a.ext[:parameters][:i]
    i_DC = a.ext[:parameters][:i_DC]

    #Same adjustment as in the D-1 market clearing model: to add the NTC values indicated in the data.jl file
    if read_data == "5-node_OBZ" || read_data == "5-node_HM"
        NTC_35 = a.ext[:parameters][:NTC_35]
        NTC_45 = a.ext[:parameters][:NTC_45]
    
    elseif read_data == "Schonheit_HM" || read_data == "Schonheit_OBZ" || read_data == "Schonheit_OBZ_adjusted"
        NTC_12, NTC_23, NTC_34, NTC_15, NTC_25, NTC_35, NTC_45 = x.ext[:parameters][:NTC_12], x.ext[:parameters][:NTC_23], x.ext[:parameters][:NTC_34], x.ext[:parameters][:NTC_15], a.ext[:parameters][:NTC_25], x.ext[:parameters][:NTC_35], x.ext[:parameters][:NTC_45]   

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
    p = x.ext[:variables][:p] = @variable(x, [z in Z, t in T], base_name = "zonal position")
    p_FB = x.ext[:variables][:p_FB] = @variable(x, [z in Z, t in T], base_name = "flow-based position")
    F_FBMC = x.ext[:variables][:F_FBMC] = @variable(x, [l in L, t in T], base_name = "commercial flow")
    F_DC = x.ext[:variables][:F_DC] = @variable(x, [l_dc in L_DC, t in T], base_name = "commercial DC flow")

    #Expressions
    Flow_FBMC = x.ext[:expressions][:Flow_FBMC] = @expression(x, [l in L, t in T], sum(ZPTDF[z,l]*p_FB[z,t] for z in Z) + sum(F_DC[l_dc,t] * (- NPTDF[findfirst(N .== df_DC[l_dc,:FromBus]),l]*df_DC[l_dc,:FromDirection] - NPTDF[findfirst(N .== df_DC[l_dc,:ToBus]),l]*df_DC[l_dc,:ToDirection]) for l_dc in L_DC))

    #Create Objective: Maximize zonal position
    MxNP = x.ext[:objective][:MxNP] = @objective(x, Max, sum(p[OBZ_zone, t] for t in T))

    #Constraints
    x.ext[:constraints][:con1] = @constraint(x, con1[t in T], sum(p[z, t] for z in Z) == 0)     #Cons [1]: balance for all zonal net positions
    x.ext[:constraints][:con2] = @constraint(x, con2[z in Z, t in T], sum(F_FBMC[l,t]*i[l,z] for l in L) + sum(F_DC[l_dc,t]*i_DC[l_dc,z] for l_dc in L_DC) - p[z,t] == 0 ) #Cons [2]: Total AC flow and DC flow per zone equals zonal net position
    x.ext[:constraints][:con3] = @constraint(x, con3[z in Z, t in T], p_FB[z,t] == p[z,t] - sum(F_DC[l_dc,t]*i_DC[l_dc,z] for l_dc in L_DC)) #Cons [3]: Defines Net position FB of each zone z that  impacts the FB domain
    x.ext[:constraints][:con4] = @constraint(x, con4[l in L, t in T], -RAM_neg[l,t] <= Flow_FBMC[l,t] <= RAM_pos[l,t] ) #Cons [4] impact of DC line on the flow on AC line is added and must be within the limits of the RAM values. 

    #constr [5]: the The flow on the cross-border DC lines must be within the limits of the NTC values (Adjust if grid topology has changed)
    if read_data =="5-node_OBZ" || read_data == "5-node_HM"
        x.ext[:constraints][:con5] = @constraint(x, con5[t in T], -NTC_35 <= F_DC[1,t] <= NTC_35) 
        x.ext[:constraints][:con6] = @constraint(x, con6[t in T], -NTC_45 <= F_DC[2,t] <= NTC_45) 

    elseif read_data == "Schonheit_HM" || read_data == "Schonheit_OBZ" || read_data == "Schonheit_OBZ_adjusted"
        #Constraints 4f: limit flow on cross-border line
        x.ext[:constraints][:con5] = @constraint(x, con5[t in T], -NTC_12 <= F_DC[11,t] <= NTC_12) 
        x.ext[:constraints][:con6] = @constraint(x, con6[t in T], -NTC_23 <= -F_DC[12,t] <= NTC_23) 
        x.ext[:constraints][:con7] = @constraint(x, con7[t in T], -NTC_34 <= F_DC[13,t] <= NTC_34) 
        x.ext[:constraints][:con8] = @constraint(x, con8[t in T], -NTC_15 <= F_DC[2,t] <= NTC_15) 
        x.ext[:constraints][:con9] = @constraint(x, con9[t in T], -NTC_25 <= F_DC[1,t]+F_DC[6,t] <= NTC_25) 
        x.ext[:constraints][:con10] = @constraint(x, con10[t in T], -NTC_35 <= F_DC[4,t]+F_DC[5,t] <= NTC_35) 
        x.ext[:constraints][:con11] = @constraint(x, con11[t in T], -NTC_45 <= F_DC[3,t] <= NTC_45) 

    elseif read_data == "Reference_Case"
        #Constraints [4f]: limit flow on cross-border line
        x.ext[:constraints][:con5] = @constraint(x, con5[t in T], -NTC_12 <= F_DC[10,t] <= NTC_12) 
        x.ext[:constraints][:con6] = @constraint(x, con6[t in T], -NTC_23 <= -F_DC[11,t] <= NTC_23)
        x.ext[:constraints][:con7] = @constraint(x, con7[t in T], -NTC_13 <= F_DC[12,t] <= NTC_13) 
        x.ext[:constraints][:con8] = @constraint(x, con8[t in T], -NTC_14 <= F_DC[2,t] <= NTC_14) 
        x.ext[:constraints][:con9] = @constraint(x, con9[t in T], -NTC_24 <= F_DC[6,t] <= NTC_24) 
        x.ext[:constraints][:con10] = @constraint(x, con10[t in T], -NTC_34 <= F_DC[5,t] <= NTC_34) 

    elseif read_data == "Simple_Hybrid"
        x.ext[:constraints][:con5] = @constraint(x, con5[t in T], -NTC_12 <= F_DC[7,t] <= NTC_12) 
        x.ext[:constraints][:con6] = @constraint(x, con6[t in T], -NTC_23 <= -F_DC[8,t] <= NTC_23)
        x.ext[:constraints][:con7] = @constraint(x, con7[t in T], -NTC_13 <= F_DC[9,t] <= NTC_13) 
        x.ext[:constraints][:con8] = @constraint(x, con8[t in T], -NTC_14 <= F_DC[2,t] <= NTC_14) 
        x.ext[:constraints][:co9] = @constraint(x, con9[t in T], -NTC_34 <= F_DC[5,t] <= NTC_34) 
    end
    return x        
end

capacity_calculation(a)
@info "Capacity Calculation Optimization model built"
       
# write_to_file(x, joinpath(base_folder_results, "model_check/Cap_calc_pre_opt.lp"))        ##Specific model formulation for debugging purposes

status = optimize!(x)

if status == MOI.OPTIMAL
    @info "Optimal solution found"
    # write_to_file(x, joinpath(base_folder_results, "model_check/Cap_calc_post_opt.lp"))
else
    @warn "No optimal solution found"
end


#Extract the results
# MxNP_OBZ = x.ext[:parameters][:MxNP_OBZ] = value.(x.ext[:variables][:max_p])
p_domain = a.ext[:parameters][:p_domain] = value.(x.ext[:variables][:p])
p_FB_domain = a.ext[:parameters][:p_FB_domain] = value.(x.ext[:variables][:p_FB])
F_FBMC_domain = a.ext[:parameters][:F_FBMC_domain] = value.(x.ext[:variables][:F_FBMC])
F_DC_domain = a.ext[:parameters][:F_DC_domain] = value.(x.ext[:variables][:F_DC])

ZPTDF = a.ext[:parameters][:ZPTDF_bc]
F_bc = a.ext[:parameters][:F_bc] = value.(m.ext[:variables][:F])

RAM_pos_max, RAM_neg_max = zeros(length(L), length(T)), zeros(length(L), length(T))
for t in T
    for l in L
        RAM_pos_max[l,t] = min(TC[l] - (F_bc[l,t] - sum(ZPTDF[z,l] * p_domain[z,t] for z in Z )), MinRAM*TC[l])
        RAM_neg_max[l,t] = min(TC[l] + (F_bc[l,t] - sum(ZPTDF[z,l] * p_domain[z,t] for z in Z )), MinRAM*TC[l])
    end
end

a.ext[:parameters][:RAM_neg_max] = RAM_neg_max
a.ext[:parameters][:RAM_pos_max] = RAM_pos_max

df_ram_pos = DataFrame(RAM_pos_max, :auto)
df_ram_neg = DataFrame(RAM_neg_max, :auto)
#CSV.write(joinpath(base_folder_results, "FB_domain/RAM_pos_max.csv"), df_ram_pos)
#CSV.write(joinpath(base_folder_results, "FB_domain/RAM_neg_max.csv"), df_ram_neg)