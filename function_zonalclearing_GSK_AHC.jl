using JuMP, Gurobi

m = Model(Gurobi.Optimizer)

set_optimizer_attribute(m, "OutputFlag", 1)  # Enable detailed Gurobi output
set_optimizer_attribute(m, "InfUnbdInfo", 1) # Provides more information on infeasibility/unboundedness


#This is the D-2 base case model.
function market_clearing!(a::Model)
    read_data = a.ext[:loops][:read_data]
    m.ext[:variables] = Dict()
    m.ext[:expressions] = Dict()
    m.ext[:constraints] = Dict()
    m.ext[:objective] = Dict()

    # Extract sets
    N = a.ext[:sets][:N]
    G = a.ext[:sets][:G]
    Z = a.ext[:sets][:Z]
    L = a.ext[:sets][:L]
    L_DC = a.ext[:sets][:L_DC]
    T = a.ext[:sets][:T]

    # Extract parameters
    GENCAP = a.ext[:parameters][:GENCAP]
    MC = a.ext[:parameters][:MC]
    DEM = a.ext[:parameters][:DEM]
    NPTDF = a.ext[:parameters][:NPTDF]
    n_in_z = a.ext[:parameters][:n_in_z]
    g_in_n = a.ext[:parameters][:g_in_n]
    TC = a.ext[:parameters][:TC]
    TC_DC = a.ext[:parameters][:TC_DC]
    inc = a.ext[:parameters][:inc] 
    inc_dc = a.ext[:parameters][:inc_dc]
    RES = a.ext[:parameters][:RES]
    CC = a.ext[:parameters][:CC]

    # Create variables
    v = m.ext[:variables][:v] = @variable(m, [g in G, t in T], lower_bound = 0, upper_bound = 1, base_name = "dispatch")    #Constraint 1b (Kenis, 2023)
    curt = m.ext[:variables][:curt] = @variable(m, [n in N, t in T], lower_bound = 0, base_name = "curtailment")            #constraint 1c (Kenis, 2023)
    p = m.ext[:variables][:p] = @variable(m, [n in N, t in T], base_name = "position")
    F = m.ext[:variables][:F] = @variable(m, [l in L, t in T], base_name = "flows")
    F_dc = m.ext[:variables][:F_dc] = @variable(m, [l_dc in L_DC, t in T], base_name = "DC flows")
    
    # Create expressions
    GC_per_time_and_zone = m.ext[:expressions][:GC_per_zone] = @expression(m, [z in Z, t in T], sum(sum(MC[g]*GENCAP[g]*v[g,t] for g in g_in_n[n]) + CC*curt[n,t] for n in n_in_z[z]) )
    p_zonal = m.ext[:expressions][:p_zonal] = @expression(m, [z in Z, t in T], sum(sum(GENCAP[g] * v[g,t] for g in g_in_n[n]) + (RES[t,findfirst(N .== n)]-curt[n,t]) - DEM[t,findfirst(N .== n)] for n in n_in_z[z]))            

    # Objective
    GC = m.ext[:objective][:GC] = @objective(m, Min, sum(sum(MC[g]*GENCAP[g]*v[g,t] for g in g_in_n[n]) +CC*curt[n,t] for t in T for n in N))

    # Constraints 
    m.ext[:constraints][:con1] = @constraint(m, [n in N, t in T], sum(GENCAP[g] * v[g,t] for g in g_in_n[n]) + (RES[t,findfirst(N .== n)]-curt[n,t]) - DEM[t,findfirst(N .== n)] - p[n,t] == 0 )     #Constraint 1d (Kenis, 2023)
    m.ext[:constraints][:con2] = @constraint(m, [n in N, t in T], sum(F[l,t]*inc[l,findfirst(N .== n)] for l in L) + sum(F_dc[l_dc,t]*inc_dc[l_dc,findfirst(N .== n)] for l_dc in L_DC) - p[n,t] == 0 )  #Constraint 1e (Kenis, 2023)
    m.ext[:constraints][:con3] = @constraint(m, [l in L, t in T], F[l,t] == sum(NPTDF[findfirst(N .== n),l]*(sum(GENCAP[g]*v[g,t] for g in g_in_n[n]) + (RES[t,findfirst(N .== n)]-curt[n,t])-DEM[t,findfirst(N .== n)] - sum(F_dc[l_dc,t]*inc_dc[l_dc,findfirst(N .== n)] for l_dc in L_DC)) for n in N )) #Constraint 1d Thesis
    m.ext[:constraints][:con4] = @constraint(m, [l in L, t in T], -TC[l] <= F[l,t] <= TC[l] ) #Constraint 1e

    #con5 use to have F[l_dc,t] instead of F_dc[l_dc,t]
    m.ext[:constraints][:con5] = @constraint(m, [l_dc in L_DC, t in T], -TC_DC[l_dc] <= F_dc[l_dc,t] <= TC_DC[l_dc] )
    m.ext[:constraints][:con6] = @constraint(m, [n in N, t in T], curt[n,t] <= RES[t,findfirst(N .== n)] )


    # If new grid topology is used (new read_data loop input), add here the nodal power balance constraints for OWFs, based on the logic described below.
    if read_data == "Schonheit_HM"|| read_data == "Schonheit_OBZ" || read_data == "Schonheit_OBZ_adjusted"

        #----- OLD-----   
        ###enter laws of Kirchoff
        ##For a radial connected wind farm, follow this logic:
        m.ext[:constraints][:con7] = @constraint(m, [t in T], -F_dc[1,t]-(RES[t,findfirst(N .== 122)]-curt[122,t]) == 0)
        #for the hybrid wind farms triangle setup
        m.ext[:constraints][:con8] = @constraint(m, [t in T], -F_dc[2,t]-(RES[t,findfirst(N .== 119)]-curt[119,t]) +F_dc[8,t] +F_dc[7,t] == 0)
        m.ext[:constraints][:con9] = @constraint(m, [t in T], -F_dc[5,t]-(RES[t,findfirst(N .== 120)]-curt[120,t]) -F_dc[7,t] +F_dc[9,t] == 0)
        m.ext[:constraints][:con10] = @constraint(m, [t in T], -F_dc[6,t]-(RES[t,findfirst(N .== 124)]-curt[124,t]) -F_dc[8,t] -F_dc[9,t] == 0)
        #For hybrid wind farms dual setup
        m.ext[:constraints][:con11] = @constraint(m, [t in T], -F_dc[3,t]-(RES[t,findfirst(N .== 121)]-curt[121,t]) +F_dc[10,t] == 0)
        m.ext[:constraints][:con12] = @constraint(m, [t in T], -F_dc[4,t]-(RES[t,findfirst(N .== 123)]-curt[123,t]) -F_dc[10,t] == 0)
    
    elseif read_data == "5-node_OBZ" || read_data == "5-node_HM"
        #-------NEW------
        m.ext[:constraints][:con7] = @constraint(m, [t in T], -F_dc[1,t]-(RES[t,findfirst(N .== 5)]-curt[5,t]) - F_dc[2,t] == 0)


    elseif read_data == "Reference_Case"
        m.ext[:constraints][:con7] = @constraint(m, [t in T], -F_dc[1,t]-(RES[t,findfirst(N .== 122)]-curt[122,t]) == 0)
        m.ext[:constraints][:con8] = @constraint(m, [t in T], -F_dc[3,t]-(RES[t,findfirst(N .== 121)]-curt[121,t]) == 0)
        m.ext[:constraints][:con9] = @constraint(m, [t in T], -F_dc[4,t]+ (RES[t,findfirst(N .== 123)]-curt[123,t]) == 0)

        #Hybrid wind farms with triangle setup
        m.ext[:constraints][:con10] = @constraint(m, [t in T], -F_dc[2,t]-(RES[t,findfirst(N .== 119)]-curt[119,t]) +F_dc[8,t] +F_dc[7,t] == 0)
        m.ext[:constraints][:con11] = @constraint(m, [t in T], -F_dc[5,t]-(RES[t,findfirst(N .== 120)]-curt[120,t]) -F_dc[7,t] +F_dc[9,t] == 0)
        m.ext[:constraints][:con12] = @constraint(m, [t in T], -F_dc[6,t]-(RES[t,findfirst(N .== 124)]-curt[124,t]) -F_dc[8,t] -F_dc[9,t] == 0)

        #Interal OBZ lines to stay within limits
        m.ext[:constraints][:con13] = @constraint(m, [t in T], -TC_DC[10] <= F_dc[10,t] <= TC_DC[10])
        m.ext[:constraints][:con14] = @constraint(m, [t in T], -TC_DC[11] <= F_dc[11,t] <= TC_DC[11])
        m.ext[:constraints][:con15] = @constraint(m, [t in T], -TC_DC[12] <= F_dc[12,t] <= TC_DC[12])

    elseif read_data == "Simple_Hybrid"
        m.ext[:constraints][:con7] = @constraint(m, [t in T], -F_dc[1,t]-(RES[t,findfirst(N .== 122)]-curt[122,t]) == 0)
        m.ext[:constraints][:con8] = @constraint(m, [t in T], -F_dc[3,t]-(RES[t,findfirst(N .== 121)]-curt[121,t]) == 0)
        m.ext[:constraints][:con9] = @constraint(m, [t in T], -F_dc[4,t]-(RES[t,findfirst(N .== 123)]-curt[123,t]) == 0)
          
        #hybrid wind farm 2 OWFs 1 interco
        m.ext[:constraints][:con10] = @constraint(m, [t in T], -F_dc[2,t]-(RES[t,findfirst(N .== 119)]-curt[119,t]) + F_dc[6,t] == 0)
        m.ext[:constraints][:con11] = @constraint(m, [t in T], -F_dc[5,t]-(RES[t,findfirst(N .== 120)]-curt[120,t]) - F_dc[6,t] == 0)

        #Internal OBZ line to stay within limits
        m.ext[:constraints][:con12] = @constraint(m, [t in T], -TC[6] <= F_dc[7,t] <= TC[6])	
    end
  
    return m
end

# Build your model
market_clearing!(a)
@info "Market clearing base case model built."
# write_to_file(m, joinpath(base_folder_results, "model_check/base_case_pre_optimization.lp"))

status = optimize!(m)

# Write to file before optimization if needed
#base_folder_results = a.ext[:loops][:base_folder_results]

#write_to_file(a, joinpath(base_folder_results, "base_case_AHC_pre_optimization.lp"))

#if termination_status(a) == MOI.INFEASIBLE_OR_UNBOUNDED
 #   println("Model is infeasible or unbounded. Further diagnostics needed.")
  #  write_to_file(a, joinpath(base_folder_results, "base_case_AHC_infeasible.lp"))
#else
   # println("Model solved successfully.")
    #write_to_file(a, joinpath(base_folder_results, "base_case_AHC_post_optimization.lp"))
#end


#if termination_status(m) == MOI.INFEASIBLE_OR_UNBOUNDED
        #println("Model is infeasible or unbounded. Further diagnostics needed.")
        #write_to_file(m, joinpath(base_folder_results, "model_check/base_case_infeasible.lp"))
        
#else
 #       println("Model pre-solved successfully.")
 #      # write_to_file(m, joinpath(base_folder_results, "model_check/base_case_post_optimization.lp"))
#end


GC_bc = a.ext[:parameters][:GC_DA] = value.(m.ext[:expressions][:GC_per_zone])
v_bc = a.ext[:parameters][:v_bc] = value.(m.ext[:variables][:v])
curt_bc = a.ext[:parameters][:curt_bc] = value.(m.ext[:variables][:curt])
p_bc = a.ext[:parameters][:p_bc] = value.(m.ext[:expressions][:p_zonal])
F_bc = a.ext[:parameters][:F_bc] = value.(m.ext[:variables][:F])
F_dc_bc = a.ext[:parameters][:F_DC_DA] = value.(m.ext[:variables][:F_dc])

set_optimizer_attribute(a, "OutputFlag", 1)
set_optimizer_attribute(a, "InfUnbdInfo", 1)

function get_GSK!(a::Model)     #The GSK calculation strategy is pro rata based on the installed generating capacity 
    a.ext[:variables] = Dict()
    a.ext[:expressions] = Dict()
    a.ext[:constraints] = Dict()
    a.ext[:objective] = Dict()

    # Extract sets
    N = a.ext[:sets][:N]
    G = a.ext[:sets][:G]
    Z = a.ext[:sets][:Z]
    L = a.ext[:sets][:L]
    T = a.ext[:sets][:T]

    # Extract parameters
    GENCAP = a.ext[:parameters][:GENCAP]
    n_in_z = a.ext[:parameters][:n_in_z]
    g_in_n = a.ext[:parameters][:g_in_n]
    all_g_in_n = a.ext[:parameters][:all_g_in_n]

    gsk = zeros(length(N),length(Z))
    for n in N
        zone_temp = df_node.Zone[df_node[:,:BusID].==n][1]
        nodes_in_zone = n_in_z[zone_temp]
        if n in nodes_in_zone
            pmax_in_zone = sum(df_gen.Pmax[[x in nodes_in_zone for x in df_gen[:,:OnBus]]])
            pmax_at_node = sum(df_gen.Pmax[[x == n for x in df_gen[:,:OnBus]]])
            gsk[findfirst(N .== n),zone_temp] = pmax_at_node/pmax_in_zone
        end
    end
    
    a.ext[:parameters][:GSK] = gsk

    return a 
end

get_GSK!(a)
GSK_pmax = get_GSK!(a).ext[:parameters][:GSK]
sum(GSK_pmax, dims=1)
# println(GSK_pmax)

ZPTDF_bc, RAM_pos_bc, RAM_neg_bc = zeros(length(Z), length(L)), zeros(length(L), length(T)), zeros(length(L), length(T))
for z in Z
    for l in L
        ZPTDF_bc[z,l] = sum(NPTDF[findfirst(N .== n),l]*a.ext[:parameters][:GSK][findfirst(N .== n),z] for n in N )
    end
end 
for t in T
    for l in L
        RAM_pos_bc[l,t] = max(TC[l] - (F_bc[l,t] - sum(ZPTDF_bc[z,l] * p_bc[z,t] for z in Z )), MinRAM*TC[l])
        RAM_neg_bc[l,t] = max(TC[l] + (F_bc[l,t] - sum(ZPTDF_bc[z,l] * p_bc[z,t] for z in Z )), MinRAM*TC[l])
    end
end
a.ext[:parameters][:ZPTDF_bc] = ZPTDF_bc
a.ext[:parameters][:RAM_pos_bc] = RAM_pos_bc 
a.ext[:parameters][:RAM_neg_bc] = RAM_neg_bc

set_optimizer_attribute(a, "OutputFlag", 1)  # Enable detailed Gurobi output
set_optimizer_attribute(a, "InfUnbdInfo", 1)


#This is the D-1 market clearing model
function market_clearing!(a::Model)
    read_data = a.ext[:loops][:read_data]
    a.ext[:variables] = Dict()
    a.ext[:expressions] = Dict()
    a.ext[:constraints] = Dict()
    a.ext[:objective] = Dict()

    # Extract sets
    N = a.ext[:sets][:N]
    G = a.ext[:sets][:G]
    Z = a.ext[:sets][:Z]
    L = a.ext[:sets][:L]
    L_DC = a.ext[:sets][:L_DC]
    T = a.ext[:sets][:T]

    # Extract parameters
    GENCAP = a.ext[:parameters][:GENCAP]
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
    RES = a.ext[:parameters][:RES]
    CC = a.ext[:parameters][:CC]
    ZPTDF = a.ext[:parameters][:ZPTDF_bc] 
    RAM_pos = a.ext[:parameters][:RAM_pos_bc] 
    RAM_neg = a.ext[:parameters][:RAM_neg_bc] 

    # If new grid topology is used (new read_data loop input), add here the NTC values as created in the data.jl file
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
    v = a.ext[:variables][:v] = @variable(a, [g in G, t in T], lower_bound = 0, upper_bound = 1, base_name = "dispatch")
    curt = a.ext[:variables][:curt] = @variable(a, [n in N, t in T], lower_bound = 0, base_name = "curtailment")
    p = a.ext[:variables][:p] = @variable(a, [z in Z, t in T], base_name = "total position")
    p_FB = a.ext[:variables][:p_FB] = @variable(a, [z in Z, t in T], base_name = "flow-based position")
    F_FBMC = a.ext[:variables][:F_FBMC] = @variable(a, [l in L, t in T], base_name = "commercial AC flow")
    F_DC = a.ext[:variables][:F_DC] = @variable(a, [l_dc in L_DC, t in T], base_name = "commercial DC flow")

    # Create expressions
    F = a.ext[:expressions][:F] = @expression(a, [l in L, t in T], sum(NPTDF[findfirst(N .== n),l]*(sum(GENCAP[g]*(v[g,t]) for g in g_in_n[n]) + (RES[t,findfirst(N .== n)]-curt[n,t])-DEM[t,findfirst(N .== n)] - sum(F_DC[l_dc,t]*inc_dc[l_dc,findfirst(N .== n)] for l_dc in L_DC)) for n in N ))

    Flow_FBMC = a.ext[:expressions][:Flow_FBMC] = @expression(a, [l in L, t in T], sum(ZPTDF[z,l]*p_FB[z,t] for z in Z) + sum(F_DC[l_dc,t] * (- NPTDF[findfirst(N .== df_DC[l_dc,:FromBus]),l]*df_DC[l_dc,:FromDirection] - NPTDF[findfirst(N .== df_DC[l_dc,:ToBus]),l]*df_DC[l_dc,:ToDirection]) for l_dc in L_DC))
   
    # ZPTDF = a.ext[:expressions][:ZPTDF] = @expression(a, [z in Z, l in L], sum(NPTDF[findfirst(N .== n),l]*GSK[findfirst(N .== n),z] for n in N )) #CHECK IF THIS GIVES DIFFERENT OUTCOMES THAN ALREADY COMPUTED BEFOREHAND!!
    # RAM_pos = a.ext[:expressions][:RAM_pos] = @expression(a, [l in L, t in T], TC[l] - (F_bc[l,t] - sum(ZPTDF[z,l] * p_bc[z,t] for z in Z )) )
    # RAM_neg = a.ext[:expressions][:RAM_neg] = @expression(a, [l in L, t in T], TC[l] + (F_bc[l,t] - sum(ZPTDF[z,l] * p_bc[z,t] for z in Z )) )
    GC_per_time_and_zone = a.ext[:expressions][:GC_per_zone] = @expression(a, [z in Z, t in T], sum(sum(MC[g]*GENCAP[g]*v[g,t] for g in g_in_n[n]) + CC*curt[n,t] for n in n_in_z[z]) )
    
    # Objective
    GC = a.ext[:objective][:GC] = @objective(a, Min, sum(sum(MC[g]*GENCAP[g]*v[g,t] for g in g_in_n[n]) +CC*curt[n,t] for t in T for n in N)) #Objective function 1a (+curtailment costs) (Kenis, 2023)

    # Constraints 
    a.ext[:constraints][:con1] = @constraint(a, [z in Z, t in T], sum(sum(GENCAP[g] * v[g,t] for g in g_in_n[n]) + (RES[t,findfirst(N .== n)]-curt[n,t]) - DEM[t,findfirst(N .== n)] for n in n_in_z[z]) - p[z,t] == 0 ) #Constraint 1d: Power balance (Kenis, 2023)
    a.ext[:constraints][:con2] = @constraint(a, [z in Z, t in T], sum(F_FBMC[l,t]*i[l,z] for l in L) + sum(F_DC[l_dc,t]*i_DC[l_dc,z] for l_dc in L_DC) - p[z,t] == 0 ) #Constraint 1e (Kenis, 2023)
    a.ext[:constraints][:con3] = @constraint(a, [z in Z, t in T], p_FB[z,t] == p[z,t] - sum(F_DC[l_dc,t]*i_DC[l_dc,z] for l_dc in L_DC)) #Constraint 4a (Kenis, 2023)
    a.ext[:constraints][:con4] = @constraint(a, [l in L, t in T], -RAM_neg[l,t] <= Flow_FBMC[l,t] <= RAM_pos[l,t] ) #Completion of constraint 4b (Kenis, 2023) --> AC flow restriction
    a.ext[:constraints][:con5] = @constraint(a, [n in N, t in T], curt[n,t] <= RES[t,findfirst(N .== n)] ) #Constraint 1c upper bound (Kenis, 2023)


    #Same adjustment as in the D-2 base case model (offshore nodal power balances). 
    if read_data == "5-node_HM" || read_data == "5-node_OBZ"
         #enter laws of Kirchoff ---- NEW ---    
        a.ext[:constraints][:con6] = @constraint(a, [t in T], -F_DC[1,t]-(RES[t,findfirst(N .== 5)]-curt[5,t]) - F_DC[2,t] == 0)
        a.ext[:constraints][:con7] = @constraint(a, [t in T], -NTC_35 <= F_DC[1,t] <= NTC_35) #limit flow on cross-border line
        a.ext[:constraints][:con8] = @constraint(a, [t in T], -NTC_45 <= F_DC[2,t] <= NTC_45) #limit flow on cross-border line

    elseif read_data == "Schonheit_HM" || read_data == "Schonheit_OBZ" || read_data == "Schonheit_OBZ_adjusted" 
        #enter laws of Kirchoff --- OLD ---    Constraint 4c (Kenis, 2023)
        #Radially connected wind farm
        a.ext[:constraints][:con6] = @constraint(a, [t in T], -F_DC[1,t]-(RES[t,findfirst(N .== 122)]-curt[122,t]) == 0)

        #Hybrids with triangle setup
        a.ext[:constraints][:con7] = @constraint(a, [t in T], -F_DC[2,t]-(RES[t,findfirst(N .== 119)]-curt[119,t]) +F_DC[8,t] +F_DC[7,t] == 0)
        a.ext[:constraints][:con8] = @constraint(a, [t in T], -F_DC[5,t]-(RES[t,findfirst(N .== 120)]-curt[120,t]) -F_DC[7,t] +F_DC[9,t] == 0)
        a.ext[:constraints][:con9] = @constraint(a, [t in T], -F_DC[6,t]-(RES[t,findfirst(N .== 124)]-curt[124,t]) -F_DC[8,t] -F_DC[9,t] == 0)

        #hybrid wind farm 2 OWFs 1 interco
        a.ext[:constraints][:con10] = @constraint(a, [t in T], -F_DC[3,t]-(RES[t,findfirst(N .== 121)]-curt[121,t]) +F_DC[10,t] == 0)
        a.ext[:constraints][:con11] = @constraint(a, [t in T], -F_DC[4,t]-(RES[t,findfirst(N .== 123)]-curt[123,t]) -F_DC[10,t] == 0)

        # #    NTC values -- OLD ---
        a.ext[:constraints][:con12] = @constraint(a, [t in T], -NTC_12 <= F_DC[11,t] <= NTC_12) 
        a.ext[:constraints][:con13] = @constraint(a, [t in T], -NTC_23 <= -F_DC[12,t] <= NTC_23)
        a.ext[:constraints][:con14] = @constraint(a, [t in T], -NTC_34 <= F_DC[13,t] <= NTC_34) 
        a.ext[:constraints][:con15] = @constraint(a, [t in T], -NTC_15 <= F_DC[2,t] <= NTC_15) 
        a.ext[:constraints][:con16] = @constraint(a, [t in T], -NTC_25 <= F_DC[1,t]+F_DC[6,t] <= NTC_25) 
        a.ext[:constraints][:con17] = @constraint(a, [t in T], -NTC_35 <= F_DC[4,t]+F_DC[5,t] <= NTC_35) 
        a.ext[:constraints][:con18] = @constraint(a, [t in T], -NTC_45 <= F_DC[3,t] <= NTC_45) 

    elseif read_data =="Reference_Case"
        #enter laws of Kirchoff  Constraints [2i] 
        #Radially connected wind farms
        a.ext[:constraints][:con6] = @constraint(a, [t in T], -F_DC[1,t]-(RES[t,findfirst(N .== 122)]-curt[122,t]) == 0)
        a.ext[:constraints][:con7] = @constraint(a, [t in T], -F_DC[3,t]-(RES[t,findfirst(N .== 121)]-curt[121,t]) == 0)
        a.ext[:constraints][:con8] = @constraint(a, [t in T], -F_DC[4,t]-(RES[t,findfirst(N .== 123)]-curt[123,t]) == 0)

        #Hybrid wind farms with triangle setup
        a.ext[:constraints][:con9] = @constraint(a, [t in T], -F_DC[2,t]-(RES[t,findfirst(N .== 119)]-curt[119,t]) +F_DC[8,t] +F_DC[7,t] == 0)
        a.ext[:constraints][:con10] = @constraint(a, [t in T], -F_DC[5,t]-(RES[t,findfirst(N .== 120)]-curt[120,t]) -F_DC[7,t] +F_DC[9,t] == 0)
        a.ext[:constraints][:con11] = @constraint(a, [t in T], -F_DC[6,t]-(RES[t,findfirst(N .== 124)]-curt[124,t]) -F_DC[8,t] -F_DC[9,t] == 0)

        #Constraints [2f]: limit flow on cross-border line
        a.ext[:constraints][:con12] = @constraint(a, [t in T], -NTC_12 <= F_DC[10,t] <= NTC_12) 
        a.ext[:constraints][:con13] = @constraint(a, [t in T], -NTC_23 <= -F_DC[11,t] <= NTC_23)
        a.ext[:constraints][:con14] = @constraint(a, [t in T], -NTC_13 <= F_DC[12,t] <= NTC_13) 
        a.ext[:constraints][:con15] = @constraint(a, [t in T], -NTC_14 <= F_DC[2,t] <= NTC_14) 
        a.ext[:constraints][:con16] = @constraint(a, [t in T], -NTC_24 <= F_DC[6,t] <= NTC_24) 
        a.ext[:constraints][:con17] = @constraint(a, [t in T], -NTC_34 <= F_DC[5,t] <= NTC_34) 

        # #Constraints to limit flow on internal OBZ lines to stay within limits of TC
        a.ext[:constraints][:con18] = @constraint(a, [t in T], -TC[7] <= F_DC[7,t] <= TC[7])
        a.ext[:constraints][:con19] = @constraint(a, [t in T], -TC[8] <= F_DC[8,t] <= TC[8])
        a.ext[:constraints][:con20] = @constraint(a, [t in T], -TC[9] <= F_DC[9,t] <= TC[9])

    elseif read_data == "Simple_Hybrid"
        #enter laws of Kirchoff Constraints [2i] 
        #Radially connected wind farms
        a.ext[:constraints][:con6] = @constraint(a, [t in T], -F_DC[1,t]-(RES[t,findfirst(N .== 122)]-curt[122,t]) == 0)
        a.ext[:constraints][:con7] = @constraint(a, [t in T], -F_DC[3,t]-(RES[t,findfirst(N .== 121)]-curt[121,t]) == 0)
        a.ext[:constraints][:con8] = @constraint(a, [t in T], -F_DC[4,t]+(RES[t,findfirst(N .== 123)]-curt[123,t]) == 0)

        #hybrid wind farm 2 OWFs 1 interco
        a.ext[:constraints][:con9] = @constraint(a, [t in T], -F_DC[2,t]-(RES[t,findfirst(N .== 119)]-curt[119,t]) + F_DC[6,t] == 0)
        a.ext[:constraints][:con10] = @constraint(a, [t in T], -F_DC[5,t]-(RES[t,findfirst(N .== 120)]-curt[120,t]) - F_DC[6,t] == 0)

        #Constraints [2f]: limit flow on cross-border line
        a.ext[:constraints][:con11] = @constraint(a, [t in T], -NTC_12 <= F_DC[7,t] <= NTC_12) 
        a.ext[:constraints][:con12] = @constraint(a, [t in T], -NTC_23 <= F_DC[8,t] <= NTC_23)
        a.ext[:constraints][:con13] = @constraint(a, [t in T], -NTC_13 <= F_DC[9,t] <= NTC_13) 
        a.ext[:constraints][:con14] = @constraint(a, [t in T], -NTC_14 <= F_DC[2,t] <= NTC_14) 
        a.ext[:constraints][:con15] = @constraint(a, [t in T], -NTC_34 <= F_DC[5,t] <= NTC_34) 

         #Constraints to limit flow on internal OBZ lines to stay within limits of TC
        a.ext[:constraints][:con16] = @constraint(a, [t in T], -TC[6] <= F_DC[6,t] <= TC[6])
    end
    return a
end

# Build your model
market_clearing!(a)
@info "Market clearing model built."
# write_to_file(a, joinpath(base_folder_results, "model_check/DA_AHC_pre_optimization.lp"))         ##If you want to check the exact model formulation, put on this comment

status = optimize!(a)
@info "Market clearing model optimised."
# write_to_file(a, joinpath(base_folder_results, "model_check/DA_AHC_post_optimization.lp"))        ##idem

# Write to file before optimization if needed
#write_to_file(a, joinpath(base_folder_results, "DA_AHC_pre_optimization.lp"))

#if termination_status(a) == MOI.INFEASIBLE_OR_UNBOUNDED
 #   println("Model is infeasible or unbounded. Further diagnostics needed.")
  #  write_to_file(a, joinpath(base_folder_results, "DA_AHC_infeasible.lp"))
#else
 #   println("Model solved successfully.")
  #  write_to_file(a, joinpath(base_folder_results, "DA_AHC_post_optimization.lp"))
#end

#Extract values from model for results processsing
GC_DA = a.ext[:parameters][:GC_DA] = value.(a.ext[:expressions][:GC_per_zone])  #generation costs per zone
v_DA = a.ext[:parameters][:v_DA] = value.(a.ext[:variables][:v])                #dispatch
curt_DA = a.ext[:parameters][:curt_DA] = value.(a.ext[:variables][:curt])       #curtailment day-ahead
p_DA = a.ext[:parameters][:p_DA] = value.(a.ext[:variables][:p])                #total position
F_DA = a.ext[:parameters][:F_DA] = value.(a.ext[:expressions][:F])              #Day-ahead AC flow
F_DC_DA = a.ext[:parameters][:F_DC_DA] = value.(a.ext[:variables][:F_DC])       #Day-ahead DC flow
RES_prod = a.ext[:parameters][:RES_prod] = value.(a.ext[:parameters][:RES])     #Renewable production
p_FB_DA = a.ext[:parameters][:p_FB] = value.(a.ext[:variables][:p_FB])          #Flow-based position
p_FBMC_DA = a.ext[:parameters][:F_FBMC] = value.(a.ext[:variables][:F_FBMC])    #commerical flow
flow_FBMC_DA = a.ext[:parameters][:Flow_FBMC] = value.(a.ext[:expressions][:Flow_FBMC])

# ZPTDF = a.ext[:parameters][:ZPTDF] = value.(a.ext[:expressions][:ZPTDF])      #Zone-to-line PTDF

