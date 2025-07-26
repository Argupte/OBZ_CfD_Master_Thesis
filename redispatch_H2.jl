b = Model(Gurobi.Optimizer)

set_optimizer_attribute(b, "OutputFlag", 1)
set_optimizer_attribute(b, "InfUnbdInfo", 1)

function redispatch!(a::Model)
    read_data = a.ext[:loops][:read_data]
    b.ext[:variables] = Dict()
    b.ext[:expressions] = Dict()
    b.ext[:constraints] = Dict()
    b.ext[:objective] = Dict()

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
    α = a.ext[:parameters][:α]
    v_DA = a.ext[:parameters][:v_DA] 
    curt_DA = a.ext[:parameters][:curt_DA]
    p_DA = a.ext[:parameters][:p_DA] 
    F_DA = a.ext[:parameters][:F_DA] 
    F_DC_DA = a.ext[:parameters][:F_DC_DA] 
    RES = a.ext[:parameters][:RES]
    CC = a.ext[:parameters][:CC]
    GC_DA = a.ext[:parameters][:GC_DA]
    e_DA = a.ext[:parameters][:e_DA]
    H2CAP = a.ext[:parameters][:H2CAP]
    WTP = a.ext[:parameters][:WTP]
    h2_in_n = a.ext[:parameters][:h2_in_n]
    e_DA = a.ext[:parameters][:e_DA]         #dispatch electrolyser


    # Create variables
    UP = b.ext[:variables][:UP] = @variable(b, [g in G, t in T], lower_bound = 0, base_name = "upward redispatch")
    DOWN = b.ext[:variables][:DOWN] = @variable(b, [g in G, t in T], lower_bound = 0, base_name = "downward redispatch")
    UP_H2 = b.ext[:variables][:UP_H2] = @variable(b, [h in H, t in T], lower_bound = 0, base_name = "upward H2 production")
    DOWN_H2 = b.ext[:variables][:DOWN_H2] = @variable(b, [h in H, t in T], lower_bound = 0, base_name = "downward H2 production")
    curt_delta = b.ext[:variables][:curt_delta] = @variable(b, [n in N, t in T], lower_bound = 0, base_name = "curtailment") 
    F_DC_delta = b.ext[:variables][:F_DC_delta] = @variable(b, [l_dc in L_DC, t in T], base_name = "DC flow adjustment")

    # Create expressions
    F_beforeRD = b.ext[:expressions][:F_beforeRD] = @expression(b, [l in L, t in T], sum(NPTDF[findfirst(N .== n),l]*(sum(GENCAP[g]*(v_DA[g,t]) for g in g_in_n[n]) + (RES[t,findfirst(N .== n)]-curt_DA[n,t]) -DEM[t,findfirst(N .== n)] - sum(F_DC_DA[l_dc,t]*inc_dc[l_dc,findfirst(N .== n)] for l_dc in L_DC)) for n in N )) #Not used, but similar to AC flows from DA clearing
    F_beforeRD = b.ext[:expressions][:F_beforeRD] = @expression(b, [l in L, t in T], sum(NPTDF[findfirst(N .== n),l]*(sum(GENCAP[g]*(v_DA[g,t]) for g in g_in_n[n]) - sum(H2CAP[h]*(e_DA[h,t]) for h in h2_in_n[n]) + (RES[t,findfirst(N .== n)]-curt_DA[n,t]) -DEM[t,findfirst(N .== n)] - sum(F_DC_DA[l_dc,t]*inc_dc[l_dc,findfirst(N .== n)] for l_dc in L_DC)) for n in N )) #Not used, but similar to AC flows from DA clearing
    
    # F_afterRD = b.ext[:expressions][:F_afterRD] = @expression(b, [l in L, t in T], sum(NPTDF[findfirst(N .== n),l]*(sum(GENCAP[g]*(v_DA[g,t]+UP[g,t]-DOWN[g,t]) for g in g_in_n[n]) + (RES[t,findfirst(N .== n)]-curt_DA[n,t]-curt_delta[n,t]) -DEM[t,findfirst(N .== n)] - sum((F_DC_DA[l_dc,t]+F_DC_delta[l_dc,t])*inc_dc[l_dc,findfirst(N .== n)] for l_dc in L_DC)) for n in N ))   #part of 5c (Kenis, 2023) --> AC FLows after RD
    F_afterRD = b.ext[:expressions][:F_afterRD] = @expression(b, [l in L, t in T], sum(NPTDF[findfirst(N .== n),l]*(sum(GENCAP[g]*(v_DA[g,t]+UP[g,t]-DOWN[g,t]) for g in g_in_n[n]) - sum(H2CAP[h]*(e_DA[h,t] + UP_H2[h,t] - DOWN_H2[h,t]) for h in h2_in_n[n]) + (RES[t,findfirst(N .== n)]-curt_DA[n,t]-curt_delta[n,t]) -DEM[t,findfirst(N .== n)] - sum((F_DC_DA[l_dc,t]+F_DC_delta[l_dc,t])*inc_dc[l_dc,findfirst(N .== n)] for l_dc in L_DC)) for n in N )) # + H2

    # RDC_per_zone = b.ext[:expressions][:RDC_per_zone] = @expression(b, [z in Z, t in T], sum(sum((1+α)*UP[g,t]*GENCAP[g]*MC[g]-(1-α)*DOWN[g,t]*GENCAP[g]*MC[g] for g in g_in_n[n]) + CC*curt_delta[n,t] for n in n_in_z[z] ))  #No formula like this in Kenis, 2023
    RDC_per_zone = b.ext[:expressions][:RDC_per_zone] = @expression(b, [z in Z, t in T], sum(sum((1+α)*UP[g,t]*GENCAP[g]*MC[g]-(1-α)*DOWN[g,t]*GENCAP[g]*MC[g] for g in g_in_n[n]) - sum((1+α)*UP_H2[h,t]*H2CAP[h]*WTP[h]-(1-α)*DOWN_H2[h,t]*H2CAP[h]*WTP[h] for h in h2_in_n[n]) + CC*curt_delta[n,t] for n in n_in_z[z] ))  #No formula like this in Kenis, 2023 + H2

   

    # Objective
    RDC = b.ext[:objective][:RDC] = @objective(b, Min, sum( sum((1+α)*UP[g,t]*GENCAP[g]*MC[g]-(1-α)*DOWN[g,t]*GENCAP[g]*MC[g] for g in g_in_n[n]) - sum((1+α)*UP_H2[h,t]*H2CAP[h]*WTP[h]-(1-α)*DOWN_H2[h,t]*H2CAP[h]*WTP[h] for h in h2_in_n[n]) + CC*curt_delta[n,t] for t in T for n in N))  #5a (Kenis, 2023)

    # Constraints 
    # b.ext[:constraints][:con1] = @constraint(b, con1[t in T], sum(sum(UP[g,t]*GENCAP[g] for g in g_in_n[n]) for n in N) == sum(sum(DOWN[g,t]*GENCAP[g] for g in g_in_n[n]) + curt_delta[n,t] for n in N))   #Power balance 5b (Kenis, 2023)

    b.ext[:constraints][:con1] = @constraint(b, [t in T], sum(sum(UP[g,t]*GENCAP[g] for g in g_in_n[n]) - sum(DOWN_H2[h,t]*H2CAP[h] for h in h2_in_n[n]) for n in N) == sum(sum(DOWN[g,t]*GENCAP[g] for g in g_in_n[n]) + sum(UP_H2[h,t]*H2CAP[h] for h in h2_in_n[n]) + curt_delta[n,t] for n in N))

    # b.ext[:constraints][:conh2_1] = @constraint(b, conh2_1[ t in T], sum(sum(UP_H2[h,t]*H2CAP[h] for h in h2_in_n[n]) for n in N) == sum(sum(DOWN_H2[h,t]*H2CAP[h] for h in h2_in_n[n]) + curt_delta[n,t] for n in N)) #constraint 5c lower bound (Kenis, 2023)

    b.ext[:constraints][:con2] = @constraint(b, [l in L, t in T], F_afterRD[l,t] + TC[l] >= 0 ) #constraint 5c lower bound (Kenis, 2023)
    b.ext[:constraints][:con3] = @constraint(b, [l in L, t in T], TC[l] - F_afterRD[l,t] >= 0 ) #constraint 5c upper bound (Kenis, 2023)
    b.ext[:constraints][:con4] = @constraint(b, [g in G, t in T], (1-v_DA[g,t]) - UP[g,t] >= 0 ) #constraint 5e (Kenis, 2023)
    b.ext[:constraints][:con5] = @constraint(b, [g in G, t in T], v_DA[g,t] - DOWN[g,t] >= 0 )  #constraint 5f (Kenis, 2023)
    b.ext[:constraints][:con6] = @constraint(b, [l_dc in L_DC, t in T], TC_DC[l_dc] >= (F_DC_DA[l_dc,t]+F_DC_delta[l_dc,t]) >= -TC_DC[l_dc] ) #constraint 5d (Kenis, 2023)
    b.ext[:constraints][:con7] = @constraint(b, [l_dc in L_DC, t in T], - F_DC_DA[l_dc,t] <= F_DC_delta[l_dc,t] <= TC_DC[l_dc] - F_DC_DA[l_dc,t]) #constraint 5h (Kenis, 2023)
    b.ext[:constraints][:con8] = @constraint(b, [n in N, t in T], -curt_DA[n,t] <= curt_delta[n,t] <= RES[t,findfirst(N .== n)]-curt_DA[n,t] )  #constraint 5g (Kenis, 2023)

    b.ext[:constraints][:conh2_1] = @constraint(b, conh2_1[h in H, t in T], (1-e_DA[h,t]) - UP_H2[h,t] >= 0 ) #NEW H2 constraint
    b.ext[:constraints][:conh2_2] = @constraint(b, conh2_2[h in H, t in T], e_DA[h,t] - DOWN_H2[h,t] >= 0 )  #NEW H2 constraint

    if read_data == "5-node_HM" || read_data == "5-node_OBZ"
        # Laws of Kirchoff in DC-area: This is for the OWFs in the OBZ
        b.ext[:constraints][:con9] = @constraint(b, [t in T], -F_DC_DA[1,t]-F_DC_delta[1,t]-(RES[t,findfirst(N .== 5)]-curt_DA[5,t]-curt_delta[5,t]) + (H2CAP[1]*e_DA[1,t] + H2CAP[1]*UP_H2[1,t] - H2CAP[1]*DOWN_H2[1,t]) - F_DC_DA[2,t] - F_DC_delta[2,t] == 0)     #constraint 5i (Kenis, 2023)

    elseif read_data == "Schonheit_HM" || read_data == "Schonheit_OBZ"  || read_data == "Schonheit_OBZ_adjusted"
        b.ext[:constraints][:con9] = @constraint(b, [t in T], -F_DC_DA[1,t]-F_DC_delta[1,t]-(RES[t,findfirst(N .== 122)]-curt_DA[122,t]-curt_delta[122,t]) == 0)
        b.ext[:constraints][:con10] = @constraint(b, [t in T], -F_DC_DA[2,t]-F_DC_delta[2,t]-(RES[t,findfirst(N .== 119)]-curt_DA[119,t]-curt_delta[119,t]) + (H2CAP[1]*e_DA[1,t] + H2CAP[1]*UP_H2[1,t] - H2CAP[1]*DOWN_H2[1,t]) +F_DC_DA[8,t] +F_DC_DA[7,t] +F_DC_delta[8,t] +F_DC_delta[7,t] == 0)
        b.ext[:constraints][:con11] = @constraint(b, [t in T], -F_DC_DA[5,t]-F_DC_delta[5,t]-(RES[t,findfirst(N .== 120)]-curt_DA[120,t]-curt_delta[120,t]) + (H2CAP[2]*e_DA[2,t] + H2CAP[2]*UP_H2[2,t] - H2CAP[2]*DOWN_H2[2,t]) -F_DC_DA[7,t] +F_DC_DA[9,t] -F_DC_delta[7,t] +F_DC_delta[9,t] == 0)
        b.ext[:constraints][:con12] = @constraint(b, [t in T], -F_DC_DA[6,t]-F_DC_delta[6,t]-(RES[t,findfirst(N .== 124)]-curt_DA[124,t]-curt_delta[124,t]) + (H2CAP[3]*e_DA[3,t] + H2CAP[3]*UP_H2[3,t] - H2CAP[3]*DOWN_H2[3,t]) -F_DC_DA[8,t] -F_DC_DA[9,t] -F_DC_delta[8,t] -F_DC_delta[9,t] == 0)
        b.ext[:constraints][:con13] = @constraint(b, [t in T], -F_DC_DA[3,t]-F_DC_delta[3,t]-(RES[t,findfirst(N .== 121)]-curt_DA[121,t]-curt_delta[121,t]) +F_DC_DA[10,t] +F_DC_delta[10,t] == 0)
        b.ext[:constraints][:con14] = @constraint(b, [t in T], -F_DC_DA[4,t]-F_DC_delta[4,t]-(RES[t,findfirst(N .== 123)]-curt_DA[123,t]-curt_delta[123,t]) -F_DC_DA[10,t] -F_DC_delta[10,t] == 0)

    elseif read_data == "Reference_Case"
        #radially connected wind farms
        b.ext[:constraints][:con9] = @constraint(b, [t in T], -F_DC_DA[1,t]-(RES[t,findfirst(N .== 122)]-curt_DA[122,t]-curt_delta[122,t]) == 0)
        b.ext[:constraints][:con10] = @constraint(b, [t in T], -F_DC_DA[3,t]-(RES[t,findfirst(N .== 121)]-curt_DA[121,t]-curt_delta[121,t]) == 0)
        b.ext[:constraints][:con11] = @constraint(b, [t in T], -F_DC_DA[4,t]-(RES[t,findfirst(N .== 123)]-curt_DA[123,t]-curt_delta[123,t]) == 0)

        #Hybrid wind farms with triangle setup
        b.ext[:constraints][:con12] = @constraint(b, [t in T], -F_DC_DA[2,t]-(RES[t,findfirst(N .== 119)]-curt_DA[119,t]-curt_delta[119,t]) 
        # + (H2CAP[1]*e_DA[1,t] + H2CAP[1]*UP_H2[1,t] - H2CAP[1]*DOWN_H2[1,t]) #Put off in case of onshore electrolyser
        +F_DC_DA[8,t] +F_DC_DA[7,t] == 0)

        b.ext[:constraints][:con13] = @constraint(b, [t in T], -F_DC_DA[5,t]-(RES[t,findfirst(N .== 120)]-curt_DA[120,t]-curt_delta[120,t]) 
        # + (H2CAP[2]*e_DA[2,t] + H2CAP[2]*UP_H2[2,t] - H2CAP[2]*DOWN_H2[2,t]) #idem
        -F_DC_DA[7,t] +F_DC_DA[9,t] == 0)

        b.ext[:constraints][:con14] = @constraint(b, [t in T], -F_DC_DA[6,t]-(RES[t,findfirst(N .== 124)]-curt_DA[124,t]-curt_delta[124,t]) 
        # + (H2CAP[3]*e_DA[3,t] + H2CAP[3]*UP_H2[3,t] - H2CAP[3]*DOWN_H2[3,t])    #idem
        -F_DC_DA[8,t] -F_DC_DA[9,t] == 0)
        
    elseif read_data == "Simple_Hybrid"     #No electrolyser incorporated in the constraints yet
        
        #Radially connected wind farms
        b.ext[:constraints][:con9] = @constraint(b, [t in T], -F_DC_DA[1,t]-(RES[t,findfirst(N .== 122)]-curt_DA[122,t]-curt_delta[122,t]) == 0)
        b.ext[:constraints][:con10] = @constraint(b, [t in T], -F_DC_DA[3,t]-(RES[t,findfirst(N .== 121)]-curt_DA[121,t]-curt_delta[121,t]) == 0)
        b.ext[:constraints][:con11] = @constraint(b, [t in T], -F_DC_DA[4,t]-(RES[t,findfirst(N .== 123)]-curt_DA[123,t]-curt_delta[123,t]) == 0)

        #hybrid wind farm 2 OWFs 1 interco
        b.ext[:constraints][:con12] = @constraint(b, [t in T], -F_DC_DA[2,t]-(RES[t,findfirst(N .== 119)]-curt_DA[119,t]-curt_delta[119,t]) +F_DC_DA[6,t] == 0)
        b.ext[:constraints][:con13] = @constraint(b, [t in T], -F_DC_DA[5,t]-(RES[t,findfirst(N .== 120)]-curt_DA[120,t]-curt_delta[120,t]) -F_DC_DA[6,t] == 0)

    end

    return status
end

# Build your model
redispatch!(a)
@info "Redispatch model built."
# write_to_file(b, joinpath(base_folder_results, "model_check/RD_pre_optimization.lp"))

status = optimize!(b)
@info "Redispatch model optimised."
# write_to_file(b, joinpath(base_folder_results, "model_check/RD_post_optimization.lp"))

#Check if redispatch was necessary, e.g. if model was infeasible or unbounded
if JuMP.termination_status(b) == MOI.INFEASIBLE_OR_UNBOUNDED
    println("Redispatch model is infeasible")
    # write_to_file(b, joinpath(base_folder_results, "model_check/RD_infeasible.lp"))
    
else
    @info "Redispatch model solved optimally."
  
end