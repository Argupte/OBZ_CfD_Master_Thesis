using Pkg
using CSV, XLSX, DataFrames
using Plots
using LinearAlgebra

read_data = a.ext[:loops][:read_data]
base_folder_data = a.ext[:loops][:base_folder_data]
base_folder_results = a.ext[:loops][:base_folder_results]
hydrogen = a.ext[:loops][:hydrogen]                          


#Read the data files
global df_node = CSV.read(joinpath(base_folder_data, "node_info.csv"), DataFrame)
global df_gen = CSV.read(joinpath(base_folder_data, "gen_info.csv"),DataFrame)
global df_load = CSV.read(joinpath(base_folder_data, "load_info.csv"), DataFrame)
global df_line = CSV.read(joinpath(base_folder_data, "line_info.csv"), DataFrame)
global df_DC = CSV.read(joinpath(base_folder_data, "DC_info.csv"), DataFrame)
global incidence = CSV.read(joinpath(base_folder_data, "incidence.csv"), DataFrame)
global incidence_dc = CSV.read(joinpath(base_folder_data, "incidence_dc.csv"), DataFrame)
global susceptance = CSV.read(joinpath(base_folder_data, "susceptance.csv"), DataFrame)
global df_pv = CSV.read(joinpath(base_folder_data, "pv.csv"), DataFrame, header=true)
global df_wind = CSV.read(joinpath(base_folder_data, "onshore.csv"), DataFrame, header=true)
global df_wind_off = CSV.read(joinpath(base_folder_data, "offshore.csv"),DataFrame, header=true)

if hydrogen == "yes"
    global df_h2 = CSV.read(joinpath(base_folder_data, "h2_info.csv"), DataFrame, header=true)
end

#Create the sets
a.ext[:sets] = Dict()
N = a.ext[:sets][:N] = df_node[:,:BusID] # Nodes: Read from the df_node 
G = a.ext[:sets][:G] = [g for g = 1:maximum(df_gen[:,:GenID])] # Generators
Z = a.ext[:sets][:Z] = [z for z = 1:maximum(df_node[:,:Zone])] # Zones
L = a.ext[:sets][:L] = [l for l = 1:length(df_line[:,:BranchID])] # Lines
L_DC = a.ext[:sets][:L_DC] = [l for l = 1:length(df_DC[:,:BranchID])] # DC lines
T = a.ext[:sets][:T] = [t for t = 1:size(df_load)[1]] # Time steps
R = ["PV","Wind","Wind Offshore"]
if hydrogen == "yes"
    H = a.ext[:sets][:H] = [h for h = 1:maximum(df_h2[:,:H2ID])]
end

#Merge renewable files into table
df_node.ZoneRes = df_node.Zone
function create_res_table()
	res_temp = zeros(Float64, length(T), length(N), length(R))
	for n in N, r in R
		zone_temp = df_node.ZoneRes[df_node[:,:BusID].==n][1]
		cap_temp = sum(coalesce.(df_gen.Pmax[(df_gen[:,:Type].==r) .&
		                              (df_gen[:,:OnBus].==n)], 0))          #coalesce is a function that replaces missing values with a default value, being 0 in this case
		if r == "PV"
			share_temp = coalesce.(df_pv[:,Symbol.(zone_temp)],0)
		elseif r == "Wind"
			share_temp = coalesce.(df_wind[:,Symbol.(zone_temp)],0)
		else
			share_temp = coalesce.(df_wind_off[:,Symbol.(zone_temp)],0)
		end
		res_temp[:, findfirst(N .== n), findfirst(R .== r)] = cap_temp*share_temp
	end

    res = zeros(Float64, length(T), length(N))
     for n in N
         for t in T
             res_tot = sum(res_temp[t,findfirst(N .== n),r] for r in [1,2,3])
             res[t,findfirst(N .== n)] = res_tot[1]
         end
     end
 	return res
 end

 res = create_res_table()

 function get_renew(t,n,r)
 	return sum(res[findfirst(T .== t), findfirst(N .== n), findfirst(R .== r)] for r in R)
 end

nNodes = length(N) #amount of nodes 
nGenerators = length(G) #amount of generators
nLines = length(L) # number of lines
nLines_DC = length(L_DC) # number of DC lines
nZones = maximum(df_node[:,:Zone]) # number of zones


#Create the parameters
a.ext[:parameters] = Dict()
global GENCAP = a.ext[:parameters][:GENCAP] = df_gen[:,:Pmax]
#global RES = a.ext[:parameters][:RES] = res
RES = a.ext[:parameters][:RES] = Matrix(CSV.read("E:/TU Delft/Thesis/Code/grid_topology/Reference_Case/RES.csv", DataFrame))

LF_PV = a.ext[:parameters][:LF_PV] = Matrix(df_pv)
LF_WINDON = a.ext[:parameters][:WINDON] = Matrix(df_wind)
LF_WINDOFF = a.ext[:parameters][:WINDOFF] = Matrix(df_wind_off)
global MC = a.ext[:parameters][:MC] = df_gen[:,:Costs]
global DEM = a.ext[:parameters][:DEM] = Matrix(df_load)

n_in_z = a.ext[:parameters][:n_in_z] = Dict(map(z -> z => [n for n in N if df_node[df_node[:,:BusID].==n, :Zone][1] == z], Z))

if hydrogen == "yes"
    global H2CAP = a.ext[:parameters][:H2CAP] = df_h2[:,:Pmax]
    global WTP = a.ext[:parameters][:WTP] = df_h2[:,:WTP] 

    h2_in_n = Dict(map(n -> begin
    n_h2 = [h2 for h2 in H if let h2_onbus = df_h2[df_h2[:, :H2ID] .== h2, :OnBus]; !isempty(h2_onbus) && h2_onbus[1] == n end]
    n => n_h2
    end, N))

    a.ext[:parameters][:h2_in_n] = h2_in_n

    h2_in_z = Dict()
    for (z, nodes) in n_in_z
        h2_in_z[z] = unique([h2 for n in nodes for h2 in get(h2_in_n, n, [])])
    end
    a.ext[:parameters][:h2_in_z] = h2_in_z
end

# for (n, ids) in h2_in_n
#     println("Node $n: Electrolysers $ids")
# end

    #Original Code Line:
# g_in_n = a.ext[:parameters][:g_in_n] = Dict(map(n -> n => [g for g in G if df_gen[df_gen[:,:GenID].==g, :OnBus][1] == n && (df_gen[df_gen[:,:GenID].==g, :Type][1] == "Offshore Wind" || df_gen[df_gen[:,:GenID].==g, :Type][1] == "Gas/CCGT" || df_gen[df_gen[:,:GenID].==g, :Type][1] == "Oil" || df_gen[df_gen[:,:GenID].==g, :Type][1] == "Nuclear" || df_gen[df_gen[:,:GenID].==g, :Type][1] == "Biomass" || df_gen[df_gen[:,:GenID].==g, :Type][1] == "Hard Coal")], N))
    #Update Code line to avoid 'empty' array (if no generator is on a node)
g_in_n = a.ext[:parameters][:g_in_n] = Dict(map(n -> begin
    n_gens = [g for g in G if !isempty(df_gen[df_gen[:,:GenID] .== g, :OnBus]) && df_gen[df_gen[:,:GenID] .== g, :OnBus][1] == n && (
        !isempty(df_gen[df_gen[:,:GenID] .== g, :Type]) && (
            df_gen[df_gen[:,:GenID] .== g, :Type][1] == "Offshore Wind" ||
            df_gen[df_gen[:,:GenID] .== g, :Type][1] == "Gas/CCGT" ||
            df_gen[df_gen[:,:GenID] .== g, :Type][1] == "Oil" ||
            df_gen[df_gen[:,:GenID] .== g, :Type][1] == "Nuclear" ||
            df_gen[df_gen[:,:GenID] .== g, :Type][1] == "Biomass" ||
            df_gen[df_gen[:,:GenID] .== g, :Type][1] == "Lignite" ||
            df_gen[df_gen[:,:GenID] .== g, :Type][1] == "Hard Coal"           
        )
    )] 
    n => n_gens
end, N))

     #Original Code Line:
# all_g_in_n = a.ext[:parameters][:all_g_in_n] = Dict(map(n -> n => [g for g in G if df_gen[df_gen[:,:GenID].==g, :OnBus][1] == n], N))
    #Update Code line to avoid 'empty' array (if no generator is on a node)
all_g_in_n = a.ext[:parameters][:all_g_in_n] = Dict(map(n -> begin
    n_gens = [g for g in G if let gens_onbus=df_gen[df_gen[:,:GenID].==g, :OnBus]; !isempty(gens_onbus) && gens_onbus[1] == n end && let gens_type=df_gen[df_gen[:,:GenID].==g, :Type]; !isempty(gens_type) && (gens_type[1] == "Offshore Wind" || gens_type[1] == "Gas/CCGT" || gens_type[1] == "Oil" || gens_type[1] == "Nuclear" || gens_type[1] == "Biomass" || gens_type[1] == "Hard Coal") end]
    n => n_gens
end, N))

l_in_z = a.ext[:parameters][:l_in_z] = Dict(map(z -> z => [l for l in df_line[:,:BranchID] if df_node[findfirst(N .== df_line[findfirst(df_line[:,:BranchID] .== l), :FromBus][1]), :Zone][1] == z || df_node[findfirst(N .== df_line[findfirst(df_line[:,:BranchID] .== l), :ToBus][1]), :Zone][1] == z], Z))
global TC = a.ext[:parameters][:TC] = df_line[:,:Pmax]          #This is the Transfer Capacity of the AC lines
# println("TC: ", TC)
global TC_DC = a.ext[:parameters][:TC_DC] = df_DC[:,:Pmax]      #This is the Transfer Capacity of the DC lines
# println("TC_DC: ", TC_DC)
α = a.ext[:parameters][:α] = 0.10                               #Cost mark-up for redispatch!! value α is set to zero iniitially
CC = a.ext[:parameters][:CC] = 0.00000001                       #Curtailment Costs (e.g. penalty in algoritm for curtailment)
MinRAM = 0.7                                                    #Percentage of Minimum RAM value for the AC lines
NTC_ind = 1                                                     #Index for NTC values --> 1 = 100% of TC_DC is given to the market

#------------------------NTC Determination ----------------------
# Function to calculate and store NTC values based on the grid topology used.
function calculate_ntc_values!(ntc_dict, read_data, TC_DC)
    if read_data == "5-node_OBZ" || read_data == "5-node_HM"
        NTC_35 = a.ext[:parameters][:NTC_35] = TC_DC[1]*NTC_ind 
        ntc_dict[:NTC_35] = (NTC_35, 1)
        
        NTC_45 = a.ext[:parameters][:NTC_45] = TC_DC[2]*NTC_ind 
        ntc_dict[:NTC_45] = (NTC_45, 2)

    elseif read_data == "Schonheit_HM" || read_data == "Schonheit_OBZ" || read_data == "Schonheit_OBZ_adjusted"
        NTC_12 = a.ext[:parameters][:NTC_12] = NTC_ind  * TC_DC[11]
        ntc_dict[:NTC_12] = (NTC_12, 11)
        
        NTC_23 = a.ext[:parameters][:NTC_23] = NTC_ind  * TC_DC[12]
        ntc_dict[:NTC_23] = (NTC_23, 12)
        
        NTC_34 = a.ext[:parameters][:NTC_34] = NTC_ind  * TC_DC[13]
        ntc_dict[:NTC_34] = (NTC_34, 13)
        
        NTC_15 = a.ext[:parameters][:NTC_15] = NTC_ind  * TC_DC[2]
        ntc_dict[:NTC_15] = (NTC_15, 2)
        
        NTC_25 = a.ext[:parameters][:NTC_25] = NTC_ind  * TC_DC[1] + NTC_ind  * TC_DC[6]
        ntc_dict[:NTC_25] = (NTC_25, 1)  # Using index 1 for demonstration
        
        NTC_35 = a.ext[:parameters][:NTC_35] = NTC_ind  * TC_DC[4] + NTC_ind  * TC_DC[5]
        ntc_dict[:NTC_35] = (NTC_35, 4)  # Using index 4 for demonstration
        
        NTC_45 = a.ext[:parameters][:NTC_45] = NTC_ind  * TC_DC[3]
        ntc_dict[:NTC_45] = (NTC_45, 3)

    elseif read_data == "Reference_Case"
        NTC_12 = a.ext[:parameters][:NTC_12] = NTC_ind  * TC_DC[10]
        ntc_dict[:NTC_12] = (NTC_12, 10)
        
        NTC_23 = a.ext[:parameters][:NTC_23] = NTC_ind  * TC_DC[11]
        ntc_dict[:NTC_23] = (NTC_23, 11)
        
        NTC_13 = a.ext[:parameters][:NTC_13] = NTC_ind  * TC_DC[12]
        ntc_dict[:NTC_13] = (NTC_13, 12)
        
        NTC_14 = a.ext[:parameters][:NTC_14] = NTC_ind  * TC_DC[2]
        ntc_dict[:NTC_14] = (NTC_14, 2)
        
        NTC_24 = a.ext[:parameters][:NTC_24] = NTC_ind  *TC_DC[6]
        ntc_dict[:NTC_24] = (NTC_24, 6)  
        
        NTC_34 = a.ext[:parameters][:NTC_34] = NTC_ind  * TC_DC[5]
        ntc_dict[:NTC_34] = (NTC_34, 5)  

    elseif read_data == "Simple_Hybrid"
        NTC_12 = a.ext[:parameters][:NTC_12] = NTC_ind  * TC_DC[7]
        ntc_dict[:NTC_12] = (NTC_12, 7)
        
        NTC_23 = a.ext[:parameters][:NTC_23] = NTC_ind  * TC_DC[8]
        ntc_dict[:NTC_23] = (NTC_23, 8)
        
        NTC_13 = a.ext[:parameters][:NTC_13] = NTC_ind  * TC_DC[9]
        ntc_dict[:NTC_13] = (NTC_13, 9)
        
        NTC_14 = a.ext[:parameters][:NTC_14] = NTC_ind  * TC_DC[2]
        ntc_dict[:NTC_14] = (NTC_14, 2)
        
        NTC_34 = a.ext[:parameters][:NTC_34] = NTC_ind  * TC_DC[5]
        ntc_dict[:NTC_34] = (NTC_34, 5)
    end
end

# Function to export NTC values to CSV  (Usded to in FB_domain.jl to draw to FB domains)
function export_ntc_values(ntc_dict, df_DC, base_folder_results)
    # Initialize vectors for NTC_ID, BranchID, and NTC values
    ntc_ids = String[]
    ntc_values = Float64[]
    branch_ids = Int[]

    # Populate vectors with NTC_ID, NTC values, and corresponding BranchIDs
    for (key, (value, index)) in ntc_dict
        push!(ntc_ids, string(key))
        push!(ntc_values, value)
        push!(branch_ids, df_DC[index, :BranchID])
    end

    # Create DataFrame from populated vectors
    ntc_df = DataFrame(BranchID = branch_ids, NTC_ID = ntc_ids, NTC = ntc_values)

    # Define the output path and ensure the directory exists
    output_path = joinpath(base_folder_results, "FB_domain/ntc_values.csv")
    mkpath(dirname(output_path))

    # Export to CSV, ensuring to overwrite existing file
    CSV.write(output_path, ntc_df)
end

ntc_dict = Dict{Symbol, Tuple{Float64, Int}}()
 
calculate_ntc_values!(ntc_dict, read_data, TC_DC)
export_ntc_values(ntc_dict, df_DC, base_folder_results)
#------------------------END OF NTC Determination ----------------------



#------------------------Determination of the nodal and zonal PTDF matrices----------------------
a.ext[:parameters][:n_in_z_island] = Dict()
n_in_z_island = a.ext[:parameters][:n_in_z_island]
for row in eachrow(df_node)
    zone = row[:Zone]
    bus_id = row[:BusID]
    if haskey(n_in_z_island, zone)
        push!(n_in_z_island[zone], bus_id)
    else
        n_in_z_island[zone] = [bus_id]
    end
end
a.ext[:parameters][:n_in_z_island] = n_in_z_island
     
incidence_zonal = zeros(length(L), length(Z))
for l in L      
 
    FromBus = df_line[:,:FromBus][l]                        #Extract FromBus and ToBus from the df_line
    ToBus = df_line[:,:ToBus][l]
 
    FromZone = df_node[:,:Zone][findfirst(N .== FromBus)]   #Find the corresponding zones (FromZone and ToZone) for the FromBus and ToBus
    ToZone = df_node[:,:Zone][findfirst(N .== ToBus)]
 
    if FromZone != ToZone                                   #Check if the transmission line connects different zones (FromZone is not equal to ToZone).
        incidence_zonal[l,FromZone] =  1            #If true, set the entry in the incidence_zonal matrix corresponding to the current line and the "FromZone" to 1 and the entry corresponding to the "ToZone" to -1.
        incidence_zonal[l,ToZone] = -1
    end
end

incidence_zonal_DC = zeros(length(L_DC), length(Z))         #The same is done for the DC lines
for l_dc in L_DC
         
    FromBus = df_DC[:,:FromBus][l_dc]
    ToBus = df_DC[:,:ToBus][l_dc]
         
    FromZone = df_node[:,:Zone][findfirst(N .== FromBus)]
    ToZone = df_node[:,:Zone][findfirst(N .== ToBus)]
         
    if FromZone != ToZone
        incidence_zonal_DC[l_dc,FromZone] =  1
        incidence_zonal_DC[l_dc,ToZone] = -1
    end  
end

#Check if the lines are cross-border lines
# for l in L
#     if sum(abs(incidence_zonal[l,z]) for z in Z) == 2       #If the sum of the absolute values of the entries in the incidence_zonal matrix corresponding to the current line is equal to 2, the line is a cross-border line.
#         @info "AC-line ",l, "is a cross-border line"
#     end 
# end
     
# for l_dc in L_DC
#     if sum(abs(incidence_zonal_DC[l_dc,z]) for z in Z) == 2 #The same is done for the DC lines
#         @info "DC-line ",l_dc, "is a cross-border line"
#     end 
# end

#Store the cross-border lines for the FB domain file for visualization purposes
function find_cross_border_lines()
    cb_lines_temp = []
    cb_lines_info = []
    for l in L
        FromBus = df_line[:,:FromBus][l]        #Extract FromBus and ToBus from the df_line
        ToBus = df_line[:,:ToBus][l]

        FromZone = df_node[:,:Zone][findfirst(N .== FromBus)]   #Find the corresponding zones (FromZone and ToZone) for the FromBus and ToBus
        ToZone = df_node[:,:Zone][findfirst(N .== ToBus)]

        if FromZone != ToZone && FromZone in Z && ToZone in Z
            cb_lines_temp = vcat(cb_lines_temp,l)
            BranchID = df_line[l, :BranchID]
            push!(cb_lines_info, (BranchID=BranchID, Line=l, FromBus=FromBus, ToBus=ToBus, FromZone=FromZone, ToZone=ToZone, LineType="AC"))
            # println("AC-line ", l, " is a cross-border line")
        end
    end

    for l_dc in L_DC
        FromBus = df_DC[:,:FromBus][l_dc]        #Extract FromBus and ToBus from the df_line_DC
        ToBus = df_DC[:,:ToBus][l_dc]

        FromZone = df_node[:,:Zone][findfirst(N .== FromBus)]   #Find the corresponding zones (FromZone and ToZone) for the FromBus and ToBus
        ToZone = df_node[:,:Zone][findfirst(N .== ToBus)]

        if FromZone != ToZone && FromZone in Z && ToZone in Z
            cb_lines_temp = vcat(cb_lines_temp,l_dc)
            BranchID = df_DC[l_dc, :BranchID]
            push!(cb_lines_info, (BranchID=BranchID, Line=l_dc, FromBus=FromBus, ToBus=ToBus, FromZone=FromZone, ToZone=ToZone, LineType="DC"))
            # println("DC-line ", l_dc, " is a cross-border line")
        end
    end

    CSV.write(joinpath(base_folder_results, "FB_domain/cross_border_lines.csv"), DataFrame(cb_lines_info))
    return cb_lines_temp
end
find_cross_border_lines()


        #  Build the Nodal PTDF matrix          
if read_data == "5-node_OBZ" || read_data == "5-node_HM"        
    MWBase = 380^2                                  #Base power in MW    
    slack_node = 1                                  #Slack node is node 1
    slack_position = findfirst(N .== slack_node)    #assign the position of the slack node
    
    inc = a.ext[:parameters][:inc] = Matrix(incidence)
    inc_dc = a.ext[:parameters][:inc_dc] = Matrix(incidence_dc)
    i = a.ext[:parameters][:i] = Matrix(incidence_zonal)
    i_DC = a.ext[:parameters][:i_DC] = Matrix(incidence_zonal_DC)
    
    line_sus_mat = Matrix(susceptance)./MWBase*Matrix(incidence)
    node_sus_mat = transpose(Matrix(incidence))*Matrix(susceptance)./MWBase*Matrix(incidence)
    line_sus_mat_ = line_sus_mat[:, 1:end .!= slack_position]
    node_sus_mat_ = node_sus_mat[1:end .!= slack_position, 1:end .!= slack_position]
    PTDF = line_sus_mat_*pinv(node_sus_mat_) #PTDF matrix without slack node
    
    zero_column = zeros(Float64, size(PTDF, 1), 1) # Add a zero colum for the slack node to hatch the correct representation of the PTDF
    global PTDF = hcat(PTDF[:,1:(slack_position-1)], zero_column, PTDF[:,slack_position:end])
    
    
    global NPTDF = a.ext[:parameters][:NPTDF] = transpose(PTDF)
else
    # Build Nodal PTDF
    MWBase = 380^2                         #Base power in MW    
    slack_node1 = 1
    slack_position1 = findfirst(n_in_z_island[1] .== slack_node1)
    slack_node2 = 39
    slack_position2 = findfirst(n_in_z_island[2] .== slack_node2)
    slack_node3 = 24
    slack_position3 = findfirst(n_in_z_island[3] .== slack_node3)

    inc = a.ext[:parameters][:inc] = Matrix(incidence)
    inc_dc = a.ext[:parameters][:inc_dc] = Matrix(incidence_dc)
    i = a.ext[:parameters][:i] = Matrix(incidence_zonal)
    i_DC = a.ext[:parameters][:i_DC] = Matrix(incidence_zonal_DC)

    # line_sus_mat = Matrix(susceptance)./MWBase*Matrix(incidence)
    # node_sus_mat = transpose(Matrix(incidence))*Matrix(susceptance)./MWBase*Matrix(incidence)
    # line_sus_mat_ = line_sus_mat[:, 1:end .!= slack_position]
    # node_sus_mat_ = node_sus_mat[1:end .!= slack_position, 1:end .!= slack_position]
    # PTDF = line_sus_mat_*inv(node_sus_mat_) #PTDF matrix without slack node
    # zero_column = zeros(Float64, length(L), 1)
    # PTDF = hcat(PTDF[:,1:(slack_position-1)], zero_column, PTDF[:,slack_position:end]) #PTDF matrix with slack node (zero column)


    incidence1, susceptance1 = zeros(length(l_in_z[1]),length(n_in_z_island[1])), zeros(length(l_in_z[1]),length(l_in_z[1]))
    incidence2, susceptance2 = zeros(length(l_in_z[2]),length(n_in_z_island[2])), zeros(length(l_in_z[2]),length(l_in_z[2]))
    incidence3, susceptance3 = zeros(length(l_in_z[3]),length(n_in_z_island[3])), zeros(length(l_in_z[3]),length(l_in_z[3]))
    for l in df_line[:,:BranchID]
        if l in l_in_z[1]
            susceptance1[findfirst(l_in_z[1] .== l),findfirst(l_in_z[1] .== l)] = Matrix(susceptance)[findfirst(df_line[:,:BranchID] .== l),findfirst(df_line[:,:BranchID] .== l)]
        end    
        for n in N
            if l in l_in_z[1] && n in n_in_z_island[1]
                incidence1[findfirst(l_in_z[1] .== l),findfirst(n_in_z_island[1] .== n)] = inc[findfirst(df_line[:,:BranchID] .== l),findfirst(N .== n)]
            end  
        end  
    end    
    for l in df_line[:,:BranchID]
        if l in l_in_z[2]
            susceptance2[findfirst(l_in_z[2] .== l),findfirst(l_in_z[2] .== l)] = Matrix(susceptance)[findfirst(df_line[:,:BranchID] .== l),findfirst(df_line[:,:BranchID] .== l)]
        end    
        for n in N
            if l in l_in_z[2] && n in n_in_z_island[2]
                incidence2[findfirst(l_in_z[2] .== l),findfirst(n_in_z_island[2] .== n)] = inc[findfirst(df_line[:,:BranchID] .== l),findfirst(N .== n)]
            end  
        end  
    end   
    for l in df_line[:,:BranchID]
        if l in l_in_z[3]
            susceptance3[findfirst(l_in_z[3] .== l),findfirst(l_in_z[3] .== l)] = Matrix(susceptance)[findfirst(df_line[:,:BranchID] .== l),findfirst(df_line[:,:BranchID] .== l)]
        end    
        for n in N
            if l in l_in_z[3] && n in n_in_z_island[3]
                incidence3[findfirst(l_in_z[3] .== l),findfirst(n_in_z_island[3] .== n)] = inc[findfirst(df_line[:,:BranchID] .== l),findfirst(N .== n)]
            end  
        end  
    end     

    line_sus_mat1 = Matrix(susceptance1)./MWBase*Matrix(incidence1)
    node_sus_mat1 = transpose(Matrix(incidence1))*Matrix(susceptance1)./MWBase*Matrix(incidence1)
    line_sus_mat_1 = line_sus_mat1[:, 1:end .!= slack_position1]
    node_sus_mat_1 = node_sus_mat1[1:end .!= slack_position1, 1:end .!= slack_position1]
    PTDF1 = line_sus_mat_1*pinv(node_sus_mat_1) #PTDF matrix without slack node
    zero_column1 = zeros(Float64, length(l_in_z[1]), 1)
    PTDF1 = hcat(PTDF1[:,1:(slack_position1-1)], zero_column1, PTDF1[:,slack_position1:end]) #PTDF matrix with slack node (zero column)

    line_sus_mat2 = Matrix(susceptance2)./MWBase*Matrix(incidence2)
    node_sus_mat2 = transpose(Matrix(incidence2))*Matrix(susceptance2)./MWBase*Matrix(incidence2)
    line_sus_mat_2 = line_sus_mat2[:, 1:end .!= slack_position2]
    node_sus_mat_2 = node_sus_mat2[1:end .!= slack_position2, 1:end .!= slack_position2]
    PTDF2 = line_sus_mat_2*pinv(node_sus_mat_2) #PTDF matrix without slack node
    zero_column2 = zeros(Float64, length(l_in_z[2]), 1)
    PTDF2 = hcat(PTDF2[:,1:(slack_position2-1)], zero_column2, PTDF2[:,slack_position2:end]) #PTDF matrix with slack node (zero column)

    line_sus_mat3 = Matrix(susceptance3)./MWBase*Matrix(incidence3)
    node_sus_mat3 = transpose(Matrix(incidence3))*Matrix(susceptance3)./MWBase*Matrix(incidence3)
    line_sus_mat_3 = line_sus_mat3[:, 1:end .!= slack_position3]
    node_sus_mat_3 = node_sus_mat3[1:end .!= slack_position3, 1:end .!= slack_position3]
    PTDF3 = line_sus_mat_3*pinv(node_sus_mat_3) #PTDF matrix without slack node
    zero_column3 = zeros(Float64, length(l_in_z[3]), 1)
    PTDF3 = hcat(PTDF3[:,1:(slack_position3-1)], zero_column3, PTDF3[:,slack_position3:end]) #PTDF matrix with slack node (zero column)

    NPTDF = zeros(length(L),length(N))
    for l in l_in_z[1]
        for n in n_in_z_island[1]
            NPTDF[findfirst(df_line[:,:BranchID] .== l),findfirst(N .== n)] = PTDF1[findfirst(l_in_z[1] .== l),findfirst(n_in_z_island[1] .== n)]
        end
    end
    for l in l_in_z[2]
        for n in n_in_z_island[2]
            NPTDF[findfirst(df_line[:,:BranchID] .== l),findfirst(N .== n)] = PTDF2[findfirst(l_in_z[2] .== l),findfirst(n_in_z_island[2] .== n)]
        end
    end
    for l in l_in_z[3]
        for n in n_in_z_island[3]
            NPTDF[findfirst(df_line[:,:BranchID] .== l),findfirst(N .== n)] = PTDF3[findfirst(l_in_z[3] .== l),findfirst(n_in_z_island[3] .== n)]
        end
    end

    NPTDF = a.ext[:parameters][:NPTDF] = transpose(NPTDF)
end

# Array that indicates whether DC line is linking the AC area and the DC area
global border_ACDC = zeros(length(L_DC))
for l_dc in L_DC
    if abs(df_DC[l_dc,:ToDirection]) == 1 | abs(df_DC[l_dc,:FromDirection]) == 1
        global border_ACDC[l_dc] = 1
    end
end
a.ext[:parameters][:border_ACDC] = border_ACDC