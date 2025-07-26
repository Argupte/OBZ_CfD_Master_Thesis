# ========================
# setup_paths.jl
# ========================

# Base folder where all grid topologies are stored
base_grid_topology_folder = "E:/TU Delft/Thesis/Code/grid_topology"

# Set base folders for storing results based on the read_data case
if read_data == "5-node_HM"
    base_folder_results = a.ext[:loops][:base_folder_results] = "Results/test_models/5-node_HM"
elseif read_data == "5-node_OBZ"
    base_folder_results = a.ext[:loops][:base_folder_results] = "Results/test_models/5-node_OBZ"
elseif read_data == "Schonheit_HM"
    base_folder_results = a.ext[:loops][:base_folder_results] = "Results/test_models/_Schonheit_HM"
elseif read_data == "Schonheit_OBZ"
    base_folder_results = a.ext[:loops][:base_folder_results] = "Results/test_models/_Schonheit_OBZ"
elseif read_data == "Schonheit_OBZ_adjusted"
    base_folder_results = a.ext[:loops][:base_folder_results] = "Results/test_models/_Schonheit_OBZ_adjusted"
elseif read_data == "Reference_Case"
    base_folder_results = a.ext[:loops][:base_folder_results] = "Results/Reference_Case"
elseif read_data == "Simple_Hybrid"
    base_folder_results = a.ext[:loops][:base_folder_results] = "Results/Simple_Hybrid"
end

# Set base folders for input data based on the read_data case
if read_data == "5-node_HM"
    base_folder_data = a.ext[:loops][:base_folder_data] = joinpath(base_grid_topology_folder, "test_networks/5-node_HM")
elseif read_data == "5-node_OBZ"
    base_folder_data = a.ext[:loops][:base_folder_data] = joinpath(base_grid_topology_folder, "test_networks/5-node_OBZ")
elseif read_data == "Schonheit_HM"
    base_folder_data = a.ext[:loops][:base_folder_data] = joinpath(base_grid_topology_folder, "test_networks/_Schonheit_HM")
elseif read_data == "Schonheit_OBZ"
    base_folder_data = a.ext[:loops][:base_folder_data] = joinpath(base_grid_topology_folder, "test_networks/_Schonheit_OBZ")
elseif read_data == "Schonheit_OBZ_adjusted"
    base_folder_data = a.ext[:loops][:base_folder_data] = joinpath(base_grid_topology_folder, "test_networks/_Schonheit_OBZ_adjusted")
elseif read_data == "Reference_Case"
    base_folder_data = a.ext[:loops][:base_folder_data] = joinpath(base_grid_topology_folder, "Reference_Case")
elseif read_data == "Simple_Hybrid"
    base_folder_data = a.ext[:loops][:base_folder_data] = joinpath(base_grid_topology_folder, "Simple_Hybrid")
end

# After this file runs, `base_folder_data` will be set based on `read_data` value
