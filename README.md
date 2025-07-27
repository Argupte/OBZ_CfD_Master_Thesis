# Contracts for Differences modelled for Offshore Wind Farms in Offshore Bidding Zones

This repository contains all code and models developed for my MSc thesis at TU Delft, which investigates how offshore wind farms interact with zonal electricity markets under different Contracts for Difference (CfD) configurations. The work includes a detailed modelling framework for flow-based market coupling, scenario generation, and welfare evaluation.

📄 **Thesis**: [TU Delft Repository Link](https://repository.tudelft.nl/record/uuid:23f99141-ac43-40f1-aad4-f8c5d94a24de)  
📘 **Programme**: MSc Sustainable Energy Technology, TU Delft  
🗓️ **Year**: 2025

---

##  Repository Structure

| File | Description |
|------|-------------|
| `main.jl` | Launches the model run, activates scenarios, sets configurations. |
| `data.jl` | Loads network, generator, load, and RES profiles; computes PTDFs and parameters. |
| `setup_paths.jl` | Configures input and output paths based on `read_data`; can be simplified using dictionary mapping. |
| `function_zonalclearing_GSK_AHC_H2.jl` | Core flow-based zonal market clearing model including hydrogen redispatch. |
| `OWF_response.jl` | Simulates RES bidding strategies under multiple CfD configurations. |
| `results_OBZ_CfD.jl` | Computes price/volume/congestion risk, export, curtailment, and revenue metrics. |
| `SocialWelfare.jl` | Compares welfare outcomes across CfD schemes using base vs. alternative comparisons. |
| `wind_technology.jl` | Processes wind speed and turbine curves to generate 744-timestep RES production. |

---

##  How to Run

### 1. Set working directory and activate environment
```julia
cd("E:/TU Delft/Thesis/Code")  # Replace with your path
import Pkg
Pkg.activate(".")
```

### 2. Install required packages
```julia
Pkg.add(["JuMP", "Gurobi", "CSV", "DataFrames", "Plots", "XLSX", "DelimitedFiles", "Statistics"])
```

### 3. Gurobi Licence Required

This project **requires Gurobi** to solve the optimisation models.  
You must have a valid Gurobi installation and academic licence.

- Apply here: [https://www.gurobi.com/academia/academic-program-and-licenses/](https://www.gurobi.com/academia/academic-program-and-licenses/)
- Validate with:
```julia
using Gurobi
Gurobi.Env()  # should return environment object if licence is valid
```

### 4. Run the model
```julia
include("main.jl")
```

Results will be saved in:
```
/Results/<scenario_folder>/
```

---

##  Supported CfD Schemes

Adjustable via `OWF_response.jl` or `results_OBZ_CfD.jl`. Supported types include:

- One-sided and two-sided CfDs
- Cap-and-floor schemes
- Financial CfDs with batch or average references
- Capability-based CfDs with iterative bidding
- Node-specific and hybrid bidding zone logic

---

##  Input Data Requirements

- `node_info.csv`, `line_info.csv`, `gen_info.csv`, `load_info.csv`, `DC_info.csv`
- `offshore.csv`, `onshore.csv`, `pv.csv` (RES profiles)
- `h2_info.csv` (hydrogen scenarios)

---

##  Contact

Email: gupteanurag1@gmail.com  
LinkedIn: [www.linkedin.com/in/anurag-gupte]
