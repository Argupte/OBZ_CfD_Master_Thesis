
import os
import pandas as pd


base_folder_path = "Results\\___Case_Group5\\_Off_2WTP32.83"  

horizontal_files = [
    'Commercial_Flow.csv',
    'Congestion_DC.csv',
    'Cost_generation.csv',
    'Curtailment.csv',
    'Dispatch.csv',
    'Flow_AC.csv',
    'Flow_DC.csv',
    'flow_FBMC_DA.csv',
    'Net_production_OWF.csv',
    'Overload.csv',
    'Price.csv',
    'RES_production.csv',
    'Total_NetPosition.csv',
    'shadow_prices/shadow_price_con1.csv',
    'shadow_prices/shadow_price_con2.csv',
    'shadow_prices/shadow_price_con3.csv',
    'shadow_prices/shadow_price_con4.csv',
    'shadow_prices/shadow_price_con5.csv',
    'redispatch/Cost_redispatch_per_zone.csv',
    'redispatch/DC_delta_RD.csv',
    'redispatch/FlowAfterRD.csv',
    'redispatch/FlowBeforeRD.csv',
    'redispatch/Redispatch_curtailment_opt.csv',
    'redispatch/Redispatch_Downward.csv',
    'redispatch/Redispatch_Upward.csv',
    'redispatch/shadow_price_con2.csv',
    'redispatch/shadow_price_con3.csv',
    'redispatch/shadow_price_con4.csv',
    'redispatch/shadow_price_con5.csv',
    'redispatch/shadow_price_con6.csv',
    'redispatch/shadow_price_con7.csv',
    'redispatch/shadow_price_con8.csv',
    'FB_domain/Flow_based_positions.csv',
    'FB_domain/max_possible_AC_flows.csv',
    'FB_domain/max_possible_DC_flows.csv',
    'FB_domain/MaxOBZ_NPs.csv',
    'FB_domain/RAM_neg_max.csv',
    'FB_domain/RAM_pos_max.csv',
    'base-case/Cost_generation_base_case.csv',
    'base-case/Curtailment_base_case.csv',
    'base-case/Dispatch_base_case.csv',
    'base-case/Flow_AC_base_case.csv',
    'base-case/Flow_DC_base_case.csv',
    'base-case/Net_Production_OWF_base_case.csv',
    'base-case/NetPosition_base_case.csv',
    'base-case/RAM_neg_base_case.csv',
    'base-case/RAM_pos_base_case.csv',
    'base-case/shadow_price_con1.csv',
    'base-case/shadow_price_con2.csv',
    'base-case/shadow_price_con3.csv',
    'base-case/shadow_price_con4.csv',
    'base-case/shadow_price_con5.csv',
    'base-case/shadow_price_con6.csv',
    'Hydrogen_dispatch.csv',            #Put off in case no hydrogen
    'redispatch/Hydrogen_Downward.csv',
    'redispatch/Hydrogen_Upward.csv',
]

vertical_files = [
    'Congestion_Rents.csv',
    'Overload_DC.csv',
    'shadow_prices/shadow_price_con6.csv',
    'shadow_prices/shadow_price_con7.csv',
    'shadow_prices/shadow_price_con8.csv',
    'shadow_prices/shadow_price_con9.csv',
    'shadow_prices/shadow_price_con10.csv',
    'shadow_prices/shadow_price_con11.csv',
    'shadow_prices/shadow_price_con12.csv',
    'shadow_prices/shadow_price_con13.csv',
    'shadow_prices/shadow_price_con14.csv',
    'shadow_prices/shadow_price_con15.csv',
    'shadow_prices/shadow_price_con16.csv',
    'shadow_prices/shadow_price_con17.csv',
    'shadow_prices/shadow_price_con18.csv',
    'shadow_prices/shadow_price_con19.csv',
    'shadow_prices/shadow_price_con20.csv',
    'redispatch/Cost_total.csv',
    'redispatch/shadow_price_con1.csv',
    'redispatch/shadow_price_con9.csv',
    'redispatch/shadow_price_con10.csv',
    'redispatch/shadow_price_con11.csv',
    'redispatch/shadow_price_con12.csv',
    'redispatch/shadow_price_con13.csv',
    'redispatch/shadow_price_con14.csv',
    'base-case/shadow_price_con7.csv',
    'base-case/shadow_price_con8.csv',
    'base-case/shadow_price_con9.csv',
    'base-case/shadow_price_con10.csv',
    'base-case/shadow_price_con11.csv',
    'base-case/shadow_price_con12.csv',
    'base-case/shadow_price_con13.csv',
    'base-case/shadow_price_con14.csv',
    'base-case/shadow_price_con15.csv'
]



def generate_hour_headers(total_length):
    return [f'x{i}' for i in range(1, total_length + 1)]

def aggregate_files(base_folder_path, horizontal_files, vertical_files):
    # Define months
    months = ['jan', 'apr', 'jul', 'oct']                                   #Adjust if more months (folders) are added
    
    # Create the total directory if it doesn't exist
    total_directory = os.path.join(base_folder_path, 'total')
    # os.makedirs(total_directory, exist_ok=True)
    
    # Function to aggregate horizontal files
    def aggregate_horizontal(file_path_template):
        dataframes = []
        total_length = 0
        for month in months:
            file_path = file_path_template.format(month=month)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, header=None)
                total_length += len(df.columns)
                dataframes.append(df)
        if dataframes:
            aggregated_df = pd.concat(dataframes, axis=1)
            hour_headers = generate_hour_headers(total_length)
            aggregated_df.columns = hour_headers
            aggregated_df = aggregated_df.iloc[1:].reset_index(drop=True)
            relative_path = os.path.relpath(file_path_template.format(month=''), base_folder_path)
            output_file = os.path.join(total_directory, relative_path)
            # os.makedirs(os.path.dirname(output_file), exist_ok=True)
            aggregated_df.to_csv(output_file, index=False)
            print(f"Horizontally aggregated data saved to {output_file}")

    # Function to aggregate vertical files
    def aggregate_vertical(file_path_template):
        dataframes = []
        for month in months:
            file_path = file_path_template.format(month=month)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                dataframes.append(df)
        if dataframes:
            aggregated_df = pd.concat(dataframes, axis=0)
            relative_path = os.path.relpath(file_path_template.format(month=''), base_folder_path)
            output_file = os.path.join(total_directory, relative_path)
            # os.makedirs(os.path.dirname(output_file), exist_ok=True)
            aggregated_df.to_csv(output_file, index=False)
            print(f"Vertically aggregated data saved to {output_file}")

    # Process horizontal files
    for file_path in horizontal_files:
        aggregate_horizontal(os.path.join(base_folder_path, '{month}', file_path))

    # Process vertical files
    for file_path in vertical_files:
        aggregate_vertical(os.path.join(base_folder_path, '{month}', file_path))



# Usage
aggregate_files(base_folder_path, horizontal_files, vertical_files)
