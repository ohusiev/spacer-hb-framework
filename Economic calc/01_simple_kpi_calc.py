#%%
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
from util_func import UtilFunctions
from util_func import PlotFunctions
from util_func import PVAnalysis

#%% INPUT FILES:
root=r"H:\My Drive\01_PHD_CODE\chapter_5\spain\bilbao_otxarkoaga"
#root=r"D:\o.husiev@opendeusto.es\01_PHD_CODE\chapter_5\spain\bilbao_otxarkoaga"


pv_file_name = "01_footprint_s_area_wb_rooftop_analysis_pv_month_pv.xlsx"
pv_path = os.path.join(root, "02_pv_calc_from_bond_rooftop/data/Otxarkoaga/",pv_file_name)
# Load the data with columns as integers
df_pv_gen = pd.read_excel(pv_path, sheet_name='Otxarkoaga' ,dtype={'build_id': str, 'census_id': str})

df_facades = gpd.read_file(os.path.join(root, "data/05_buildings_with_energy_and_co2_values+HDemProj.geojson"))
#%%
#df_pv_temp= df_pv_gen.groupby('build_id').sum().reset_index()
df_facades['build_id'] = df_facades['build_id'].astype(str)

df_facades = pd.merge(df_facades, df_pv_gen[["build_id", "r_area", "installed_kWp", "n_panel", "Total, kWh"]], on="build_id", how='left')
df_facades[["r_area", "n_panel", "Total, kWh"]] = df_facades[["r_area", "n_panel", "Total, kWh"]].fillna(0)

# Upload Envelope cost data
df_costs = pd.read_excel("241211_econom_data.xlsx", sheet_name='cost_facade')
df_h_dem_reduction_coeffs = pd.read_excel("241211_econom_data.xlsx", sheet_name='HDemReductionCoeffs', index_col='id')
# Upload electricity prices and monthly self-consumption percentages
df_prices = pd.read_excel("241211_econom_data.xlsx", sheet_name='cost_electricity')
# Upload monthly self-consumption percentages
df_monthly_self_cons_percentages = pd.read_excel("241211_econom_data.xlsx", sheet_name='sel-cons_%_fixed', index_col=0)
monthly_self_cons_percentages = df_monthly_self_cons_percentages.set_index('Month')['self-cons'].to_dict()

#%%
util= UtilFunctions()
util_pv = PVAnalysis()
util_plot = PlotFunctions()

#%%Plot PV generation by month total with title :"Estimated Monthly PV Generation"
print("Filtering PV data")
df_facades=df_facades[df_facades['Codigo_Uso']=='V']
#df_pv_temp = df_pv_gen[(df_pv_gen["building"] == "V") & (df_pv_gen["Total, kWh"] > 0)].sum()

columns_to_plot = [i for i in range(1, 13)]
util_plot.bar_plot_month_pv_columns(df_pv_gen[df_pv_gen['build_id']=='Total'], columns_to_plot, title="Monthly Data Distribution", xlabel="Months", ylabel="PV Electricity (kWh)", font_size=18)

#%%
# reindex by census_id
df_pv_gen_census = df_pv_gen.groupby('census_id').sum()
#select only month columns
df_pv_gen_census = df_pv_gen_census[columns_to_plot]
# round values to integer
df_pv_gen_census = df_pv_gen_census.round(0)
#plot stackplot
util_plot.plot_census_stackplot(df_pv_gen_census, title="Estimated Monthly PV Generation by Census", xlabel="Months", ylabel="PV Electricity (kWh)", font_size=18)

# %%
#SCENARIO NAME 
SCENARIO = "S2" # options : "S1" or "S2"
heating_energy_price_euro_per_kWh = 0.243 # - 0.243 El price S1 of 2024 for Spain#0.1 #0.307  # €/kWh
energy_price_growth_rate = 0#0.02  # 2% annual increase in energy prices
pv_degradation_rate = 0#0.01  # 1% per year
#%%
# FACADE ECONOMIC CALCULATIONS
# Select the scenario Name for the costs
c_facade_m2 = df_costs.loc[(df_costs["Type"]=='costs') & (df_costs["Component"]=='facade')&(df_costs["Name"]!='Maint'), [SCENARIO]].sum().values[0]
maint_facade_m2 = df_costs.loc[(df_costs["Type"]=='costs') & (df_costs["Component"]=='facade')&(df_costs["Name"]=='Maint'), [SCENARIO]].sum().values[0]
c_windows_m2 = df_costs.loc[(df_costs["Type"]=='costs') & (df_costs["Component"]=='windows'), [SCENARIO]].sum().values[0]
maint_windows_m2 = df_costs.loc[(df_costs["Type"]=='costs') & (df_costs["Component"]=='windows')&(df_costs["Name"]=='Maint'), [SCENARIO]].sum().values[0]
c_roof_m2 = df_costs.loc[(df_costs["Type"]=='costs') & (df_costs["Component"]=='roof'), [SCENARIO]].sum().values[0]
maint_roof_m2 = df_costs.loc[(df_costs["Type"]=='costs') & (df_costs["Component"]=='roof')&(df_costs["Name"]=='Maint'), [SCENARIO]].sum().values[0]

#%%
#Investment (with Laubor) (I) and Maintenance (M) costs for Facades, Windows and Roof
df_facades[f"{SCENARIO}_Facd_I_C"]= df_facades["Total_fa_area"] * c_facade_m2
df_facades[f"{SCENARIO}_Facd_M_C"]= df_facades["Total_fa_area"] * maint_facade_m2

df_facades[f"{SCENARIO}_Wind_I_C"]= df_facades["Tot_window_area"] * c_windows_m2
df_facades[f"{SCENARIO}_Wind_M_C"]= df_facades["Tot_window_area"] * maint_windows_m2

df_facades[f"{SCENARIO}_Roof_I_C"]= df_facades["r_area"] * c_roof_m2
df_facades[f"{SCENARIO}_Roof_M_C"]= df_facades["r_area"] * maint_roof_m2

#%%
#HEATING DEMAND REDUCTION CALCULATIONS
intervention_combinations = {
    1: ["facade", "roof"], # low_intervention_min or high_intervention_min
    2: ["facade", "windows", "roof"],#low_intervention_max or high_intervention_max
}

REGION = 'Pais_Vasco'
COMBINATION_ID = 2 #select the components of the envelope that are considered for the heating demand reduction
#
#Coomponents of the envelope that are considered for the heating demand reduction
combination = intervention_combinations[COMBINATION_ID]
#%% Function to adjust heating demand estimates from two methods in a vectorized manner
def adjust_heating_demand_vectorized(df,menthod1_col_name,menthod2_col_name, relative_diff_threshold =5, base_weight=0.5, scaling_factor=0.01):
    """
    Adjusts heating demand estimates from two methods in a vectorized manner.
    Applies dynamic weighting if one estimate is 'relative_diff_threshold'-times more than value of the other;
    otherwise, returns the menthod2_col_name by default.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'Method1' :INSPIRE Project and 'Method2': HotMaps columns.
    - menthod1_col_name (str): The name of the column containing the first heating demand estimate.
    - menthod2_col_name (str): The name of the column containing the second heating demand estimate.
    - relative_diff_threshold (float): The threshold for the relative difference between the two methods.
    - base_weight (float): The base weight assigned to Method1. Method2's base weight will be (1 - base_weight).
    - scaling_factor (float): A factor to scale the proportional adjustment based on the difference.

    Returns:
    - pd.Series: Adjusted heating demand estimates. 
    """
    method1 = df[menthod1_col_name]
    method2 = df[menthod2_col_name]

    # Calculate the ratio of the methods
    ratio = method1 / method2

    # Identify where one estimate is more than three times the other
    condition = (ratio > relative_diff_threshold) | (ratio < 1/relative_diff_threshold)

    # Calculate the absolute difference between the two methods
    difference = np.abs(method1 - method2)

    # Calculate adjustment coefficient based on the difference
    adjustment_coefficient = 1 + (scaling_factor * difference)

    # Initialize weights
    weight1 = np.where(method1 > method2, base_weight * adjustment_coefficient, base_weight / adjustment_coefficient)
    weight2 = np.where(method1 > method2, (1 - base_weight) / adjustment_coefficient, (1 - base_weight) * adjustment_coefficient)

    # Normalize weights to ensure they sum to 1
    total_weight = weight1 + weight2
    weight1 /= total_weight
    weight2 /= total_weight

    # Calculate the adjusted heating demand
    adjusted_value = (weight1 * method1) + (weight2 * method2)

    # For cases where the condition is not met, return the value of method2
    adjusted_value = np.where(condition, adjusted_value, method2)
    print (f'By default, the adjusted value is the value of `{menthod2_col_name}` when the condition is not met')
    return pd.Series(adjusted_value, index=df.index).round(2)

def calculate_heating_demand_reduction(df_h_dem_reduction_coeffs, region, scenario, combination):
    h_dem_reduction_coeff = df_h_dem_reduction_coeffs.loc[
        (df_h_dem_reduction_coeffs["Region"] == region) &
        (df_h_dem_reduction_coeffs["Type"] == 'HDemReduction') &
        (df_h_dem_reduction_coeffs["Component"] == 'envelope') &
        (df_h_dem_reduction_coeffs["Name"].isin(combination)),
        [scenario]
    ].sum().values[0].round(3)
    if pd.isna(h_dem_reduction_coeff):
        h_dem_reduction_coeff = 0
        print(f"No heating demand reduction coefficient found for Region: {region}, Scenario: {scenario}, Combination: {combination}")
    return h_dem_reduction_coeff
#%%
h_dem_reduction_coeff = calculate_heating_demand_reduction(df_h_dem_reduction_coeffs, REGION, SCENARIO, combination)


"""# Example DataFrame
data = {
    'BuildingID': [1, 2, 3],
    'Method1': [70.0, 65.0, 80.0],  # kWh/m²·year from the first method
    'Method2': [6.9, 60.0, 85.0]    # kWh/m²·year from the second method
}
df = pd.DataFrame(data)

# Apply the vectorized adjustment function
df['AdjustedDemand'] = adjust_heating_demand_vectorized(df, 'Method1', 'Method2')
print(df)"""

# HEATING DEMAND SAVINGS CALCULATIONS
df_facades['HDemProj'] = df_facades['HDemProj'].fillna(0)
df_facades['HDem_iNSPiRE'] = df_facades['HDem_iNSPiRE'].fillna(0)
df_facades['HDemProj'] = adjust_heating_demand_vectorized(df_facades,  'HDem_iNSPiRE','HDemProj')

df_facades[f'Comb{COMBINATION_ID}{SCENARIO}_HDem_kWh/m2'] = df_facades['HDemProj']* (1-calculate_heating_demand_reduction(df_h_dem_reduction_coeffs, REGION, SCENARIO, combination))
# calculate gross floor area
# Syntetic data for floor numbers
#df_facades['n_floorsEstim'] = np.random.randint(5, 12, df_facades.shape[0])

#%% 
#Estimation based on the original data for Otxarkoaga
data = {
    "census_id": [4802003003, 4802003005, 4802003006, 4802003007, 4802003009, 4802003010, 4802003011, 4802003015],
    "Num dwellings": [501, 830, 404, 464, 536, 350, 664, 464],
    "Total Population": [1037, 1624, 846, 914, 1220, 737, 1395, 1088],
    "Avg dwelling size": [79, 59, 59, 59, 73, 59, 57, 80],
    "Buildings": [29, 58, 25, 28, 36, 22, 50, 19]  # Assuming missing value for last row
}

df_otxarkoaga = pd.DataFrame(data)


# Calculate Gross Floor Area and Number of Dwellings
df_otxarkoaga['census_id'] = df_otxarkoaga['census_id'].astype(str)
# check if the columns are already in dataframe
if 'Avg dwelling size' not in df_facades.columns:
    df_facades = df_facades.merge(df_otxarkoaga[['census_id', 'Num dwellings', 'Avg dwelling size']], on='census_id', how='left')
if 'grossFloorArea' not in df_facades.columns:
    df_facades['grossFloorArea'] = df_facades['f_area'] * df_facades['n_floorsEstim']

#df_facades['grossFloorArea'] = df_facades['Num dwellings'] * df_facades['Avg dwelling size']
if 'n_dwellOriginal' not in df_facades.columns:
    df_facades['n_dwellOriginal'] = (df_facades['grossFloorArea'] / df_facades['Avg dwelling size']).round(0)
#Estimation based on the average dwelling size assuming 55 m2 per dwelling
if 'n_dwellEstim' not in df_facades.columns:
    AVG_DWE_SIZE = 55  # Average dwelling size in m2
    df_facades['n_dwellEstim'] = (df_facades['grossFloorArea'] / AVG_DWE_SIZE).round(0)

#%%
# Calculate Energy Savings for Gross Floor Area
df_facades[f'Comb{COMBINATION_ID}{SCENARIO}_HDem_Savings_kWh'] = (df_facades['HDemProj'] - df_facades[f'Comb{COMBINATION_ID}{SCENARIO}_HDem_kWh/m2']) * df_facades['grossFloorArea'] 
# Calculate Heating Savings

df_facades[f'Comb{COMBINATION_ID}{SCENARIO}_HDem_Savings_Euro'] = df_facades[f'Comb{COMBINATION_ID}{SCENARIO}_HDem_Savings_kWh'] * heating_energy_price_euro_per_kWh

#Round All values to 3 decimal points
df_facades.loc[:, df_facades.columns[-8:]] = df_facades.loc[:, df_facades.columns[-8:]].round(3)

#%% Save to file
df_facades.to_file(os.path.join(root,"05_buildings_with_energy_and_co2_values+HDemProj_facade_costs_with_HDem_corr.geojson"), driver='GeoJSON')
# %% ROOFTOP PV ECONOMIC CALCULATIONS
FILE_NAME_FILER = "01_footprint_s_area_wb_rooftop_analysis_pv_month_pv_filter.xlsx"
PV_M_C = 0.025

df_pv_gen['PV_I_C'] = df_pv_gen['installed_kWp'].apply(lambda x: util_pv.pv_size_to_cost_equation(x))
df_pv_gen['PV_M_C'] = df_pv_gen['PV_I_C']*PV_M_C
#Round to 3 decimal points
df_pv_gen.loc[:, ['PV_I_C', 'PV_M_C']] = df_pv_gen.loc[:, ['PV_I_C', 'PV_M_C']].round(3)

#%%
# Save PV to file
save_path = os.path.join(root, pv_file_name)

util.add_sheet_to_excel(df_pv_gen, save_path, sheet_name='pv_cost', index=False)

#%%
# Join Rooftop PV costs to the Facades dataframe
df_facades = pd.merge(df_facades, df_pv_gen[["build_id", "PV_I_C", "PV_M_C"]], on="build_id", how='left')
#%%
df_facades.to_file(os.path.join(root,"05_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV.geojson"), driver='GeoJSON')


df_pv_results = util_pv.calculate_pv_metrics_adv_df(df_pv_gen, df_prices, monthly_self_cons_percentages, degradation_rate = pv_degradation_rate, price_increase_rate = energy_price_growth_rate)
df_pv_results.drop(columns=['census_id'], inplace=True)
df_economic_analysis = df_facades.merge(df_pv_results, on=['build_id'], how='left')
print(df_economic_analysis[['build_id', 'census_id', 'PV_self_cons_kWh', 'PV_to_grid_kWh', 'PV_self_cons_Euro', 'PV_to_grid_Euro',]])
#%% Calculate PV metrics
"""df_pv_results = util_pv.calculate_pv_metrics(df_pv_gen, df_prices, monthly_self_cons_percentages)
df_pv_results.drop(columns=['census_id'], inplace=True)
df_economic_analysis = df_facades.merge(df_pv_results, on=['build_id'], how='left')
print(df_economic_analysis[['build_id', 'census_id', 'PV_self_cons_kWh', 'PV_to_grid_kWh', 'PV_self_cons_Euro', 'PV_to_grid_Euro',]])"""
#%%
# Save to files
df_economic_analysis.to_file(os.path.join(root,"05_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV_economic.geojson"), driver='GeoJSON')

#Remove the geometry column and save to excel
df_economic_analysis = df_economic_analysis.drop(columns='geometry')
df_economic_analysis.to_excel(os.path.join(root,"05_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV_economic_otxarkpaga_orginal_price_incr+pv_degrad.xlsx"), index=False)

#%% TESTING FUNCTIONS TO calculate PV savings with degradation and price increase

def average_price_increase(row, price_increase, lifespan=20):
    initial_price = row
    """
    Calculate the average annual price over the lifespan.

    Parameters:
    - initial_price (float): Initial price in Euro.
    - price_increase (float): Annual price increase rate (e.g., 0.02 for 2%).
    - lifespan (int): Lifespan of the system in years.

    Returns:
    - float: Average annual price considering price increase.
    """
    total_price_increase = sum(initial_price *(1 + price_increase) ** (t - 1) for t in range(1, lifespan + 1)
                               )
    return total_price_increase / lifespan

def calculate_pv_row_average(row, degradation_rate=0.01, lifespan=20):
    """
    Calculate the average annual PV production considering degradation.

    Parameters:
    - initial_production (float): Initial annual production in kWh.
    - degradation_rate (float): Annual PV degradation rate (e.g., 0.01 for 1%).
    - lifespan (int): Lifespan of the system in years.

    Returns:
    - float: Average annual PV production.
    """
    initial_production = row
    total_production = sum(
        initial_production * (1 - degradation_rate) ** (t - 1) for t in range(1, lifespan + 1)
    )
    return total_production / lifespan
    # Function to calculate PV_self_cons_Euro and PV_to_grid_Euro
def calculate_pv_metrics_adv(df_pv, df_prices, monthly_self_cons_percentages, incl_degradation=0, incl_price_increase=0):
    """
    Calculate PV metrics for each building in the dataset.

    Parameters:
    - df_pv (pd.DataFrame): PV data with monthly generation columns.
    - df_prices (pd.DataFrame): Electricity buy and sell prices by month.
    - monthly_self_cons_percentages (dict): Self-consumption percentages by month.
    - incl_degradation (bool): Whether to include degradation in calculations.
    - incl_price_increase (bool): Whether to include price increase in calculations.

    Returns:
    - pd.DataFrame: DataFrame with calculated PV metrics.
    """
    monthly_columns = list(range(1, 13))  # Monthly columns
    results = []

    for index, row in df_pv.iterrows():
        pv_self_cons_kWh = 0
        pv_to_grid_kWh = 0
        pv_self_cons_euro = 0
        pv_to_grid_euro = 0
        
        for month in monthly_columns:
            monthly_generation = row[month]
            if incl_degradation != 0:
                # Calculate monthly generation with degradation
                monthly_generation = calculate_pv_row_average(row[month],degradation_rate=incl_degradation, lifespan=20)
            
            buy_price = df_prices.loc[month - 1, 'Electricity Buy']  # Match price by month
            sell_price = df_prices.loc[month - 1, 'Electricity Sell']
            if incl_price_increase != 0:
                # Calculate average price increase
                buy_price = average_price_increase(buy_price, price_increase=incl_price_increase)
                sell_price = average_price_increase(sell_price, price_increase=incl_price_increase)
            
            # Get the self-consumption percentage for the current month
            month_name = df_prices.loc[month - 1, 'Month']
            self_cons_percent = monthly_self_cons_percentages[month_name]
            
            # Self-consumed and exported energy
            self_consumed_energy_kWh = monthly_generation * self_cons_percent
            exported_energy_kWh = monthly_generation * (1 - self_cons_percent)
            
            # Add to total kWh
            pv_self_cons_kWh += self_consumed_energy_kWh
            pv_to_grid_kWh += exported_energy_kWh
            
            # Monetary values
            pv_self_cons_euro += self_consumed_energy_kWh * buy_price
            pv_to_grid_euro += exported_energy_kWh * sell_price
        
        # Append results
        results.append({
            'build_id': row['build_id'],
            'census_id': row['census_id'],
            'PV_self_cons_kWh': round(pv_self_cons_kWh, 3),
            'PV_to_grid_kWh': round(pv_to_grid_kWh, 3),
            'PV_self_cons_Euro': round(pv_self_cons_euro, 3),
            'PV_to_grid_Euro': round(pv_to_grid_euro, 3)
        })
    
    return pd.DataFrame(results)
""" 
# Example usage
data_pv = {
    'build_id': [1, 2, 3],
    'census_id': [101, 102, 103],
    'r_area': [50, 60, 70],  # Roof area
    'n_panel': [20, 24, 28],  # Number of panels
    's_roof': [45, 55, 65],  # Roof suitability factor
    1: [300, 310, 320],  # kWh generated in January
    2: [280, 290, 300],  # February
    3: [320, 330, 340],  # March
    4: [360, 370, 380],  # April
    5: [400, 410, 420],  # May
    6: [420, 430, 440],  # June
    7: [410, 420, 430],  # July
    8: [390, 400, 410],  # August
    9: [350, 360, 370],  # September
    10: [310, 320, 330],  # October
    11: [290, 300, 310],  # November
    12: [280, 290, 300],  # December
    'Total, kWh': [4420, 4520, 4620],  # Total annual generation
    'installed_kWp': [5, 6, 7],  # Installed capacity
}
df_pv = pd.DataFrame(data_pv)
# Example table for electricity prices
electricity_prices = {
    'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'Electricity Buy': [0.251, 0.248, 0.246, 0.2435, 0.241, 0.2454, 0.238, 0.2365, 0.2349, 0.234, 0.235, 0.2347],
    'Electricity Sell': [0.0695, 0.1335, 0.0896, 0.0737, 0.07, 0.0786, 0.0875, 0.1173, 0.0989, 0.1049, 0.1084, 0.1035],
}
df_prices = pd.DataFrame(electricity_prices)

# Define monthly self-consumption percentages
monthly_self_cons_percentages = {
    'January': 0.6, 'February': 0.5, 'March': 0.55, 'April': 0.65,
    'May': 0.7, 'June': 0.75, 'July': 0.8, 'August': 0.7,
    'September': 0.6, 'October': 0.55, 'November': 0.5, 'December': 0.6,
}
"""

# Calculate PV metrics
df_pv_metrics = calculate_pv_metrics_adv(df_pv_gen, df_prices, monthly_self_cons_percentages)
df_pv_metrics_degradation = calculate_pv_metrics_adv(df_pv_gen, df_prices, monthly_self_cons_percentages, incl_degradation=0.01)
df_pv_metrics_degradation_price = calculate_pv_metrics_adv(df_pv_gen, df_prices, monthly_self_cons_percentages, incl_degradation=0.01, incl_price_increase=0.01)


# %%  Example DataFrame to calculate PV metrics by census with 
#different monthly self-consumption percentages

# Example DataFrame for PV generation
data_pv = {
    'build_id': [1, 2, 3],
    'census_id': [101, 102, 103],
    'r_area': [50, 60, 70],  # Roof area
    'n_panel': [20, 24, 28],  # Number of panels
    's_roof': [45, 55, 65],  # Roof suitability factor
    1: [300, 310, 320],  # kWh generated in January
    2: [280, 290, 300],  # February
    3: [320, 330, 340],  # March
    4: [360, 370, 380],  # April
    5: [400, 410, 420],  # May
    6: [420, 430, 440],  # June
    7: [410, 420, 430],  # July
    8: [390, 400, 410],  # August
    9: [350, 360, 370],  # September
    10: [310, 320, 330],  # October
    11: [290, 300, 310],  # November
    12: [280, 290, 300],  # December
    'Total, kWh': [4420, 4520, 4620],  # Total annual generation
    'installed_kWp': [5, 6, 7],  # Installed capacity
}
df_pv = pd.DataFrame(data_pv)

# Example table for electricity prices
electricity_prices = {
    'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'Electricity Buy': [0.251, 0.248, 0.246, 0.2435, 0.241, 0.2454, 0.238, 0.2365, 0.2349, 0.234, 0.235, 0.2347],
    'Electricity Sell': [0.0695, 0.1335, 0.0896, 0.0737, 0.07, 0.0786, 0.0875, 0.1173, 0.0989, 0.1049, 0.1084, 0.1035],
}
df_prices = pd.DataFrame(electricity_prices)

# Define monthly self-consumption percentages
monthly_self_cons_percentages = {
    'January': 0.6, 'February': 0.5, 'March': 0.55, 'April': 0.65,
    'May': 0.7, 'June': 0.75, 'July': 0.8, 'August': 0.7,
    'September': 0.6, 'October': 0.55, 'November': 0.5, 'December': 0.6,
}
monthly_self_cons_percentages_by_census = {
    101: {
        'January': 0.6, 'February': 0.5, 'March': 0.55, 'April': 0.65,
        'May': 0.7, 'June': 0.75, 'July': 0.8, 'August': 0.7,
        'September': 0.6, 'October': 0.55, 'November': 0.5, 'December': 0.6,
    },
    102: {
        'January': 0.65, 'February': 0.55, 'March': 0.6, 'April': 0.7,
        'May': 0.75, 'June': 0.8, 'July': 0.85, 'August': 0.75,
        'September': 0.65, 'October': 0.6, 'November': 0.55, 'December': 0.65,
    }}

util_pv = PVAnalysis()
df_pv_results = util_pv.calculate_pv_metrics_by_census(df_pv, df_prices, monthly_self_cons_percentages_by_census)

#%%
# EXAMPLE DataFrame for PV generation
data_pv = {
    'build_id': [1, 2, 3],
    'census_id': [101, 101, 103],
    'r_area': [50, 60, 70],  # Roof area
    'n_panel': [20, 24, 28],  # Number of panels
    's_roof': [45, 55, 65],  # Roof suitability factor
    1: [100, 150, 320],  # kWh generated in January
    2: [100, 150, 300],  # February
    3: [100, 150, 340],  # March
    #4: [360, 370, 380],  # April
    #5: [400, 410, 420],  # May
    #6: [420, 430, 440],  # June
    #7: [410, 420, 430],  # July
    #8: [390, 400, 410],  # August
    #9: [350, 360, 370],  # September
    #10: [310, 320, 330],  # October
    #11: [290, 300, 310],  # November
    12: [280, 450, 300],  # December
    'Total, kWh': [300, 4520, 4620],  # Total annual generation
    'installed_kWp': [5, 6, 7],  # Installed capacity
}
df_pv = pd.DataFrame(data_pv)


# Example table for electricity prices
electricity_prices = {
    'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'Electricity Buy': [0.251, 0.248, 0.246, 0.2435, 0.241, 0.2454, 0.238, 0.2365, 0.2349, 0.234, 0.235, 0.2347],
    'Electricity Sell': [0.0695, 0.1335, 0.0896, 0.0737, 0.07, 0.0786, 0.0875, 0.1173, 0.0989, 0.1049, 0.1084, 0.1035],
}
df_prices = pd.DataFrame(electricity_prices)

# Define monthly self-consumption percentages
monthly_self_cons_percentages = {
    'January': 0.6, 'February': 0.5, 'March': 0.55, 'April': 0.65,
    'May': 0.7, 'June': 0.75, 'July': 0.8, 'August': 0.7,
    'September': 0.6, 'October': 0.55, 'November': 0.5, 'December': 0.6,
}

# Function to calculate PV_self_cons_kWh, PV_to_grid_kWh, PV_self_cons_Euro, and PV_to_grid_Euro
def calculate_pv_metrics(df_pv, df_prices, monthly_self_cons_percentages):
    monthly_columns = list(range(1, 4))  # Monthly columns
    results = []

    for index, row in df_pv.iterrows():
        pv_self_cons_kWh = 0
        pv_to_grid_kWh = 0
        pv_self_cons_euro = 0
        pv_to_grid_euro = 0
        
        for month in monthly_columns:
            monthly_generation = row[month]
            buy_price = df_prices.loc[month - 1, 'Electricity Buy']  # Match price by month
            sell_price = df_prices.loc[month - 1, 'Electricity Sell']
            
            # Get the self-consumption percentage for the current month
            month_name = df_prices.loc[month - 1, 'Month']
            self_cons_percent = monthly_self_cons_percentages[month_name]
            
            # Self-consumed and exported energy
            self_consumed_energy_kWh = monthly_generation * self_cons_percent
            exported_energy_kWh = monthly_generation * (1 - self_cons_percent)
            
            # Add to total kWh
            pv_self_cons_kWh += self_consumed_energy_kWh
            pv_to_grid_kWh += exported_energy_kWh
            
            # Monetary values
            pv_self_cons_euro += self_consumed_energy_kWh * buy_price
            pv_to_grid_euro += exported_energy_kWh * sell_price
        
        # Append results
        results.append({
            'build_id': row['build_id'],
            'census_id': row['census_id'],
            'PV_self_cons_kWh': pv_self_cons_kWh,
            'PV_to_grid_kWh': pv_to_grid_kWh,
            'PV_self_cons_Euro': pv_self_cons_euro,
            'PV_to_grid_Euro': pv_to_grid_euro
        })
    
    return pd.DataFrame(results)

# Calculate and merge results
df_pv_results = calculate_pv_metrics(df_pv, df_prices, monthly_self_cons_percentages)
df_pv = df_pv.merge(df_pv_results, on=['build_id', 'census_id'])

# Display results
print(df_pv[['build_id', 'census_id', 'PV_self_cons_kWh', 'PV_to_grid_kWh', 'PV_self_cons_Euro', 'PV_to_grid_Euro']])

# %% EXAMPLE STACKPLOT
data = {
    "Months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "4802003003": [12262.0, 15716.0, 24622.0, 29748.0, 34731.0, 35146.0, 35511.0, 31794.0, 25243.0, 19425.0, 13165.0, 11358.0],
    "4802003005": [21682.0, 27708.0, 43283.0, 52160.0, 60784.0, 61458.0, 62124.0, 55710.0, 44333.0, 34215.0, 23262.0, 20113.0],
    "4802003006": [11180.0, 14087.0, 21696.0, 25813.0, 29815.0, 30026.0, 30405.0, 27448.0, 22077.0, 17275.0, 11932.0, 10448.0],
    "4802003007": [12909.0, 16284.0, 25107.0, 29896.0, 34549.0, 34802.0, 35234.0, 31794.0, 25553.0, 19976.0, 13781.0, 12058.0],
    "4802003009": [18347.0, 22880.0, 34853.0, 41069.0, 47125.0, 47334.0, 47994.0, 43569.0, 35338.0, 27959.0, 19535.0, 17250.0],
    "4802003010": [9331.0, 11995.0, 18848.0, 22837.0, 26709.0, 27048.0, 27320.0, 24430.0, 19354.0, 14849.0, 10030.0, 8626.0],
    "4802003011": [22518.0, 28727.0, 44806.0, 53930.0, 62794.0, 63475.0, 64168.0, 57573.0, 45858.0, 35443.0, 24142.0, 20908.0],
}
def plot_census_stackplot(dataframe, title="Stackplot", xlabel="X-axis", ylabel="Y-axis", font_size=14):
    """
    Plots a stackplot for the given DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame where columns are x-axis values (i.e., Months, int type) 
                                  and rows are groups for stacking (i.e., Census IDs, str type).
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    dataframe.columns = dataframe.columns.astype(str)

    # Extract x-axis values and y-axis groups from the DataFrame
    x = dataframe.index  # x-axis (e.g., Months)
    y = dataframe.values.T  # y-axis values for stacking (transpose for stackplot)

    # Plot the stackplot
    plt.figure(figsize=(12, 6))
    plt.stackplot(x, y, labels=dataframe.columns)
    plt.title(title)
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)
    plt.legend(loc="upper left", title="Census ID")
    plt.xticks(x, rotation=0, fontsize=font_size)
    plt.yticks(rotation=0, fontsize=font_size)  # Remove x-axis values from y-ticks
    plt.tight_layout()
    plt.show()

# Convert to DataFrame
df_pv_temp = pd.DataFrame(data).set_index("Months")

plot_census_stackplot(df_pv_temp, title="Estimated Monthly PV Generation by Census", xlabel="Months", ylabel="PV Electricity (kWh)")


# %%
