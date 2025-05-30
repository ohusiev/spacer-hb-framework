#%%
import pandas as pd
import os
import sys

import importlib

module_name = "Economic calc.util_func"
util_func = importlib.import_module(module_name)
df_sensitivity_total =pd.DataFrame()
df_sensitivity_vulner_zones_monthly = pd.DataFrame()
# For testing, generate sample data
#dataframe1, dataframe2, dataframe3, dataframe4, dataframe5, dataframe6 = test_data_generator()
#%%
pv_pct = 0.25  # percentage of dwellings with PV [0.25, 0.5, 0.75, 1]
no_pv_pct = 1 - pv_pct
# Load your dataframes (replace 'your_path' with the actual file path or data source)
#dataframe1 = pd.read_csv('data/04_energy_consumption_profiles/dwell_share_1/04_aggregated_pv_gen_by_census_id_monthly.csv')  # PV generation data per building, monthly
pv_file_name = "01_footprint_s_area_wb_rooftop_analysis_pv_month_pv.xlsx"
pv_path = os.path.join("02_pv_calc_from_bond_rooftop/data/Otxarkoaga/",pv_file_name)
# Load the data with columns as integers
dataframe1 = pd.read_excel(pv_path, sheet_name='Otxarkoaga' ,dtype={'census_id': str, 'census_id': str})
dataframe1 = dataframe1.groupby('census_id').sum().reset_index()

#dataframe2 = pd.read_csv('your_path_to_dataframe2.csv')  # census data, NumDw and NumBui_TOT
dataframe3 = pd.read_csv(f'data/04_energy_consumption_profiles/dwell_share_{pv_pct}/04_aggreg_cons_prof_with_pv_by_census_id_monthly_{pv_pct}.csv',dtype={'census_id': str})  # monthly consumption of PV households per census_id
dataframe7=pd.read_csv(f'data/04_energy_consumption_profiles/dwell_share_{pv_pct}/04_aggreg_cons_prof_no_pv_by_census_id_monthly_{no_pv_pct}.csv',dtype={'census_id': str}) #monthly consumption of non-PV households per census_id

dataframe4 = pd.read_csv(f'data/04_energy_consumption_profiles/dwell_share_{pv_pct}/04_self_cons_pct_month_{pv_pct}.csv',dtype={'census_id': str})  # self-consumption % by census_id, monthly
dataframe5 = pd.read_csv(f'data/04_energy_consumption_profiles/dwell_share_{pv_pct}/04_cov_pct_no_pv_month_{no_pv_pct}.csv',dtype={'census_id': str})  # monthly consumption of non-PV households within potential EC per census_id
dataframe6 = pd.read_csv(f'data/04_energy_consumption_profiles/dwell_share_{pv_pct}/04_cov_pct_pv_month_{pv_pct}.csv',dtype={'census_id': str})  # monthly consumption of PV households within potential EC per census_id

df_prices = pd.read_excel("Economic calc/241211_econom_data.xlsx", sheet_name='cost_electricity',index_col=0)

# DOWNLOAD STATISTICAL DATA
#root=r"H:\My Drive\01_PHD_CODE\chapter_5\spain\bilbao_otxarkoaga"
root=r"D:\o.husiev@opendeusto.es\01_PHD_CODE\chapter_5\spain\bilbao_otxarkoaga"

stat_data = pd.read_excel(os.path.join(root, "data\\04_energy_consumption_profiles", "00_data_census_id_ener_consum_profiling.xlsx"), sheet_name='04_dwelling_profiles_census',  dtype={'census_id': str}, index_col=0)
stat_data.index = stat_data.index.astype(str)
#%%
# Step 1: Aggregate PV generation per census_id by month
# For simplicity, assuming dataframe1 contains monthly columns (e.g., 'gen_m1', 'gen_m2', ... 'gen_m12')
# rename the columns of dataframe1 to 'gen_m1', 'gen_m2', ... 'gen_m12'
#rename the columns of dataframe1 to 'gen_m1', 'gen_m2', ... 'gen_m12'
dataframe1.rename(columns={i: f'gen_m{i}' for i in range(1, 13)}, inplace=True)
monthly_gen_cols = [f'gen_m{i}' for i in range(1, 13)]
pv_gen = dataframe1.groupby('census_id')[monthly_gen_cols].sum().reset_index()

# Step 2: Consumption profile and self-consumption percentage per census_id, monthly
consumption_profile = dataframe3  # Each column represents monthly consumption (e.g., 'cons_m1', 'cons_m2', ... 'cons_m12')
self_consumption = dataframe4  # Each column represents monthly self-consumption % (e.g., 'self_m1', 'self_m2', ... 'self_m12')
cons_no_pv_cov = dataframe5 # Each column represents monthly consumption % of non-PV households within potential EC (e.g., 'no_pv_cov_m1', 'no_pv_cov_m2', ... 'no_pv_cov_m12')
cons_pv_cov = dataframe6 # Each column represents monthly consumption % of PV households within potential EC (e.g., 'with_pv_cov_m1', 'with_pv_cov_m2', ... 'with_pv_cov_m12')
cons_no_pv = dataframe7 # Each column represents monthly consumption of non-PV households (e.g., 'cons_no_pv_m1', 'cons_no_pv_m2', ... 'cons_no_pv_m12')
# Merge PV generation, consumption, and self-consumption data on census_id
energy_data = pd.merge(pv_gen, consumption_profile, on='census_id')
energy_data = pd.merge(energy_data, self_consumption, on='census_id')
energy_data = pd.merge(energy_data, cons_no_pv_cov, on='census_id')
energy_data = pd.merge(energy_data, cons_pv_cov, on='census_id')
energy_data = pd.merge(energy_data, cons_no_pv, on='census_id')
# Step 2.1: Define electricity prices

p_electr = 0.2  # EUR/kWh, conventional electricity price
p_feed = 0.05    # EUR/kWh, feed-in tariff for surplus PV
p_electr_EC_buy = 0.15  # EUR/kWh, buy price of electricity within the EC
p_electr_EC_sell = 0.1  # EUR/kWh, sell price of electricity  within EC

# Just assume that monthly the price within EC will be approx 20% lower to buy than conventional electricity price and 20% higher to sell than conventional feed-in tariff
ec_price_coef = 0.5  # EC pricing impact factor


#%%
# Step 3: Calculate self-consumption (direct consumption) and residuals
for i in range(1, 13):
    p_electr = df_prices.loc[i,'Electricity Buy']  # EUR/kWh, conventional electricity price
    p_feed = df_prices.loc[i,'Electricity Sell']    # EUR/kWh, feed-in tariff for surplus PV
    p_electr_EC_buy = df_prices.loc[i,'Electricity Buy'] *(1-ec_price_coef)  # EUR/kWh, buy price of electricity within the EC
    p_electr_EC_sell = df_prices.loc[i,'Electricity Sell'] *(1+ec_price_coef) # EUR/kWh, sell price of electricity  within EC
    gen_col = f'gen_m{i}'
    cons_col = f'cons_m{i}'
    cons_no_pv_col = f'no_pv_cons_m{i}'
    self_col = f'self_m{i}'
    no_pv_pct_cov_col = f'no_pv_cov_m{i}'
    pv_pct_cov_col = f'with_pv_cov_m{i}'
    
    # Direct self-consumption
    energy_data[f'gen_used_dir_m{i}'] = energy_data[gen_col] * energy_data[self_col]
    energy_data[f'load_cov_dir_m{i}'] = energy_data[f'gen_used_dir_m{i}']
    
    # Residual load and surplus PV generation after direct consumption
    energy_data[f'gen_surpl_dir_m{i}'] = energy_data[gen_col] - energy_data[f'load_cov_dir_m{i}']
    energy_data[f'load_resid_dir_m{i}'] = energy_data[cons_col] - energy_data[f'load_cov_dir_m{i}'] # it can be that the residual load is negative if the consumption is lower than the direct consumption
    
    # Load covered within EC for non-PV households
    energy_data[f'load_cov_EC_BwoPV_m{i}'] = energy_data[gen_col] * energy_data[no_pv_pct_cov_col] #change this
    energy_data[f'load_resid_EC_BwoPV_m{i}'] = energy_data[cons_no_pv_col] - energy_data[f'load_cov_EC_BwoPV_m{i}'] #change this
    
    # Load covered within EC for PV households
    energy_data[f'load_cov_EC_BwPV_m{i}'] = energy_data[gen_col] * energy_data[pv_pct_cov_col] #change this
    energy_data[f'load_resid_EC_BwPV_m{i}'] = energy_data[f'load_resid_dir_m{i}'] - energy_data[f'load_cov_EC_BwPV_m{i}'] #change this
    
    # Surplus generation after local consumption within the EC
    energy_data[f'gen_surpl_EC_m{i}'] = energy_data[f'gen_surpl_dir_m{i}'] - energy_data[f'load_cov_EC_BwPV_m{i}'] - energy_data[f'load_cov_EC_BwoPV_m{i}']

    # Set possible negative values to zero
    energy_data[f'gen_surpl_dir_m{i}'] = energy_data[f'gen_surpl_dir_m{i}'].clip(lower=0)
    energy_data[f'load_resid_dir_m{i}'] = energy_data[f'load_resid_dir_m{i}'].clip(lower=0)
    energy_data[f'load_cov_EC_BwoPV_m{i}'] = energy_data[f'load_cov_EC_BwoPV_m{i}'].clip(lower=0)
    energy_data[f'load_resid_EC_BwoPV_m{i}'] = energy_data[f'load_resid_EC_BwoPV_m{i}'].clip(lower=0)
    energy_data[f'load_cov_EC_BwPV_m{i}'] = energy_data[f'load_cov_EC_BwPV_m{i}'].clip(lower=0)
    energy_data[f'load_resid_EC_BwPV_m{i}'] = energy_data[f'load_resid_EC_BwPV_m{i}'].clip(lower=0)
    energy_data[f'gen_surpl_EC_m{i}'] = energy_data[f'gen_surpl_EC_m{i}'].clip(lower=0)

    # Conventional costs without EC
    energy_data[f'C_BwPV_m{i}'] = energy_data[f'load_resid_dir_m{i}'] * p_electr - energy_data[f'gen_surpl_dir_m{i}'] * p_feed
    energy_data[f'C_BwoPV_m{i}'] = energy_data[cons_no_pv_col] * p_electr
    if pv_pct ==1:
        energy_data[f'C_BwoPV_m{i}'] = energy_data[cons_col] * p_electr
    # Costs with EC for PV households
    energy_data[f'C_EC_BwPV_m{i}'] = (
        energy_data[f'load_resid_EC_BwPV_m{i}'] * p_electr
        + energy_data[f'load_cov_EC_BwPV_m{i}'] * p_electr_EC_buy
        - (energy_data[f'load_cov_EC_BwPV_m{i}'] + energy_data[f'load_cov_EC_BwoPV_m{i}']) * p_electr_EC_sell
        - energy_data[f'gen_surpl_EC_m{i}'] * p_feed
    )
    
    # Costs with EC for non-PV households
    energy_data[f'C_EC_BwoPV_m{i}'] = (
        energy_data[f'load_resid_EC_BwoPV_m{i}'] * p_electr
        + energy_data[f'load_cov_EC_BwoPV_m{i}'] * p_electr_EC_buy
    )
    if pv_pct ==1:
        energy_data[f'C_EC_BwPV_m{i}'] = (
        energy_data[f'load_resid_EC_BwPV_m{i}'] * p_electr
        + energy_data[f'load_cov_EC_BwPV_m{i}'] * p_electr_EC_buy
        - (energy_data[f'load_cov_EC_BwPV_m{i}'] + energy_data[f'load_cov_EC_BwoPV_m{i}']) * p_electr_EC_sell
        - energy_data[f'gen_surpl_EC_m{i}'] * p_feed
    )

#%%
energy_data = energy_data.set_index('census_id')

#%% Create Excel writer
with pd.ExcelWriter(f'EC_costs_results_{pv_pct}_price_diff_{ec_price_coef}.xlsx') as writer:
    # Sheet 1: gen_consumption
    gen_consumption_cols = [f'gen_m{i}' for i in range(1, 13)] + [f'cons_m{i}' for i in range(1, 13)] + [f'self_m{i}' for i in range(1, 13)] + [f'gen_used_dir_m{i}' for i in range(1, 13)] + [f'load_cov_dir_m{i}' for i in range(1, 13)] + [f'gen_surpl_dir_m{i}' for i in range(1, 13)] + [f'load_resid_dir_m{i}' for i in range(1, 13)]
    energy_data[['census_id'] + gen_consumption_cols].to_excel(writer, sheet_name='gen_consumption', index=False)
    
    # Sheet 2: loadsinEC
    loads_inEC_cols = [f'load_cov_EC_BwoPV_m{i}' for i in range(1, 13)] + [f'load_resid_EC_BwoPV_m{i}' for i in range(1, 13)] + [f'load_cov_EC_BwPV_m{i}' for i in range(1, 13)] + [f'load_resid_EC_BwPV_m{i}' for i in range(1, 13)] + [f'gen_surpl_EC_m{i}' for i in range(1, 13)]
    energy_data[['census_id'] + loads_inEC_cols].to_excel(writer, sheet_name='loads_within_EC', index=False)
    
    # Sheet 3: costs_without_EC
    costs_without_EC_cols = [f'C_BwPV_m{i}' for i in range(1, 13)] + [f'C_BwoPV_m{i}' for i in range(1, 13)]
    energy_data[['census_id'] + costs_without_EC_cols].to_excel(writer, sheet_name='costs_without_EC', index=False)
    
    # Sheet 4: cost_with_EC
    cost_with_EC_cols = [f'C_EC_BwPV_m{i}' for i in range(1, 13)] + [f'C_EC_BwoPV_m{i}' for i in range(1, 13)]
    energy_data[['census_id'] + cost_with_EC_cols].to_excel(writer, sheet_name='costs_with_EC', index=False)

energy_data = energy_data.set_index('census_id')
#%%
# Plotting
df_total_cost = pd.DataFrame(index = energy_data.index)
df_total_cost['Avg Costs, dwellings with PV'] = energy_data[[f'C_BwPV_m{i}' for i in range(1, 13)]].sum(axis=1)
df_total_cost['Avg Costs, dwellings without PV'] = energy_data[[f'C_BwoPV_m{i}' for i in range(1, 13)]].sum(axis=1)
df_total_cost['Avg Costs in EC, dwellings with PV'] = energy_data[[f'C_EC_BwPV_m{i}' for i in range(1, 13)]].sum(axis=1)
df_total_cost['Avg Costs in EC, dwellings without PV'] = energy_data[[f'C_EC_BwoPV_m{i}' for i in range(1, 13)]].sum(axis=1)
#%%
util_func.EconomicAnalysisGraphs.plot_ec_costs(df_total_cost, C_ec_b_PV='Avg Costs in EC, dwellings with PV',C_ec_b_nPV = 'Avg Costs in EC, dwellings without PV', C_cov_b_PV ='Avg Costs, dwellings with PV', C_cov_b_nPV= 'Avg Costs, dwellings without PV')

#%%
df_total_cost['Avg Costs, dwellings with PV'] = df_total_cost['Avg Costs, dwellings with PV'].div(stat_data['Total number of dwellings'] * float(pv_pct), axis=0)

if pv_pct ==1:
    df_total_cost['Avg Costs, dwellings without PV'] = (
    df_total_cost['Avg Costs, dwellings without PV'].div(stat_data['Total number of dwellings'] * float(pv_pct), axis=0)
    )
else:
    df_total_cost['Avg Costs, dwellings without PV'] = (
    df_total_cost['Avg Costs, dwellings without PV'].div(stat_data['Total number of dwellings'] * float(1 - pv_pct), axis=0)
    )

df_total_cost['Avg Costs in EC, dwellings with PV'] = (
    df_total_cost['Avg Costs in EC, dwellings with PV'].div(stat_data['Total number of dwellings'] * float(pv_pct), axis=0)
)
if pv_pct ==1:
    df_total_cost['Avg Costs in EC, dwellings without PV'] = (
        df_total_cost['Avg Costs in EC, dwellings without PV'].div(stat_data['Total number of dwellings'] * float(pv_pct), axis=0)
    )
else:
    df_total_cost['Avg Costs in EC, dwellings without PV'] = (
        df_total_cost['Avg Costs in EC, dwellings without PV'].div(stat_data['Total number of dwellings'] * float(1 - pv_pct), axis=0)
    )


#%%
if pv_pct ==1:
    df_total_cost['Savings, Dwell with PV, %'] = df_total_cost.apply(
    lambda row: (
        (1 - (row['Avg Costs in EC, dwellings with PV'] / row['Avg Costs, dwellings without PV'])) * 100
        if (row['Avg Costs, dwellings with PV'] * row['Avg Costs in EC, dwellings without PV'] < 0)  # Different signs
        else ((row['Avg Costs, dwellings without PV'] - row['Avg Costs in EC, dwellings with PV']) / row['Avg Costs, dwellings without PV']) * 100
    ) if row['Avg Costs, dwellings with PV'] != 0 else 0,  # Avoid division by zero
    axis=1
    )
    df_total_cost['Savings, Dwell without PV, %'] = 0
else:
    df_total_cost['Savings, Dwell with PV, %'] = df_total_cost.apply(
        lambda row: (
            (1 - (row['Avg Costs in EC, dwellings with PV'] / row['Avg Costs, dwellings with PV'])) * 100
            if (row['Avg Costs, dwellings with PV'] * row['Avg Costs in EC, dwellings with PV'] < 0)  # Different signs
            else ((row['Avg Costs, dwellings with PV'] - row['Avg Costs in EC, dwellings with PV']) / row['Avg Costs, dwellings with PV']) * 100
        ) if row['Avg Costs, dwellings with PV'] != 0 else 0,  # Avoid division by zero
        axis=1
    )

    df_total_cost['Savings, Dwell without PV, %'] = df_total_cost.apply(
        lambda row: (
            (1 - (row['Avg Costs in EC, dwellings without PV'] / row['Avg Costs, dwellings without PV'])) * 100
            if (row['Avg Costs, dwellings without PV'] * row['Avg Costs in EC, dwellings without PV'] < 0)  # Different signs
            else ((row['Avg Costs, dwellings without PV'] - row['Avg Costs in EC, dwellings without PV']) / row['Avg Costs, dwellings without PV']) * 100
        ) if row['Avg Costs, dwellings without PV'] != 0 else 0,  # Avoid division by zero
        axis=1
    )

#df_total_cost['Savings, Dwell without PV, %'] = ((df_total_cost['Avg Costs, dwellings without PV'] - df_total_cost['Avg Costs in EC, dwellings without PV']) / df_total_cost['Avg Costs, dwellings without PV'])*100

# Make a plot of savings column 
plot = df_total_cost[['Savings, Dwell with PV, %', 'Savings, Dwell without PV, %']].plot(kind = 'bar', ylabel = "Savings in %", figsize=(10, 6), title = f"Savings in % for {pv_pct*100}% of dwellings with PV, price_diff:{ec_price_coef*100}%", color = ['grey', 'black'])
plot.set_xticklabels(df_total_cost.index, rotation=45)

util_func.EconomicAnalysisGraphs.plot_ec_costs(df_total_cost, C_ec_b_PV='Avg Costs in EC, dwellings with PV',C_ec_b_nPV = 'Avg Costs in EC, dwellings without PV', C_cov_b_PV ='Avg Costs, dwellings with PV', C_cov_b_nPV= 'Avg Costs, dwellings without PV')

#Append DF

df_total_cost['Dwellings share with PV'] = pv_pct
df_total_cost['Price diff'] = ec_price_coef
df_sensitivity_total = df_sensitivity_total.append(df_total_cost)

#%% Monthly Sensitivity Analysis
df_temp = pd.DataFrame()
df_temp = energy_data[[f'C_EC_BwoPV_m{i}' for i in range(1,13)]]
df_temp = pd.concat([df_temp, energy_data[[f'C_EC_BwPV_m{i}' for i in range(1, 13)]]], axis=1)

df_temp = pd.concat([df_temp, energy_data[[f'C_BwoPV_m{i}' for i in range(1, 13)]]], axis=1)
df_temp = pd.concat([df_temp, energy_data[[f'C_BwPV_m{i}' for i in range(1, 13)]]], axis=1)
df_temp['Dwellings share with PV'] = pv_pct
df_temp['Price diff'] = ec_price_coef
df_sensitivity_vulner_zones_monthly = pd.concat([df_sensitivity_vulner_zones_monthly,df_temp])

#%%
df_sensitivity_total.to_excel('sensitivity_analysis.xlsx')
#%% TEST of PLOT
import matplotlib.pyplot as plt
data = {
    'C_cov_b_PV': [1, 2, 3, 4],#C_cov_b_PV,#[1, 2, 3, 4],
    'C_cov_b_nPV': [4, 3, 2, 1],# C_cov_b_nPV,# [4, 3, 2, 1],
    'C_ec_b_PV': [2, 3, 1, 4],#C_ec_b_PV, # [2, 3, 1, 4],
    'C_ec_b_nPV':[3, 1, 4, 2]# C_ec_b_nPV, #[3, 1, 4, 2]
}
df = pd.DataFrame(data)
# Convert data to DataFrame , index = [0]
def plot_ec_costs(df):
    #df = pd.DataFrame(data)
#df = pd.DataFrame(data, index = [0])
# GRAPH
    fig, ax = plt.subplots()

# Plot bar plots for Variable1 and Variable2
    df[['C_ec_b_PV', 'C_ec_b_nPV']].plot(kind='bar', ax=ax, width=0.7)
    for i, row in df.iterrows():
        ax.text(i - 0.15, row['C_ec_b_PV'] + 0.05, round(row['C_ec_b_PV'], 2), color='black', ha='center')
        ax.text(i + 0.15, row['C_ec_b_nPV'] + 0.05, round(row['C_ec_b_nPV'], 2), color='black', ha='center')
# Plotting transparent bars for the first set of values
#ax.bar(df.index,df['C_cov_b_PV'], color='red', alpha=0.5, edgecolor='none',width=0.4)

# Plotting transparent bars for the second set of values
#ax.bar(df.index,df['C_cov_b_nPV'], color='blue', alpha=0.5, edgecolor='none',width=0.4)

# Plotting top sides of bars for the first set of values
    for i, value in enumerate(df['C_cov_b_PV']):
        plt.plot([i - 0.3, i + 0.3], [value, value], color='blue')

# Plotting top sides of bars for the second set of values
    for i, value in enumerate(df['C_cov_b_nPV']):
        plt.plot([i - 0.3, i + 0.3], [value, value], color='red')
# Plot scatter plots for Variable3 and Variable4
    ax.scatter(df.index, df['C_cov_b_PV'], color='red', marker='_', label='C_cov_b_nPV')
    ax.scatter(df.index, df['C_cov_b_nPV'], color='blue',marker='_', label='C_cov_b_PV')

# Set labels and title
    ax.set_xlabel('Use case')
    ax.set_ylabel('Costs EUR')
    ax.set_title('Comparing estimated costs')
    ax.set_xticklabels(df.index, rotation=0)
    ax.legend(loc='best',bbox_to_anchor=(1, 1)).set_visible(True)
    plt.show()

plot_ec_costs(df)

#util_func.EconomicAnalysisGraphs.plot_ec_costs(df)
#%% Sample data generator for testing
import numpy as np
import pandas as pd
def test_data_generator():

    # Generate sample data for dataframe1: PV generation per building per month
    np.random.seed(0)
    dataframe1 = pd.DataFrame({
        'build_id': np.arange(1, 21),  # 20 buildings
        'census_id': np.random.choice(['C1', 'C2', 'C3', 'C4'], 20),
        **{f'gen_m{i}': np.random.uniform(100, 500, 20) for i in range(1, 13)}
    })

    # Generate sample data for dataframe2: census data with number of dwellings and total buildings
    dataframe2 = pd.DataFrame({
        'census_id': ['C1', 'C2', 'C3', 'C4'],
        'NumDw': np.random.randint(50, 200, 4),
        'NumBui_TOT': np.random.randint(10, 20, 4)
    })

    # Generate sample data for dataframe3: monthly consumption profile per census_id
    dataframe3 = pd.DataFrame({
        'census_id': ['C1', 'C2', 'C3', 'C4'],
        **{f'cons_m{i}': np.random.uniform(2000, 6000, 4) for i in range(1, 13)}
    })

    # Generate sample data for dataframe4: self-consumption percentage per census_id per month
    dataframe4 = pd.DataFrame({
        'census_id': ['C1', 'C2', 'C3', 'C4'],
        **{f'self_m{i}': np.random.uniform(0.2, 0.4, 4) for i in range(1, 13)}
    })
    dataframe5 = pd.DataFrame({
        'census_id': ['C1', 'C2', 'C3', 'C4'],
        **{f'no_pv_cov_m{i}': np.random.uniform(0.05, 0.1, 4) for i in range(1, 13)}
    })
    dataframe6 = pd.DataFrame({
        'census_id': ['C1', 'C2', 'C3', 'C4'],
        **{f'with_pv_cov_m{i}': np.random.uniform(0.01, 0.1, 4) for i in range(1, 13)}
    })

    # Display generated data for verification
    return dataframe1, dataframe2, dataframe3, dataframe4, dataframe5, dataframe6
# %%
# %%
# Creating a 2x2 grid of Bar Plots for Savings Across EC Scenarios and Price Differences
vulnerable_zones=["4802003006", "4802003007","4802003009","4802003010"]
df = df_sensitivity_total.loc[df_sensitivity_total.index.isin(vulnerable_zones)]
df = df.reset_index()
# Setting up the figure with subplots
def plot_savings_distribution(df, values = "Savings, Dwell with PV, %",colormap="blue"):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    price_diff_values = df["Price diff"].unique()
    price_diff_values.sort()
# Iterate over each subplot and plot the corresponding bar plot
    for i, ax in enumerate(axes.flat):
        if i < len(price_diff_values):
            price_diff = price_diff_values[i]
            df_subset = df[df["Price diff"] == price_diff]
        
        # Creating a bar plot for Savings
            df_subset_pivot = df_subset.pivot(index="Dwellings share with PV", columns="census_id", values=values)
            df_subset_pivot.plot(kind="bar", ax=ax, colormap=colormap, alpha=0.7, width=0.8, fontsize=14)
        
        # Titles and labels
            ax.set_title(f"EC price sell: {int(price_diff*100)}% less grid feed-in, EC price buy: {int(price_diff*100)}% less grid electricity", fontsize=14)
            ax.set_xlabel("Dwellings Share with PV (%)" , fontsize=14)
            ax.set_ylabel("Savings (%)", fontsize=14)
            ax.grid(True)
            ax.legend(title="Census ID", loc="best")
            ax.set_xticklabels(labels=[int(s*100) for s in list(df_subset_pivot.index)], rotation=0, fontsize=14)
# Adjust layout for better spacing
    plt.tight_layout()
# Show the plot
    plt.show()

plot_savings_distribution(df,values = "Savings, Dwell with PV, %", colormap='cividis')

plot_savings_distribution(df,values = "Savings, Dwell without PV, %", colormap='copper')
#%% TEST PLOT LINE CHART
# Re-import necessary libraries after execution state reset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define initial electricity prices
p_electr = 0.2  # EUR/kWh, conventional electricity price
p_feed = 0.05    # EUR/kWh, feed-in tariff for surplus PV

# EC pricing assumptions
ec_buy_price = p_electr * 0.8  # 20% lower than conventional electricity price
ec_sell_price = p_feed * 1.2   # 20% higher than feed-in tariff

# EC price coefficient scenarios
ec_price_coef = [0.1, 0.25, 0.5]  # EC pricing impact factors

# Create sensitivity analysis dataset
price_scenarios = np.linspace(p_electr * 0.7, p_electr * 1.3, 10)  # Vary electricity prices between -30% to +30%
data = []

for coef in ec_price_coef:
    for price in price_scenarios:
        ec_cost = price * (1 - coef)  # EC price adjustment per coefficient
        grid_cost = p_electr  # Conventional grid cost remains constant
        savings = (grid_cost - ec_cost) / grid_cost * 100  # Savings percentage

        data.append([coef, price, ec_cost, grid_cost, savings])

# Convert to DataFrame
df = pd.DataFrame(data, columns=["EC_Coef", "Price_Scenario", "EC_Cost", "Grid_Cost", "Savings (%)"])

# Plot Sensitivity Analysis: Line Chart of Cost Variations
plt.figure(figsize=(10, 6))
for coef in ec_price_coef:
    subset = df[df["EC_Coef"] == coef]
    plt.plot(subset["Price_Scenario"], subset["EC_Cost"], marker='o', label=f"EC Cost (Coef {coef})")

plt.axhline(y=p_electr, color='r', linestyle='--', label="Conventional Grid Price")
plt.axhline(y=p_feed, color='g', linestyle='--', label="Feed-in Tariff")

plt.xlabel("Electricity Price Scenario (€/kWh)")
plt.ylabel("Cost (€/kWh)")
plt.title("Sensitivity Analysis: EC Cost vs. Conventional Grid Price")
plt.legend()
plt.grid(True)
plt.show()

#%% 
# Convert the provided dataset into a Pandas DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create DataFrame
df = df_sensitivity_total

# Plot Line Chart for Cost Comparison Across Different Dwelling Shares
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df,
    x="Dwellings share with PV",
    y="Savings, Dwell with PV, %",
    label="Avg Costs, dwellings with PV",
    marker="o"
)
sns.lineplot(
    data=df,
    x="Dwellings share with PV",
    y="Savings, Dwell without PV, %",
    label="Avg Costs, dwellings without PV",
    marker="s"
)

# Chart labels and title
plt.xlabel("Dwellings Share with PV")
plt.ylabel("Average Cost per Dwelling (€)")
plt.title("Cost Comparison Across Different Dwelling Shares with PV")
plt.legend()
plt.grid(True)

# Show plot
plt.show()

# %%
# Importing necessary libraries for 3D visualization
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Extract relevant data for the 3D Bar Plot
x = df["Dwellings share with PV"]  # X-axis: Participation levels (25%, 50%, 75%, 100%)
y = df["Price diff"]  # Y-axis: Price difference scenarios (0.1, 0.25, 0.5)
z_savings_with_pv = df["Savings, Dwell with PV, %"]  # Z-axis: Savings with PV
z_savings_without_pv = df["Savings, Dwell without PV, %"]  # Z-axis: Savings without PV

# Creating 3D Bar Plot
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection="3d")

# Plot bars for savings with PV
for i in range(len(x)):
    ax.bar3d(x[i], y[i], 0, 0.05, 0.05, z_savings_with_pv[i], shade=True)

# Labels and title
ax.set_xlabel("Dwellings Share with PV (%)")
ax.set_ylabel("Price Difference (EC Pricing Coefficient)")
ax.set_zlabel("Savings (%) - Dwellings with PV")
ax.set_title("3D Bar Plot: Savings vs. Dwellings Share & Price Difference")

# Show plot
plt.show()
#%%
