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
util= UtilFunctions()
root=r"D:\o.husiev@opendeusto.es\01_PHD_CODE\chapter_5\spain\bilbao_otxarkoaga"
#root=r"H:\My Drive\01_PHD_CODE\chapter_5\spain\bilbao_otxarkoaga"
# PV data
#pv_file = os.path.join(root, "02_pv_calc_from_bond_rooftop/data/Otxarkoaga/+nominal_capacity", "01_footprint_s_area_wb_rooftop_analysis_pv_month_pv.xlsx")
pv_file = os.path.join(root, "02_pv_calc_from_bond_rooftop/data/Otxarkoaga/", "01_footprint_s_area_wb_rooftop_analysis_pv_month_pv.xlsx")

df_pv_gen = pd.read_excel(pv_file, sheet_name='Otxarkoaga')
df_self_cons_pct = pd.DataFrame()

stat_data = pd.read_excel(os.path.join(root, "data\\04_energy_consumption_profiles", "00_data_census_id_ener_consum_profiling.xlsx"), sheet_name='04_dwelling_profiles_census', index_col=0)
# Facades data
df_facades = gpd.read_file(os.path.join(root, "data/05_buildings_with_energy_and_co2_values+HDemProj.geojson"))
#df_facades = df_facades.rename(columns={"CodEdifici": "build_id"})
df_pv_temp= df_pv_gen.groupby('build_id').sum().reset_index()
df_facades['build_id'] = df_facades['build_id'].astype(str)
df_pv_temp['build_id'] = df_pv_temp['build_id'].astype(str)
df_facades = pd.merge(df_facades, df_pv_temp[["build_id", "r_area", "installed_kWp", "n_panel", "Total, kWh"]], on="build_id", how='left')
df_facades[["r_area", "n_panel", "Total, kWh"]] = df_facades[["r_area", "n_panel", "Total, kWh"]].fillna(0)
## CALCULATION OF SELF CONSUMPTION PERCENTAGE

### Read the files with different percentage of dwelling included into an aggregated self-consumption profile
standart_month_cons_cols = {"cons_m1": 1, "cons_m2": 2, "cons_m3": 3, "cons_m4": 4, "cons_m5": 5, "cons_m6": 6, "cons_m7": 7, "cons_m8": 8, "cons_m9": 9, "cons_m10": 10, "cons_m11": 11, "cons_m12": 12}
standart_month_pv_gen_cols= {"self_m1": 1, "self_m2": 2, "self_m3": 3, "self_m4": 4, "self_m5": 5, "self_m6": 6, "self_m7": 7, "self_m8": 8, "self_m9": 9, "self_m10": 10, "self_m11": 11, "self_m12": 12}


#%%
files = os.listdir("self_cons_estim")
df_self_cons_pct, dwelling_accounted_pct =  util.load_data("self_cons_estim", files)

#%%
df_pv_gen_census = df_pv_gen.groupby('census_id').sum()
df_self_cons_calc = pd.DataFrame()

for i in dwelling_accounted_pct:
    temp_filter = df_self_cons_pct[df_self_cons_pct['%dwelling_accounted'] == i].copy()
    for key, value in standart_month_pv_gen_cols.items():
        temp_filter.loc[:, key ] = temp_filter.loc[:, key ].mul(df_pv_gen_census.loc[:, value], axis=0)
        #temp_filter.loc[:, value] = temp_filter.loc[:, value] * df_pv_gen_census.loc[:, key]
    df_self_cons_calc = pd.concat([df_self_cons_calc, temp_filter], axis=0)
#%% Write to excel and create csv separately
with pd.ExcelWriter("pv_self_cons_econom.xlsx", mode='w', engine='openpyxl') as writer:  
    df_self_cons_pct.to_excel(writer, sheet_name='self_cons_pct')
    df_self_cons_calc.to_excel(writer, sheet_name='self_cons_calc')

df_self_cons_pct.to_csv("pv_self_cons_pct.csv")
df_self_cons_calc.to_csv("pv_self_cons_calc.csv")

#%% CALCULATE AND PLOT AVERAGE SELF-CONSUMPTION PER Dwelling Percentage
df_self_cons_calc_per_dwelling = pd.DataFrame()
for i in dwelling_accounted_pct:
    df_temp = df_self_cons_calc.loc[df_self_cons_calc['%dwelling_accounted'] == i, df_self_cons_calc.columns[0:12]].div(stat_data['Total number of dwellings']*float(i), axis=0)
    df_temp['%dwelling_accounted'] = i
    df_temp=df_temp.dropna()
    df_self_cons_calc_per_dwelling = pd.concat([df_self_cons_calc_per_dwelling, df_temp], axis=0)

PlotFunctions.plot_self_consumption_trends(
    df_self_cons_calc_per_dwelling,
    df_self_cons_pct,
    xlabel='Months',
    ylabel='Total Self-Consumed Electricity (kWh)',
    header='Monthly Average PV Electricity Self-Consumption per dwelling Percentage by Scenario'
)
""" Prepare aggregated table of self-consumption per dwelling energy
scenarios = df_self_cons_calc_per_dwelling['%dwelling_accounted'].unique()
monthly_columns = [col for col in df_self_cons_calc_per_dwelling.columns if 'self_m' in col]

df_aggredated = pd.DataFrame()
for scenario in scenarios:
    scenario_data = df_self_cons_calc_per_dwelling[df_self_cons_calc_per_dwelling['%dwelling_accounted'] == scenario]
    monthly_totals = scenario_data[monthly_columns].mean()
    df_aggredated[scenario] = monthly_totals"""


#%% Fileter per census_id
CENSUS_ID= 4802003011#15

df_self_cons_calc_per_dwelling_filter= df_self_cons_calc_per_dwelling[df_self_cons_calc_per_dwelling.index ==CENSUS_ID]
df_self_cons_pct_filer= df_self_cons_pct[df_self_cons_pct.index ==CENSUS_ID]
#%%
PlotFunctions.plot_self_consumption_trends(
    df_self_cons_calc_per_dwelling_filter,
    df_self_cons_pct_filer,
    xlabel='Months',
    ylabel='Total Self-Consumed Electricity (kWh)',
    header=f'Monthly Average PV Electricity Self-Consumption in {CENSUS_ID} section per dwelling Percentage by Scenario'
)
print(f'Monthly Average PV Electricity Self-Consumption in {CENSUS_ID} section per dwelling Percentage by Scenario')
#%% Plotting function with adjustable y_min and y_max
def plot_self_consumption_trends(df_calc_self_cons_per_dwelling, df_self_cons_pct_, xlabel, ylabel, header, y_min=None, y_max=None):

    scenarios = df_calc_self_cons_per_dwelling['%dwelling_accounted'].unique()
    monthly_columns = [col for col in df_calc_self_cons_per_dwelling.columns if 'self_m' in col]

    plt.figure(figsize=(12, 6))
    for scenario in scenarios:
        scenario_data = df_calc_self_cons_per_dwelling[df_calc_self_cons_per_dwelling['%dwelling_accounted'] == scenario]
        monthly_totals = scenario_data[monthly_columns].mean()
        plt.plot(monthly_columns, monthly_totals, marker='o', label=f"Scenario {float(scenario)*100:.0f}%")

    plt.title(header)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(y_min, y_max)
    plt.xticks(rotation=45)
    plt.legend(title='% of Dwellings Accounted')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    for scenario in scenarios:
        scenario_data = df_self_cons_pct_[df_self_cons_pct_['%dwelling_accounted'] == scenario]
        monthly_averages = scenario_data[monthly_columns].mean()
        plt.plot(monthly_columns, monthly_averages, marker='o', label=f"Scenario {float(scenario)*100:.0f}%")

    plt.title('Monthly Self-Consumption Percentage by Scenario')
    plt.xlabel('Months')
    plt.ylabel('Average Self-Consumption Percentage')
    #plt.ylim(y_min, y_max)
    plt.xticks(rotation=45)
    plt.legend(title='% of Dwellings Accounted')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
#%%
plot_self_consumption_trends(
    df_self_cons_calc_per_dwelling_filter,
    df_self_cons_pct_filer,
    xlabel='Months',
    ylabel='Total Self-Consumed Electricity (kWh)',
    header=f'Monthly Average PV Electricity Self-Consumption in {CENSUS_ID} section per dwelling Percentage by Scenario', y_min=0, y_max=200
)
print(f'Monthly Average PV Electricity Self-Consumption in {CENSUS_ID} section per dwelling Percentage by Scenario')
#%% Upload consumption data from folder
files = os.listdir("cons")
df_consumption, dwelling_accounted_pct =  util.load_data("cons", files)

#%% Filter per dwelling percentage
df_results = pd.DataFrame() # Create an empty dataframe to store the results
#%%
#PCT_DWELLING = 0.5 # 1, 0.75, 0.5, 0.25
PCT_DWELLING = [1, 0.75, 0.5, 0.25]
for PCT_DWELLING in dwelling_accounted_pct:
    df_consumption_filter = df_consumption.loc[df_consumption["%dwelling_accounted"] == str(PCT_DWELLING)]
    df_pv_sef_consm_filter= df_self_cons_calc.loc[df_self_cons_calc["%dwelling_accounted"] == str(PCT_DWELLING)]

    df_consumption_filter=df_consumption_filter.rename(columns=standart_month_cons_cols)
    df_pv_sef_consm_filter=df_pv_sef_consm_filter.rename(columns=standart_month_pv_gen_cols)

    df_joined=pd.DataFrame()
    pv_generation = df_pv_gen_census.loc[:,1:12].sum()#.mean()
    consumption = df_consumption_filter.loc[:,1:12].sum()#.mean()
    self_cons = df_pv_sef_consm_filter.loc[:,1:12].sum()#.mean()

    df_joined['pv_generation'] = pv_generation
    df_joined['aggreg_consumption'] = consumption
    df_joined['PV_self_consumed'] = self_cons

    df_joined['Consumed_from_grid'] = df_joined['aggreg_consumption']-df_joined['PV_self_consumed']
    df_joined['Surplus_to_Grid'] = df_joined['pv_generation'] - df_joined['PV_self_consumed']
    df_joined['Surplus_to_Grid']=df_joined['Surplus_to_Grid'].clip(lower=0)
    df_plot = df_joined.transpose()
    title = f'Distribution of annual generation, consumption and self-consumption for {float(PCT_DWELLING)*100}% of dwellings'
    PlotFunctions.plot_combined_load_data_from_df_adj(df_plot, title,legend_show=False,y_min=None, y_max=1400000)
    #PlotFunctions.plot_combined_load_data_from_df(df_plot, title)
    #only for iteration the real data won't plot with extra columns
    df_joined['Filter'] = PCT_DWELLING
    df_joined["Month"] = df_joined.index
    df_results = pd.concat([df_results, df_joined], axis=0)

#Save to excel
df_results.to_excel("distribution of_self_cons_cons__pv_gen.xlsx")
#df_results = df_results.reset_index().melt(id_vars=["Filter", "Month"], var_name="Category", value_name="Value")
#%%
df_heatmap = df_self_cons_pct.copy()
df_heatmap['Total']=df_heatmap[list(standart_month_pv_gen_cols.keys())].mean(axis=1)*100 # Mean of the self-consumption percentage

df_heatmap=df_heatmap.pivot_table(index=df_heatmap.index, columns='%dwelling_accounted', values='Total')

df_heatmap =df_heatmap.rename(columns={'1': '100%', '0.75': '75%', '0.5': '50%', '0.25': '25%'})
# Sample Data for Scenario Matrix

import seaborn as sns
# Plot Heatmap
def heatmap(df_heatmap, title = "Scenario Matrix Heatmap: aggregated PV self-consumption vs Share of Dwellings participated in direct self-consumption",cbarname = 'Benefit Levels (%)',cmap = 'YlGnBu'):
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_heatmap, annot=True, cmap=cmap, cbar_kws={'label': cbarname}, fmt='.0f',vmin=0, vmax=100)#'.2f')
    plt.title(title)
    plt.xlabel('Scenarios')
    plt.ylabel('Census ID')
    plt.show()

heatmap(df_heatmap, cbarname = 'Self-Consumption Percentage (%)')
#%%
df_result_self_sufficiency = pd.DataFrame()
for PC_DWELLING in dwelling_accounted_pct:
    df_temp1 =df_self_cons_calc.loc[df_self_cons_calc['%dwelling_accounted'] == PC_DWELLING]
    df_temp2 =df_consumption.loc[df_consumption['%dwelling_accounted'] == PC_DWELLING]
    # standartize name of columns
    df_temp1=df_temp1.rename(columns=standart_month_pv_gen_cols)
    df_temp2=df_temp2.rename(columns=standart_month_cons_cols)
    #Devide df_temp1 by df_temp2
    df_temp1 = df_temp1.loc[:, standart_month_pv_gen_cols.values()].div(df_temp2.loc[:, standart_month_cons_cols.values()], axis=0)
    df_temp1['%dwelling_accounted'] = PC_DWELLING
    df_result_self_sufficiency = pd.concat([df_result_self_sufficiency, df_temp1], axis=0)

df_result_self_sufficiency['Total']=df_result_self_sufficiency[list(standart_month_pv_gen_cols.values())].mean(axis=1)*100 # Mean of the self-consumption percentage

df_result_self_sufficiency=df_result_self_sufficiency.pivot_table(index=df_result_self_sufficiency.index, columns='%dwelling_accounted', values='Total').round(2)
df_result_self_sufficiency =df_result_self_sufficiency.rename(columns={'1': '100%', '0.75': '75%', '0.5': '50%', '0.25': '25%'})
# Plot Heatmap
heatmap(df_result_self_sufficiency, title = "Scenario Matrix Heatmap: self-sufficiency", cbarname = 'Self-sSufficiency Percentage (%)', cmap='YlGnBu')

# Example data

# %% #Test plot with adjusted y_min and y_max
def plot_combined_load_data_from_df(df, title, y_min=None, y_max=1400000):
    """
    Plot load data as a combined individual and stacked bar chart using a DataFrame.

    The DataFrame should have:
    - Columns as categories (e.g., 'HH1', 'HH2', 'HH3', ...)
    - Index as labels (e.g., ['baseline load', 'total load', 'direct consumption', 'EC consumption', 'residual load']).
    
    Parameters:
        df (DataFrame): DataFrame with rows as labels and columns as categories.
        title (str): Title of the plot.
        y_min (float, optional): Minimum value for the y-axis.
        y_max (float, optional): Maximum value for the y-axis.
    """
    # Extract categories (column names) and labels (index)
    categories = df.columns.tolist()  # Categories are column names (e.g., HH1, HH2, HH3, ...)
    labels = df.index.tolist()  # Labels are the row names (e.g., baseline load, total load, etc.)

    # Number of categories (x-axis labels)
    n_categories = len(categories)

    # Bar width and x-axis positions
    bar_width = 0.3  # Width of each bar
    x_indices = np.arange(n_categories)

    # Colors for the bars (adjust if more labels are added)
    colors = ['gray', 'blue', 'orange', 'green', 'red']

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the baseline load as a separate bar
    ax.bar(x_indices, df.iloc[0], width=bar_width, label=labels[0], color=colors[0])

    # Plot the total load as another separate bar
    ax.bar(x_indices + bar_width, df.iloc[1], width=bar_width, label=labels[1], color=colors[1])

    # Plot the stacked components
    bottom = np.zeros(n_categories)
    for i in range(2, len(df)):
        ax.bar(x_indices + 2 * bar_width, df.iloc[i], width=bar_width, bottom=bottom, label=labels[i], color=colors[i])
        bottom += np.array(df.iloc[i])  # Update the bottom for stacking

    # Add labels, legend, and grid
    ax.set_xlabel('Month', fontsize=14)
    ax.set_ylabel('Load in kWh', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x_indices + 1.5 * bar_width)
    ax.set_xticklabels(categories, fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Set y-axis limits if provided
    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()
'''
# Sample data in DataFrame format
data = {
    'HH1': [7500, 8000, 2000, 100, 5900],
    'HH2': [6800, 7000, 1500, 80, 5420],
    'HH3': [8200, 8500, 1800, 120, 6580],
    'HH4': [8000, 8200, 1700, 110, 6390],
    'HH5': [7700, 7900, 1600, 90, 6210],
    'HH6': [7600, 7800, 1500, 70, 6230],
    'HH7': [7400, 7600, 1400, 80, 6120],
    'HH8': [7500, 7700, 1450, 85, 6165],
    'HH9': [7800, 8000, 1550, 95, 6355],
    'HH10': [7200, 7500, 1300, 75, 6125]
}

# Labels for rows
labels = ['baseline load', 'total load', 'load covered by direct consumption', 'load covered within EC', 'residual load']

# Create the DataFrame
df = pd.DataFrame(data, index=labels)

# Plot the data
plot_combined_load_data_from_df(df)
'''
# %%
