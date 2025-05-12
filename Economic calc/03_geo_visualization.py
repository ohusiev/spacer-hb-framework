
#%%
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

census_file= r'H:\My Drive\01_PHD_CODE\chapter_5\spain\bilbao_otxarkoaga\vector\stat_census\Otxarkoaga.shp'
df_data = pd.read_excel(r'H:\My Drive\01_PHD_CODE\chapter_5\spain\bilbao_otxarkoaga\data\06_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV_economic+EUAC_ORIGIN_ADD_SHEET.xlsx', sheet_name='Sheet3', dtype={'census_id': str})

df_census = gpd.read_file(census_file)

df_filter = df_data[df_data["Filter"]=='Comb2S1']
#df_census['census_id'] = df_census['census_id'].astype(str)
df = df_census.merge(df_filter, on='census_id')

#%%
#1. Heatmap of Energy Demand and CO₂ Emissions by Census Section (Geospatial)

# Merging energy data with GeoDataFrame
#gdf_sample = gdf_sample.merge(df[['census_id', 'Envelope_Energy_kWh', 'CO2_per_m2']], on='census_id')
# Plot Energy Demand Heatmap
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Energy demand heatmap
df.plot(column='Envelope_Energy_kWh', cmap='Reds', legend=True, ax=ax[0], edgecolor='black', linewidth=0.5)
ax[0].set_title("Heatmap of Energy Demand by Census Section")
# Adding labels for each element
for x, y, label in zip(df.geometry.centroid.x, df.geometry.centroid.y, df['Envelope_Energy_kWh']):
    ax[0].text(x, y, round(label, 2), fontsize=8, ha='center', va='center', color='black')

# CO2 Emissions heatmap
df.plot(column='CO2_per_m2', cmap='Blues', legend=True, ax=ax[1], edgecolor='black', linewidth=0.5)
ax[1].set_title("Heatmap of CO₂ Emissions by Census Section")
# remove labes of y and x axis
ax[0].set_yticklabels([])
ax[0].set_xticklabels([])
ax[1].set_yticklabels([])
ax[1].set_xticklabels([])
# Adding labels for each element
for x, y, label in zip(df.geometry.centroid.x, df.geometry.centroid.y, df['CO2_per_m2']):
    ax[1].text(x, y, round(label, 2), fontsize=8, ha='center', va='center', color='black')

plt.show()
#%% AdJUSTED HEATMAPS with non ajustable colorbar
def geo_heatmap(df, column_1='NRPE_kWh_per_dwelling', column_2='EUAC_per_dwelling'):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Coumn_1 heatmap
    df.plot(column=column_1, cmap='autumn', legend=True, ax=ax[0], edgecolor='black', linewidth=0.5)
    ax[0].set_title(f"Heatmap of `{column_1}` by Census Section")
    # Adding labels for each element
    for x, y, label in zip(df.geometry.centroid.x, df.geometry.centroid.y, df[column_1]):
        ax[0].text(x, y, round(label, 2), fontsize=8, ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))

    # Column_2 heatmap
    df.plot(column=column_2, cmap='summer', legend=True, ax=ax[1], edgecolor='black', linewidth=0.5)
    ax[1].set_title(f"Heatmap of `{column_2}` by Census Section")
    # remove labes of y and x axis
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    """# Add label for colorbar
    legend = ax[0].get_legend()
    if legend:
        legend.set_title(column_1)"""
    # Adding labels for each element
    for x, y, label in zip(df.geometry.centroid.x, df.geometry.centroid.y, df[column_2]):
        ax[1].text(x, y, round(label, 2), fontsize=8, ha='center', va='center', color='black',  bbox=dict(facecolor='white', alpha=0.5))#, backgroundcolor='white')

    plt.subplots_adjust(wspace=0.008)  # Adjusts the width spacing between subplots

    plt.show()

geo_heatmap(df,column_1='NRPE_kWh_per_dwelling', column_2='EUAC_per_dwelling')

#%% ADJUSTED HEATMAPS WITH adjustable colorbar
def geo_heatmap(df, column_1='NRPE_kWh_per_dwelling', column_2='EUAC_per_dwelling', vmin1=None, vmax1=None, vmin2=None, vmax2=None):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Column_1 heatmap
    df.plot(column=column_1, cmap='autumn', legend=True, ax=ax[0], edgecolor='black', linewidth=0.5, vmin=vmin1, vmax=vmax1)
    ax[0].set_title(f"Heatmap of `{column_1}` by Census Section")
    # Adding labels for each element
    for x, y, label in zip(df.geometry.centroid.x, df.geometry.centroid.y, df[column_1]):
        ax[0].text(x, y, round(label, 2), fontsize=8, ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))

    # Column_2 heatmap
    df.plot(column=column_2, cmap='summer', legend=True, ax=ax[1], edgecolor='black', linewidth=0.5, vmin=vmin2, vmax=vmax2)
    ax[1].set_title(f"`{column_2}`")
    # remove labels of y and x axis
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    # Adding labels for each element
    for x, y, label in zip(df.geometry.centroid.x, df.geometry.centroid.y, df[column_2]):
        ax[1].text(x, y, round(label, 2), fontsize=8, ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))

    plt.subplots_adjust(wspace=0.008)  # Adjusts the width spacing between subplots

    plt.show()

geo_heatmap(df,column_1='NRPE_Envelope_kWh_per_m2', column_2='Envelope_EUAC_per_m2', vmin1=0, vmax1=200, vmin2=0, vmax2=50)
#%%
#2. Cost vs. Benefit Scatter Plot (EUAC vs. Energy Savings)

import seaborn as sns

# Scatter plot of Cost vs. Energy Savings
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Envelope_EUAC', y='Envelope_Energy_kWh', hue='CO2_per_m2', size='CO2_per_m2', sizes=(20, 200), palette="coolwarm")
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("Envelope EUAC (Cost)")
plt.ylabel("Energy Savings (kWh)")
plt.title("Cost vs. Energy Savings per Census Section")
plt.legend(title="CO₂ Emissions (per m²)")
plt.grid()
plt.show()
#%%
#3. Boxplot of EUAC per Dwelling Across Census Sections

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='census_id', y='EUAC_per_dwelling')
plt.xticks(rotation=90)
plt.xlabel("Census Section")
plt.ylabel("EUAC per Dwelling")
plt.title("Distribution of EUAC per Dwelling Across Census Sections")
plt.grid()
plt.show()
#%%
#4. Standard Deviation of Cost and Energy Savings (Variability Analysis)
# Compute standard deviations
std_df = df.groupby('census_id')[['Envelope_EUAC', 'Envelope_Energy_kWh']].std().reset_index()

# Bar plot showing standard deviation of cost and savings
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(data=std_df, x='census_id', y='Envelope_EUAC', ax=ax[0], color='red')
ax[0].set_title("Variability in Envelope EUAC per Census Section")
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)

sns.barplot(data=std_df, x='census_id', y='Envelope_Energy_kWh', ax=ax[1], color='blue')
ax[1].set_title("Variability in Energy Savings per Census Section")
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)

plt.show()

#%% Sample Data for GeoDataFrame
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

# Sample data similar to the original dataset
sample_data = {
    'census_id': ['4802003003', '4802003005', '4802003006', '4802003007', '4802003009'],
    'Envelope_EUAC': np.random.randint(20000, 150000, size=5),
    'EUAC': np.random.randint(20000, 150000, size=5),
    'Envelope_Energy_kWh': np.random.randint(500000, 2000000, size=5),
    'Total_kWh': np.random.randint(200000, 600000, size=5),
    'EUAC_per_dwelling': np.random.randint(50, 300, size=5),
    'Envelope_EUAC_per_dwelling': np.random.randint(50, 300, size=5),
    'CO2_per_m2': np.random.uniform(5, 15, size=5)
}

# Convert to DataFrame
df_sample = pd.DataFrame(sample_data)

# Creating example geometries (polygon areas for census sections)
geometry = [
    Polygon([(-113.5, 53.5), (-113.4, 53.5), (-113.4, 53.6), (-113.5, 53.6)]),
    Polygon([(-113.6, 53.5), (-113.5, 53.5), (-113.5, 53.6), (-113.6, 53.6)]),
    Polygon([(-113.7, 53.5), (-113.6, 53.5), (-113.6, 53.6), (-113.7, 53.6)]),
    Polygon([(-113.8, 53.5), (-113.7, 53.5), (-113.7, 53.6), (-113.8, 53.6)]),
    Polygon([(-113.9, 53.5), (-113.8, 53.5), (-113.8, 53.6), (-113.9, 53.6)])
]

# Creating a GeoDataFrame
gdf_sample = gpd.GeoDataFrame(df_sample, geometry=geometry)

# Assign a Coordinate Reference System (CRS) for proper mapping
gdf_sample.set_crs(epsg=4326, inplace=True)

# Display the sample GeoDataFrame

# %%
# Function to calculate standard deviation of EUAC for specific scenarios
def calculate_high_std_dev(df, column, filter_column, scenarios):
    """
    Calculates the standard deviation of a specified column (e.g., EUAC) for given scenarios.

    Parameters:
    - df (pd.DataFrame): DataFrame containing data.
    - column (str): Column name to calculate standard deviation for (e.g., 'EUAC').
    - filter_column (str): Column that contains scenario labels (e.g., 'Filter').
    - scenarios (list): List of scenario names to filter data (e.g., ['Comb2S1', 'Comb2S2']).

    Returns:
    - float: Standard deviation of the specified column for the given scenarios.
    """
    filtered_df = df[df[filter_column].isin(scenarios)]
    return filtered_df[column].std()


# Calculate standard deviation of EUAC for Comb2S1 and Comb2S2
scenarios_to_check = ['Comb2S1', 'Comb2S2']
std_dev_euac = calculate_high_std_dev(df, 'EUAC', 'Filter', scenarios_to_check)

# Display result
std_dev_euac

# %%
