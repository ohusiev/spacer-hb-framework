#%%
import pandas as pd
import geopandas as gpd
import os, yaml
import numpy as np

#%% INPUT FILES:
data_struct_file = '02_pv_calc_from_bond_rooftop/pv_energy.yml'
with open(data_struct_file, 'r', encoding="utf-8") as f:
    data_struct = yaml.safe_load(f)

rayon = "otxarkoaga".lower()
work_dir = data_struct[rayon]['work_dir']

# SET SELF_CONSUMPTION PARAMETERS to be used in the file name
pv_pct = 0.25
no_pv_pct = 1 - pv_pct

# Load the dataframes of aggregated consumption profiles
aggregated_profiles_file_path = 'data/04_energy_consumption_profiles/'

df_aggregated_profiles = pd.read_csv(
    os.path.join(aggregated_profiles_file_path, f'dwell_share_{pv_pct}/04_2_aggregated_1h_profiles_with_pv_dwell_share_{pv_pct}.csv'),
    parse_dates=['Time']
)
df_aggregated_profiles.set_index('Time', inplace=True)

df_aggregated_profiles_no_pv = pd.read_csv(
    os.path.join(aggregated_profiles_file_path, f'dwell_share_{pv_pct}/04_2_aggregated_1h_profiles_no_pv_dwell_share_{no_pv_pct}.csv'),
    parse_dates=['Time']
)
df_aggregated_profiles_no_pv.set_index('Time', inplace=True)

# Load the pv generation aggregated profile
pv_file_path = os.path.join("02_pv_calc_from_bond_rooftop/", work_dir, "01_footprint_s_area_wb_rooftop_analysis_pv_month_pv.xlsx")
pv_df = pd.read_excel(pv_file_path, sheet_name='Otxarkoaga')
pv_df_hourly = pd.read_csv('02_pv_calc_from_bond_rooftop/data/Otxarkoaga/pv_generation_hourly.csv')

# Reassigning the time range for the pv generation data to be aligned with the aggregated profiles
time_range = pd.date_range(start='2021-01-01 00:00:00', end='2021-12-31 23:00:00', freq='H')
pv_df_hourly['Time'] = time_range
pv_df_hourly.set_index('Time', inplace=True)

#%% TIME ALIGNMENT
# Transform the time format of the index column
# Standardize the time format for all DataFrames
#pv_df_hourly.index = pd.to_datetime(pv_df_hourly.index, utc=True).tz_convert(None) + pd.Timedelta(hours=2)
#pv_df_hourly.index = pv_df_hourly.index.strftime('%d/%m/%Y %H:%M')

df_aggregated_profiles['Time'] = df_aggregated_profiles['Time'].str.replace('2021', '2023')
df_aggregated_profiles['Time'] = pd.to_datetime(df_aggregated_profiles['Time'])#, format='%d/%m/%Y %H:%M')
df_aggregated_profiles.set_index('Time', inplace=True)

df_aggregated_profiles_no_pv['Time'] = df_aggregated_profiles_no_pv['Time'].str.replace('2021', '2023')
df_aggregated_profiles_no_pv['Time'] = pd.to_datetime(df_aggregated_profiles_no_pv['Time'])#, format='%d/%m/%Y %H:%M')
df_aggregated_profiles_no_pv.set_index('Time', inplace=True)

#df_aggregated_profiles.index = df_aggregated_profiles.index.str.replace('2021', '2023')

# %% # Function to aggregate energy consumption profiles by month and season
def aggregate_by_month_and_season(df):
    # Ensure the index is in datetime format
    #df.index = pd.to_datetime(df.index, format='%d/%m/%Y %H:%M')
    
    # Create a month column
    df['month'] = df.index.month
    
    # Create a season column
    def get_season(date):
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'
    
    df['season'] = df.index.map(get_season)
    
    # Group by month and season and calculate the sum
    monthly_aggregation = df.groupby('month').sum()#.drop(columns=['season'])
    seasonal_aggregation = df.groupby('season').sum().drop(columns=['month'])
    
    return monthly_aggregation, seasonal_aggregation

#%% # Function to calculate self-consumption percentage
def calc_dir_self_consumption(df_generation, df_pv_consumption, df_no_pv_consumption):
    # Replace zero values in generation data with NaN to avoid division by zero
    #df_generation = df_generation.replace(0, np.nan)
    #df_pv_consumption = df_generation.replace(0, np.nan)
    #df_no_pv_consumption = df_generation.replace(0, np.nan)
    #df_generation=df_generation.round(4)
    #df_pv_consumption=df_pv_consumption.round(4)
    #df_no_pv_consumption=df_no_pv_consumption.round(4)
    
    # Calculate the hourly self-consumed energy by taking the minimum of generation and consumption
    #df_self_consumed_hourly = np.minimum(df_generation, df_pv_consumption)
    
    # Convert generation data to kWh
    df_generation = (df_generation) #/ 1000
    #print (f"Converted generation data to kWh in file: {df_generation}")
    
    # Calculate the hourly self-consumption percentage
    df_self_consumed_hourly = np.minimum(df_generation, df_pv_consumption)
    # correction
    df_self_cons_pct_hourly = (df_self_consumed_hourly/ df_generation)
    
    # Ensure the index is in datetime format
    #df_self_cons_pct_hourly.index = pd.to_datetime(df_self_cons_pct_hourly.index, format='%d/%m/%Y %H:%M')
    
    # Aggregate the hourly self-consumption percentage to monthly by averaging each month
    #df_self_cons_pct_monthly = df_self_cons_pct_hourly.resample('M').mean().round(4)
    #df_self_cons_pct_monthly = df_self_cons_pct_monthly.T
    df_self_cons_pct_monthly = (df_self_consumed_hourly.resample('M').sum() / df_generation.resample('M').sum()).round(4)
    df_self_cons_pct_monthly = df_self_cons_pct_monthly.T
    df_self_cons_pct_monthly.columns = [f'self_m{month}' for month in df_self_cons_pct_monthly.columns.month]

    # Calculate residual generation and load within the energy community (EC)
    df_residual_generation = df_generation - df_self_consumed_hourly
    #df_residual_generation[df_residual_generation < 0] = 0

    df_residual_consumption = df_pv_consumption - df_self_consumed_hourly
    df_residual_consumption[df_residual_consumption < 0] = 0
    df_self_cons_pct_hourly.to_csv(f'data/04_energy_consumption_profiles/dwell_share_{pv_pct}/self_cons_estimation_files\\04_1_self_cons_pct_hourly.csv')
    df_residual_generation.to_csv(f'data/04_energy_consumption_profiles/dwell_share_{pv_pct}/self_cons_estimation_files/04_2_pv_residual_generation.csv')
    df_self_consumed_hourly.to_csv(f'data\\04_energy_consumption_profiles/dwell_share_{pv_pct}/self_cons_estimation_files/04_1_self_consumed_hourly.csv')
    df_residual_consumption.to_csv(f'data\\04_energy_consumption_profiles/dwell_share_{pv_pct}/self_cons_estimation_files/04_2_pv_residual_load.csv')

    # Calculate the coverage of consumption without PV
    no_pv_ec_cov_consumption = df_residual_generation.where(df_residual_generation <= df_no_pv_consumption, df_no_pv_consumption)
    #no_pv_ec_cov_consumption=np.minimum(df_residual_generation, df_no_pv_consumption)
    no_pv_ec_resid_generation = df_residual_generation - no_pv_ec_cov_consumption
    #no_pv_ec_resid_generation[no_pv_ec_resid_generation < 0] = 0
    no_pv_ec_resid_generation.to_csv(f'data\\04_energy_consumption_profiles\\dwell_share_{pv_pct}\\self_cons_estimation_files\\04_3_no_pv_ec_resid_generation.csv')

    no_pv_resid_consumption = df_no_pv_consumption - no_pv_ec_cov_consumption
    #no_pv_resid_consumption[no_pv_resid_consumption < 0] = 0
    no_pv_resid_consumption.to_csv(f'data\\04_energy_consumption_profiles\\dwell_share_{pv_pct}\\self_cons_estimation_files\\04_4_no_pv_resid_consumption.csv')

    # Calculate the coverage of consumption with PV
    pv_ec_cov_consumption = no_pv_ec_resid_generation.where(no_pv_ec_resid_generation <= df_residual_consumption, df_residual_consumption)
    #pv_ec_cov_consumption[pv_ec_cov_consumption < 0] = 0

    # Calculate the hourly coverage percentage without PV
    df_cov_pct_no_pv_hourly = (no_pv_ec_cov_consumption / df_generation)
    df_cov_pct_no_pv_hourly = df_cov_pct_no_pv_hourly.fillna(0)
    df_cov_pct_no_pv_hourly.index = pd.to_datetime(df_cov_pct_no_pv_hourly.index, format='%d/%m/%Y %H:%M')
    
    # Aggregate the hourly coverage percentage without PV to monthly by averaging each month
    #df_cov_pct_no_pv_monthly = df_cov_pct_no_pv_hourly.resample('M').mean().round(4)
    df_cov_pct_no_pv_monthly = no_pv_ec_cov_consumption.resample('M').sum() / df_generation.resample('M').sum()
    df_cov_pct_no_pv_monthly = df_cov_pct_no_pv_monthly.T
    df_cov_pct_no_pv_monthly.columns = [f'no_pv_cov_m{month}' for month in df_cov_pct_no_pv_monthly.columns.month]

    # Calculate the hourly coverage percentage with PV
    df_cov_pct_pv_hourly = (pv_ec_cov_consumption / df_generation)
    df_cov_pct_pv_hourly = df_cov_pct_pv_hourly.fillna(0)
    df_cov_pct_pv_hourly.index = pd.to_datetime(df_cov_pct_pv_hourly.index, format='%d/%m/%Y %H:%M')
    
    # Aggregate the hourly coverage percentage with PV to monthly by averaging each month
    #df_cov_pct_pv_monthly = df_cov_pct_pv_hourly.resample('M').mean().round(4)
    df_cov_pct_pv_monthly = pv_ec_cov_consumption.resample('M').sum() / df_generation.resample('M').sum()
    df_cov_pct_pv_monthly = df_cov_pct_pv_monthly.T
    df_cov_pct_pv_monthly.columns = [f'with_pv_cov_m{month}' for month in df_cov_pct_pv_monthly.columns.month]

    return df_self_cons_pct_monthly, df_cov_pct_no_pv_monthly, df_cov_pct_pv_monthly


#%%
matching_columns=pv_df_hourly.columns.intersection(df_aggregated_profiles.columns)
matching_index=pv_df_hourly.index.intersection(df_aggregated_profiles.index)
#%%
if not os.path.exists(f"data/04_energy_consumption_profiles/dwell_share_{pv_pct}/self_cons_estimation_files"):
    os.makedirs(f"data/04_energy_consumption_profiles/dwell_share_{pv_pct}/self_cons_estimation_files")

# Subset both DataFrames to only the matching rows and columns
df_self_cons_pct_monthly,df_cov_pct_no_pv_monthly, df_cov_pct_pv_monthly = calc_dir_self_consumption(pv_df_hourly.loc[matching_index, matching_columns],df_aggregated_profiles.loc[matching_index, matching_columns],df_aggregated_profiles_no_pv.loc[matching_index, matching_columns]
)
#%%
# Save the results
df_self_cons_pct_monthly.to_csv(f'data\\\\04_energy_consumption_profiles\\dwell_share_{pv_pct}\\04_self_cons_pct_month_{pv_pct}.csv',index=True, index_label='census_id')
df_cov_pct_no_pv_monthly.to_csv(f'data\\04_energy_consumption_profiles\\dwell_share_{pv_pct}\\04_cov_pct_no_pv_month_{no_pv_pct}.csv',index=True, index_label='census_id')
df_cov_pct_pv_monthly.to_csv(f'data\\04_energy_consumption_profiles\\dwell_share_{pv_pct}\\04_cov_pct_pv_month_{pv_pct}.csv',index=True, index_label='census_id')
#%% # Function to aggregate energy consumption profiles by month and season
def get_aggregated_profiles(df_aggregated_profiles, matching_columns, col_suffix=None):
    # Get the aggregated profiles by month and season by census_id column
    monthly_agg, seasonal_agg = aggregate_by_month_and_season(df_aggregated_profiles)
    
    # Transform columns to int
    monthly_agg = monthly_agg.loc[:, matching_columns]
    #monthly_agg.columns = monthly_agg.columns.astype(int)
    
    # Transpose the dataframe
    monthly_agg = monthly_agg.T
    monthly_agg.columns = monthly_agg.columns.astype(int)
    
    # Add a total column
    monthly_agg.columns = [f'{col_suffix}{month}' for month in monthly_agg.columns]
    monthly_agg['Total, kWh'] = monthly_agg.sum(axis=1).round(4)
    
    return monthly_agg, seasonal_agg
#%%
# Usage
monthly_agg, seasonal_agg = get_aggregated_profiles(df_aggregated_profiles, matching_columns, col_suffix='cons_m')
monthly_agg.to_csv(f'data\\04_energy_consumption_profiles\\dwell_share_{pv_pct}\\04_aggreg_cons_prof_with_pv_by_census_id_monthly_{pv_pct}.csv', index=True, index_label='census_id')
#%%
monthly_agg, seasonal_agg = get_aggregated_profiles(df_aggregated_profiles_no_pv, matching_columns, col_suffix='no_pv_cons_m')
monthly_agg.to_csv(f'data\\04_energy_consumption_profiles\\dwell_share_{pv_pct}\\04_aggreg_cons_prof_no_pv_by_census_id_monthly_{no_pv_pct}.csv', index=True, index_label='census_id')

#%% 

# Get the pv profiles by census_id column
pv_df = pv_df[pv_df['census_id'].notna()]
pv_df['census_id'] = pv_df['census_id'].astype(int)
pv_census_aggreg_df= pv_df.groupby('census_id').sum([1,2,3,4,5,6,7,8,9,10,11,12, 'Total, kWh']).drop(columns=['plain_roof']).round(4)
pv_census_aggreg_df.columns = [f'gen_m{month}' if month in [1,2,3,4,5,6,7,8,9,10,11,12] else month for month in pv_census_aggreg_df.columns]
#%%
# Save the results
pv_census_aggreg_df.to_csv(f'data\\04_energy_consumption_profiles\\dwell_share_{pv_pct}\\04_aggregated_pv_gen_by_census_id_monthly.csv', index=True)

#%%
# Get intersection of both index and columns
matching_index = pv_census_aggreg_df.index.intersection(monthly_agg.index)
matching_columns = pv_census_aggreg_df.columns.intersection(monthly_agg.columns)

# Subset both DataFrames to only the matching rows and columns
generation_matching = pv_census_aggreg_df.loc[matching_index, matching_columns]
consumption_matching = monthly_agg.loc[matching_index, matching_columns]

# Perform element-wise division
df_self_consumption = (generation_matching / consumption_matching).round(4)

df_self_consumption
#%%
# Save the results
df_self_consumption.to_csv(f'data\\04_energy_consumption_profiles\\dwell_share_{pv_pct}\\04_net_balance_gen_to_cons_by_census_id_monthly.csv', index=True)


#%% SYNTEHTIC DATA GENERATION
# Generate a date range with hourly frequency for one year
date_range = pd.date_range(start="2023-01-01 00:00", end="2023-12-31 23:00", freq="H")

# Create a sample DataFrame for generation and consumption with random data
# Assume 3 unique ID units for demonstration
id_units = ["ID1", "ID2", "ID3"]

# Random generation values between 0 and 200kW for each hour and each ID
generation_data = np.random.uniform(0, 200000000, (len(date_range), len(id_units)))
df_generation = pd.DataFrame(generation_data, index=date_range, columns=id_units)
# Random consumption values between 0 and 120kW for each hour and each ID (to allow for cases where consumption > generation)
consumption_data = np.random.uniform(0, 120000, (len(date_range), len(id_units)))
df_pv_consumption = pd.DataFrame(consumption_data, index=date_range, columns=id_units)
df_no_pv_consumption = pd.DataFrame(consumption_data, index=date_range, columns=id_units)
# Display the generated data for testing
df_generation.head(), df_pv_consumption.head()
## Run the function
df_self_cons_pct_monthly,df_cov_pct_no_pv_monthly, df_cov_pct_pv_monthly = calc_dir_self_consumption(df_generation, df_pv_consumption,df_no_pv_consumption)
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Data for Scenario Matrix
scenarios = ['1.1', '1.2', '2.1', '2.2', '3.1', '3.2', '0.1', '0.2']
participation_rates = ['100%', '75%', '50%', '25%']
benefit_levels = np.random.rand(4, 8) * 100  # Randomized benefit levels for demonstration

# Create DataFrame for Heatmap
heatmap_df = pd.DataFrame(benefit_levels, index=participation_rates, columns=scenarios)

# Plot Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_df, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Benefit Levels (%)'})
plt.title('Scenario Matrix Heatmap: Benefit Levels vs Participation Rates')
plt.xlabel('Scenarios')
plt.ylabel('Participation Rates')
plt.show()

# Sample Data for Stacked Bar Chart
segments = ['Low Income', 'Middle Income', 'High Income']
economic_benefits = {
    '1.1': [20, 40, 40],
    '1.2': [25, 35, 40],
    '2.1': [30, 30, 40],
    '2.2': [35, 25, 40],
    '3.1': [40, 20, 40],
    '3.2': [45, 15, 40],
    '0.1': [50, 10, 40],
    '0.2': [55, 5, 40]
}

# Convert to DataFrame
stacked_df = pd.DataFrame(economic_benefits, index=segments)

# Plot Stacked Bar Chart
stacked_df.T.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='viridis')
plt.title('Distribution of Economic Benefits Across Socio-economic Segments')
plt.xlabel('Scenarios')
plt.ylabel('Economic Benefits (%)')
plt.legend(title='Socio-economic Segments')
plt.show()


# %%
# Sample Data for Line and Area Charts
participation_rates = [100, 75, 50, 25]
economic_savings = [80, 60, 40, 20]  # Economic savings as participation decreases
energy_vulnerability = [20, 30, 40, 50]  # Vulnerability increases as participation decreases

# Line Chart: Trend Analysis of Economic Savings
plt.figure(figsize=(10, 6))
plt.plot(participation_rates, economic_savings, marker='o', linestyle='-', color='blue', label='Economic Savings')
plt.title('Trend Analysis: Economic Savings vs Participation Rates')
plt.xlabel('Participation Rates (%)')
plt.ylabel('Economic Savings (%)')
plt.grid(True)
plt.legend()
plt.show()

# Area Chart: Trend Analysis of Energy Vulnerability
plt.figure(figsize=(10, 6))
plt.fill_between(participation_rates, energy_vulnerability, color='red', alpha=0.5, label='Energy Vulnerability')
plt.plot(participation_rates, energy_vulnerability, marker='o', color='red')
plt.title('Trend Analysis: Energy Vulnerability vs Participation Rates')
plt.xlabel('Participation Rates (%)')
plt.ylabel('Energy Vulnerability Index')
plt.grid(True)
plt.legend()
plt.show()

# Sample Data for Sensitivity Heatmaps
parameters = ['Electricity Price', 'PV Efficiency', 'Self-Consumption Rate']
scenarios = ['1.1', '1.2', '2.1', '2.2', '3.1', '3.2', '0.1', '0.2']
sensitivity_data = np.random.uniform(-20, 20, (len(parameters), len(scenarios)))

# Create DataFrame for Sensitivity Heatmap
sensitivity_df = pd.DataFrame(sensitivity_data, index=parameters, columns=scenarios)

# Plot Sensitivity Heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(sensitivity_df, annot=True, cmap='coolwarm', cbar_kws={'label': 'Impact on Economic Benefits (%)'})
plt.title('Sensitivity Heatmap: Impact of Parameter Changes on Economic Benefits')
plt.xlabel('Scenarios')
plt.ylabel('Parameters')
plt.show()


# %%
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt

# Sample Data for Choropleth Map
# Create a mock GeoDataFrame
np.random.seed(42)
num_zones = 10
zones = gpd.read_file("data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
zones = zones[zones['continent'] == 'Europe'].head(num_zones)  # Limit to 10 regions

# Mock Energy Vulnerability and Benefit Distribution Data
zones['Energy Vulnerability'] = np.random.randint(10, 100, num_zones)
zones['Benefit Distribution'] = np.random.randint(20, 80, num_zones)

# Plot Choropleth Map
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

zones.plot(column='Energy Vulnerability', ax=ax[0], legend=True, cmap='OrRd')
ax[0].set_title('Choropleth Map: Energy Vulnerability')

zones.plot(column='Benefit Distribution', ax=ax[1], legend=True, cmap='YlGnBu')
ax[1].set_title('Choropleth Map: Benefit Distribution')

plt.show()

# Sample Data for Hexbin Map
# Create random points for self-consumption and participation levels
x = np.random.randn(1000) * 10
y = np.random.randn(1000) * 15
values = np.random.rand(1000) * 100  # Random self-consumption values

# Plot Hexbin Map
plt.figure(figsize=(12, 8))
hb = plt.hexbin(x, y, C=values, gridsize=30, cmap='viridis', reduce_C_function=np.mean)
cb = plt.colorbar(hb, label='Self-Consumption (%)')
plt.title('Hexbin Map: Self-Consumption Distribution')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# %%
