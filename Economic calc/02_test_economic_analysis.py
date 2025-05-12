#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from util_func import PVAnalysis
from util_func import UtilFunctions
from matplotlib.font_manager import FontProperties

#%%# Sample DataFrame structure based on your description
# Assuming 'df' is your DataFrame
data = {
    'build_id': [1, 2, 3, 4, 5],
    'census_id': [101, 101, 102, 102, 103],
    'S1_Facd_I_C': [10000, 15000, 12000, 13000, 11000],
    'S1_Facd_M_C': [500, 700, 600, 650, 550],
    'S1_Wind_I_C': [8000, 9000, 8500, 8700, 8200],
    'S1_Wind_M_C': [400, 450, 420, 430, 410],
    'S1_Roof_I_C': [7000, 7500, 7200, 7300, 7100],
    'S1_Roof_M_C': [350, 375, 360, 365, 355],
    'PV_I_C': [15000, 16000, 15500, 15800, 15200],
    'PV_M_C': [600, 620, 610, 615, 605],
    'HeatDem_kWh/m2': [200, 220, 210, 215, 205],
    'S1_HeatDem_kWh/m2': [150, 160, 155, 158, 152],
    'PV_Total, kWh': [5000, 5200, 5100, 5150, 5050],
    'PV_self_cons,kWh': [4000, 4100, 4050, 4080, 4020],
    'PV_to_grid,kWh': [1000, 1100, 1050, 1070, 1030],
    'PV_self_cons_Euro': [400, 410, 405, 408, 402],
    'PV_to_grid_Euro': [100, 110, 105, 107, 103]

}
df = pd.DataFrame(data)
#%%
root=r"H:\My Drive\01_PHD_CODE\chapter_5\spain\bilbao_otxarkoaga"
#root=r"D:\o.husiev@opendeusto.es\01_PHD_CODE\chapter_5\spain\bilbao_otxarkoaga"

#df = pd.read_excel(root + r"\data\05_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV_economic.xlsx")
#df = pd.read_excel(root + r"\data\05_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV_economic_otxarkpaga_orginal.xlsx")
df = pd.read_excel(root + r"\data\05_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV_economic_otxarkpaga_orginal_price_incr+pv_degrad.xlsx")

util_pv = PVAnalysis()
util_func = UtilFunctions()
combined_df = pd.DataFrame()
#%%
# select only residential buildings and buildings with PV
df = df[(df["Codigo_Uso"] == "V") & (df["Total, kWh"] > 0)]

#%% Simple calculations for NPV and EUAC with one intervention combination
# Assuming 'df' is your DataFrame

# Define parameters
r = 0.05  # 5% discount rate
n = 30    # 20-year period

intervention_combinations = {
    1: ["facade", "roof"], # low_intervention_min or high_intervention_min
    2: ["facade", "windows", "roof"],#low_intervention_max or high_intervention_max
}
df['Initial_Investment'] = df[['S1_Facd_I_C', 'S1_Roof_I_C', 'PV_I_C']].sum(axis=1)
df['Annual_Maintenance'] = df[['S1_Facd_M_C', 'S1_Roof_M_C', 'PV_M_C']].sum(axis=1)
df['Annual_Savings'] = df['PV_self_cons_Euro'] + df['PV_to_grid_Euro']+ df['Comb1S1_HDem_Savings_Euro']
df['Net_Annual_Cash_Flow'] = df['Annual_Savings'] - df['Annual_Maintenance']

# NPV calculation
def calculate_npv(row, r, n):
    return -row['Initial_Investment'] + row['Net_Annual_Cash_Flow'] * (1 - (1 + r) ** -n) / r

df['NPV'] = df.apply(lambda row: calculate_npv(row, r, n), axis=1)

# EUAC calculation
def calculate_euac(row, r, n):
    crf = r * (1 + r) ** n / ((1 + r) ** n - 1)
    return row['Initial_Investment'] * crf + row['Annual_Maintenance'] - row['Annual_Savings']

df['EUAC'] = df.apply(lambda row: calculate_euac(row, r, n), axis=1)

# Aggregate by census_id if needed
agg_df = df.groupby('census_id').agg({
    'NPV': 'sum',
    'EUAC': 'sum',
    'n_dwellEstim': 'sum',
    'n_dwellOriginal': 'sum'
}).reset_index()
agg_df["NPV_per_dwelling"] = agg_df["NPV"]/agg_df[dwelling_col]
agg_df["EUAC_per_dwelling"] = agg_df["EUAC"]/agg_df[dwelling_col]
print(agg_df)
#%% 
# Define parameters
r = 0.05  # 5% discount rate
n = 20    # 20-year period for PV system
n_int = 30 # 30-year period for interventions
energy_price_growth_rate = 0#0.02  # 2% annual increase in energy prices
heating_energy_price_euro_per_kWh = 0.243 #0.1 #0.307 # assumed cost as for electricity
pv_degradation_rate = 0#0.01  # 1% per year

"""# NPV calculation for Envelope
def calculate_npv(row, r, n, initial_investment_col, net_annual_cf_col):
    return -row[initial_investment_col] + row[net_annual_cf_col] * (1 - (1 + r) ** -n) / r

df['Envelope_NPV'] = df.apply(lambda row: calculate_npv(row, r, n_int, 'Envelope_Initial_I', 'Envelope_Net_Annual_CF'), axis=1)
df['PV_NPV'] = df.apply(lambda row: calculate_npv(row, r, n, 'PV_I_C', 'PV_Net_Annual_CF'), axis=1)
"""
# EUAC calculation for Envelope
def calculate_euac(row, r, n, initial_investment_col, annual_maintenance_col, annual_savings_col):
    crf = r * (1 + r) ** n / ((1 + r) ** n - 1)  # Capital Recovery Factor
    if initial_investment_col is None and annual_savings_col is None: # for Reference case calculation
        return row[annual_maintenance_col] * crf
    else:
        return row[initial_investment_col] * crf + row[annual_maintenance_col] - row[annual_savings_col]

#%%
#HEATING DEMAND REDUCTION CALCULATIONS
intervention_combinations = {
    1: ["facade", "roof"], # low_intervention_min or high_intervention_min
    2: ["facade", "windows", "roof"],#low_intervention_max or high_intervention_max
}
CAL_REFERENCE_CASE = False
SCENARIO = "S2"
COMBINATION_ID = 2 #select the components of the envelope that are considered for the heating demand reduction
#Dwelling Column name
dwelling_col = "n_dwellOriginal" # 'n_dwellEstim' or 'n_dwellOriginal'
#%% Calculate EUAC for Envelope and PV
if CAL_REFERENCE_CASE is False:
    # Calculate initial investment and annual maintenance for Envelope and PV
    if "windows" in intervention_combinations[COMBINATION_ID]:
        df['Envelope_Initial_I'] = df[[f'{SCENARIO}_Facd_I_C',f'{SCENARIO}_Wind_I_C',  f'{SCENARIO}_Roof_I_C']].sum(axis=1)
        df['Envelope_Annual_M'] = df[[f'{SCENARIO}_Facd_M_C', f'{SCENARIO}_Wind_M_C', f'{SCENARIO}_Roof_M_C']].sum(axis=1)
    if "windows" not in intervention_combinations[COMBINATION_ID]:
        df['Envelope_Initial_I'] = df[[f'{SCENARIO}_Facd_I_C',  f'{SCENARIO}_Roof_I_C' ]].sum(axis=1)
        df['Envelope_Annual_M'] = df[[f'{SCENARIO}_Facd_M_C', f'{SCENARIO}_Roof_M_C']].sum(axis=1)

    # Calculate annual savings for Envelope and PV separately
    df['Envelope_Annual_Savings'] = df[f'Comb{COMBINATION_ID}{SCENARIO}_HDem_Savings_Euro']
    df['heating_energy_price_euro_per_kWh'] = heating_energy_price_euro_per_kWh 
    df['heating_energy_price_euro_per_kWh'] = util_pv.average_price_increase(df['heating_energy_price_euro_per_kWh'],energy_price_growth_rate, lifespan=30)
    df['Envelope_Energy_Euro'] = df[f'Comb{COMBINATION_ID}{SCENARIO}_HDem_kWh/m2']* df['grossFloorArea']*df['heating_energy_price_euro_per_kWh']

    df['PV_Annual_Savings'] = df['PV_self_cons_Euro'] + df['PV_to_grid_Euro']
    
    #Calculate Annual Cost (For EUAC)
    df["Envelope_Annual_Cost"]= df['Envelope_Annual_M'] + df['Envelope_Energy_Euro']
    df["PV_Annual_Cost"] = df['PV_M_C'] - df['PV_Annual_Savings']

    # Calculate net annual cash flow for Envelope and PV separately (for NPV)
    """df['Envelope_Net_Annual_CF'] =  df['Envelope_Annual_Savings'] - df['Envelope_Annual_M']
    df['PV_Net_Annual_CF'] =  df['PV_Annual_Savings'] - df['PV_M_C'] """

    # CALCULATE Present Value for Envelope and PV separately
    df['Envelope_PV'] = util_pv.calculate_present_value(df, 'Envelope_Initial_I',  'Envelope_Annual_Cost', r, n_int)
    df['PV_PV'] = util_pv.calculate_present_value(df, 'PV_I_C',  'PV_Annual_Cost', r, n)

    # CALCUALTE EUAC FOR SCENARIO
    df['Envelope_EUAC'] = util_pv.calculate_euac(df, r, n_int, 'Envelope_PV')
    df['PV_EUAC'] = util_pv.calculate_euac(df, r, n, 'PV_PV')
    #df[f'Comb{COMBINATION_ID}{SCENARIO}_Initial_Investment'] = df['Envelope_Initial_I'] + df['PV_I_C']
    #df['Envelope_EUAC'] = df.apply(lambda row: calculate_euac(row, r, n_int, 'Envelope_Initial_I', 'Envelope_Annual_M', 'Envelope_Annual_Savings'), axis=1)
    #df['PV_EUAC'] = df.apply(lambda row: calculate_euac(row, r, n, 'PV_I_C', 'PV_M_C', 'PV_Annual_Savings'), axis=1)
    df['EUAC'] = df['Envelope_EUAC'] + df['PV_EUAC']

    df['Envelope_Energy_kWh'] = df[f'Comb{COMBINATION_ID}{SCENARIO}_HDem_kWh/m2']* df['grossFloorArea']
else:
    df['Envelope_Energy_kWh'] = df['HDemProj']* df['grossFloorArea']
    df['heating_energy_price_euro_per_kWh'] = heating_energy_price_euro_per_kWh 
    df['heating_energy_price_euro_per_kWh'] = util_pv.average_price_increase(df['heating_energy_price_euro_per_kWh'],energy_price_growth_rate, lifespan=30)
    
    df['Envelope_Energy_Euro'] = df['Envelope_Energy_kWh']*df['heating_energy_price_euro_per_kWh']
    df['PV_Annual_Savings'] = df['PV_self_cons_Euro'] + df['PV_to_grid_Euro']

    # Calculate net annual cash flow for Envelope and PV separately (for NPV)
    """df['Envelope_Net_Annual_CF'] = -df['Envelope_Energy_Euro']
    df['PV_Net_Annual_CF'] =  df['PV_Annual_Savings'] -df['PV_M_C']"""

    # Calculate Annual Cost (For EUAC)
    df["Envelope_Annual_Cost"]= df['Envelope_Energy_Euro']
    df["PV_Annual_Cost"] = df['PV_M_C'] - df['PV_Annual_Savings']

    # CALCULATE Present Value for Envelope and PV separately
    df['Envelope_Initial_I'] = 0
    df['Envelope_PV'] = util_pv.calculate_present_value(df, 'Envelope_Initial_I',  'Envelope_Annual_Cost', r, n_int)
    df['PV_PV'] = util_pv.calculate_present_value(df, 'PV_I_C',  'PV_Annual_Cost', r, n)

    # CALCUALTE EUAC FOR SCENARIO
    df['Envelope_EUAC'] = util_pv.calculate_euac(df, r, n_int, 'Envelope_PV')
    df['PV_EUAC'] = util_pv.calculate_euac(df, r, n, 'PV_PV')

    #df['PV_EUAC'] = df.apply(lambda row: calculate_euac(row, r, n, 'PV_I_C', 'PV_M_C', 'PV_Annual_Savings'), axis=1)
    #df['Envelope_EUAC'] = df.apply(lambda row: calculate_euac(row, r, n_int, None, 'Envelope_Energy_Euro', None), axis=1)
    df['EUAC'] = df['Envelope_EUAC'] + df['PV_EUAC']

#REFERENCE CASE
"""df['ref_case_kWh'] = df['HDemProj']* df['grossFloorArea']
df['ref_case_Euro'] = df['ref_case_kWh']*heating_energy_price_euro_per_kWh
df['ref_case_EUAC'] = df.apply(lambda row: calculate_euac(row, r, n_int, '', 'ref_case_Euro', ''), axis=1)"""
#agg_df['PV_kWh_per_dwelling'] = df['Total, kWh']/agg_df["n_dwellEstim"]
#agg_df["Energy_kWh_per_dwelling"] = df['Envelope_kWh_per_dwelling'] + agg_df['PV_kWh_per_dwelling']

# Aggregate by census_id if needed
if CAL_REFERENCE_CASE is False:
    agg_df = df.groupby('census_id').agg({
        #'NPV': 'sum',
        'Envelope_EUAC': 'sum',
        'PV_EUAC': 'sum',
        'Envelope_Energy_kWh': 'sum',
        "Envelope_Annual_Cost" : 'sum',
        "PV_Annual_Cost": 'sum',
        'Total, kWh': 'sum',
        'EUAC': 'sum',
        'n_dwellEstim': 'sum',
        "n_dwellOriginal": 'sum',
        "Envelope_Initial_I": 'sum',
        "PV_I_C": 'sum',
        "Envelope_Annual_M": 'sum',
        "PV_M_C": 'sum',
        "Envelope_Annual_Savings": 'sum',
        "PV_Annual_Savings": 'sum',
        'grossFloorArea': 'sum'
        #"ref_case_kWh": 'sum',
        #'ref_case_EUAC': 'sum'
    }).reset_index()

if CAL_REFERENCE_CASE:
    agg_df = df.groupby('census_id').agg({
        #'NPV': 'sum',
        'Envelope_EUAC': 'sum',
        'PV_EUAC': 'sum',
        'Envelope_Energy_kWh': 'sum',
        "Envelope_Annual_Cost" : 'sum',
        "PV_Annual_Cost": 'sum',
        'Total, kWh': 'sum',
        'EUAC': 'sum',
        'n_dwellEstim': 'sum',
        "n_dwellOriginal": 'sum',
        'grossFloorArea': 'sum'
        #"Envelope_Initial_I": 0,
        #"PV_I_C": 'sum',
        #"Envelope_Annual_M": 0,
        #"PV_M_C": 0,
        #"Envelope_Annual_Savings": 0,
        #"PV_Annual_Savings": 0
     }).reset_index()

#Estimation based on the original data for Otxarkoaga
"""data = {
    "census_id": [4802003003, 4802003005, 4802003006, 4802003007, 4802003009, 4802003010, 4802003011, 4802003015],
    "Num dwellings": [501, 830, 404, 464, 536, 350, 664, 464],
    "Total Population": [1037, 1624, 846, 914, 1220, 737, 1395, 1088],
    "Avg dwelling size": [79, 59, 59, 59, 73, 59, 57, 80],
    "Buildings": [29, 58, 25, 28, 36, 22, 5019, None]  # Assuming missing value for last row
}

df_otxarkoaga = pd.DataFrame(data)
#agg_df['n_dwellInput']= pd.merge(agg_df, df, on='census_id', how='left')['Num dwellings']
agg_df['n_dwellEstim']= pd.merge(agg_df, df_otxarkoaga, on='census_id', how='left')['Num dwellings']"""
# EUAC per dwelling
#agg_df["NPV_per_dwelling"] = agg_df["NPV"]/agg_df["n_dwellEstim"]
agg_df["EUAC_per_dwelling"] = agg_df["EUAC"]/agg_df[dwelling_col]
agg_df['Envelope_EUAC_per_dwelling'] = agg_df['Envelope_EUAC']/agg_df[dwelling_col]
agg_df["Envelope_Energy_kWh_per_dwelling"] = agg_df["Envelope_Energy_kWh"]/agg_df[dwelling_col]
agg_df['PV_kWh_per_dwelling'] = agg_df['Total, kWh']/agg_df[dwelling_col]
agg_df['PV_EUAC_per_dwelling'] = agg_df['PV_EUAC']/agg_df[dwelling_col]
nrpe_factor = 2.007 # electricity factor for NRPE 
co2_elect_factor = 0.357 # electricity factor for CO2 emissions

# NRPE and CO2 emissions
agg_df['NRPE_Envelope_kWh_per_dwelling'] = agg_df['Envelope_Energy_kWh_per_dwelling'] * nrpe_factor
agg_df['NRPE_kWh_per_dwelling'] = (agg_df['Envelope_Energy_kWh_per_dwelling'] - agg_df['PV_kWh_per_dwelling']) * nrpe_factor

agg_df['CO2_Envelope_per_dwelling'] = agg_df['Envelope_Energy_kWh_per_dwelling'] * co2_elect_factor
agg_df['CO2_per_dwelling'] = (agg_df['Envelope_Energy_kWh_per_dwelling'] - agg_df['PV_kWh_per_dwelling']) * co2_elect_factor

# EUAC per m2
agg_df['EUAC_per_m2'] = agg_df['EUAC']/agg_df["grossFloorArea"]
agg_df['Envelope_EUAC_per_m2'] = agg_df['Envelope_EUAC']/agg_df["grossFloorArea"]
if CAL_REFERENCE_CASE is False:
    agg_df['PV_EUAC_per_m2'] = agg_df['PV_EUAC']/agg_df["grossFloorArea"]

# NPRE and CO2 emissions per m2
agg_df['NRPE_Envelope_kWh_per_m2'] = agg_df['Envelope_Energy_kWh']/agg_df["grossFloorArea"] * nrpe_factor
agg_df['NRPE_kWh_per_m2'] = (agg_df['Envelope_Energy_kWh'] - agg_df['Total, kWh'])/agg_df["grossFloorArea"] * nrpe_factor

agg_df['CO2_Envelope_per_m2'] = agg_df['Envelope_Energy_kWh']/agg_df["grossFloorArea"] * co2_elect_factor
agg_df['CO2_per_m2'] = (agg_df['Envelope_Energy_kWh'] - agg_df['Total, kWh'])/agg_df["grossFloorArea"] * co2_elect_factor



"""agg_df['ref_case_EUAC_per_dwelling'] = agg_df['ref_case_EUAC']/agg_df["n_dwellEstim"]
agg_df['NRPE_ref_case_kWh_per_dwelling'] = agg_df['ref_case_kWh']/agg_df["n_dwellEstim"] * nrpe_factor"""

#Sum column that starts with 'Comb' and ends with 'Initial_Investment'
#agg_df["Total_Initial_Investment"] = agg_df.filter(regex=f'^Comb{COMBINATION_ID}{SCENARIO}_Initial_Investment$').sum(axis=1)

#print(agg_df)
def append_long_dataframe(df, COMBINATION_ID, SCENARIO):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    # Add the extra columns
    df_copy['Filter'] = f'Comb{COMBINATION_ID}{SCENARIO}'
    #df_copy['SCENARIO'] = scenario
    
    return df_copy

# Example usage
if CAL_REFERENCE_CASE:
    COMBINATION_ID = 0
    SCENARIO = "ref_case"
temp_df = append_long_dataframe(agg_df, COMBINATION_ID, SCENARIO)
combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
if CAL_REFERENCE_CASE:
    print(combined_df)
#%% SAVE TO EXCEL
#util_func.add_sheet_to_excel(combined_df, root + r"\data\06_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV_economic+EUAC_ORIGIN_ADD_SHEET.xlsx", "Sheet0")
#combined_df.to_excel(root + r"\data\06_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV_economic+EUAC_ORIGIN.xlsx", index=False)
#%% SEABORN Scatterplot with varying point sizes and hues

# Set theme for the plot
sns.set_theme(style="white")
x_label="NRPE_Envelope_kWh_per_dwelling" # "NRPE_kWh_per_m2" "NRPE_Envelope_kWh_per_m2" "NRPE_kWh_per_dwelling", "NRPE_Envelope_kWh_per_dwelling"
y_label= "Envelope_EUAC_per_dwelling"#  "EUAC_per_m2" "Envelope_EUAC_per_m2""Envelope_EUAC_per_dwelling", "Envelope_EUAC_per_dwelling"
# Create scatterplot using seaborn's relplot
plot = sns.relplot(
    x=x_label,
    y= y_label,#"Envelope_EUAC_per_dwelling",
    hue='Filter',#'census_id', "Filter",
    size=dwelling_col,
    sizes=(40, 400),
    alpha=0.7,
    palette="muted",
    height=6,
    data=combined_df
)
plot.set(xlim=(-2000, 10000), ylim=(0, 1500))  # Adjust these values as needed

#plot.set(xlim=(-25, 180), ylim=(0, 30))  # Adjust these values as needed

# Add red dotted line parallel to y-axis at x=0
plt.axvline(x=0, color='red', linestyle='--')
#plt.axhline(y=0, color='red', linestyle='--')

# Add labels and title
plot.set_axis_labels(x_label, y_label)
#plot.fig.suptitle("Scatterplot: EUAC vs NPV per Dwelling with FilterCategory as Hue", y=1.02)
plt.show()
#%%
# Set theme for the plot
sns.set_theme(style="white")

# Create scatterplot using seaborn's relplot
fig, ax = plt.subplots(figsize=(10, 6))
plot = sns.scatterplot(
    x="NRPE_kWh_per_dwelling",#NRPE_kWh_per_dwelling",#"NRPE_Envelope_kWh_per_dwelling"
    y= "EUAC_per_dwelling",#"EUAC_per_dwelling",#"Envelope_EUAC_per_dwelling",#"EUAC_per_dwelling",#
    hue="Filter",
    size=dwelling_col,
    sizes=(40, 400),
    alpha=0.7,
    palette="muted",
    data=combined_df,
    ax=ax,
    legend=True # Hide legend
)

# Set fixed x and y axis range
#ax.set_xlim(-300, 7000)  # Adjust these values as needed
#ax.set_ylim(-300, 3000)   # Adjust these values as needed
#y labels with 100 increment
#ax.set_yticks(np.arange(-300, 700, 100))
# Add grid
ax.grid(True)
#%% TEsting matplotlib Scatterplot with continuous color scale
# Sample data creation to simulate the dataframe

import matplotlib.pyplot as plt
import pandas as pd


data = {
    "census_id": [1, 2, 3, 4, 5],
    "NPV": [1000, 1500, 800, 1200, 950],
    "EUAC": [200, 250, 180, 220, 190],
    "n_dwellEstim": [10, 15, 8, 12, 9],
    "NPV_per_dwelling": [100, 100, 100, 100, 105],
    "EUAC_per_dwelling": [20, 17, 22.5, 18.33, 21.11],
    "Filter": ["A", "B", "A", "C", "B"]
}

df = pd.DataFrame(data)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df["EUAC_per_dwelling"],
    df["NPV_per_dwelling"],
    c=df["Filter"].astype('category').cat.codes,
    s=df["EUAC"] * 0.1,  # Scale sizes for better visualization
    cmap="viridis",
    alpha=0.7,
    edgecolors="w",
    linewidth=0.5
)

# Add labels and title
plt.colorbar(scatter, label="FilterCategory (as code)")
plt.xlabel("EUAC per Dwelling")
plt.ylabel("NPV per Dwelling")
plt.title("Scatterplot: EUAC vs NPV per Dwelling with FilterCategory as Hue")
plt.show()
#%% TESTING FUNCTIONS TO calculate PV savings with degradation and price increase
def calculate_average_savings(initial_production, degradation_rate, lifespan, initial_price, price_increase):
    """
    Calculate average annual savings considering PV degradation and energy price increase.

    Parameters:
    - initial_production (float): Initial annual production (e.g., in kWh).
    - degradation_rate (float): Annual degradation rate (e.g., 0.005 for 0.5%).
    - lifespan (int): Expected lifespan of the PV system (in years).
    - initial_price (float): Initial energy price (e.g., Euro/kWh).
    - price_increase (float): Annual energy price increase rate (e.g., 0.02 for 2%).

    Returns:
    - float: Average annual savings over the lifespan.
    """
    total_savings = sum(
        initial_production * (1 - degradation_rate) ** (t - 1) * (initial_price * (1 + price_increase) ** (t - 1))
        for t in range(1, lifespan + 1)
    )
    return total_savings / lifespan

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

# Example usage
data_pv = {
    'build_id': [1, 2, 3],
    'census_id': [101, 102, 103],
    'r_area': [50, 60, 70],  # Roof area
    'n_panel': [20, 24, 28],  # Number of panels
    's_roof': [45, 55, 65],  # Roof suitability factor
    1: [300.534, 310, 320],  # kWh generated in January
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

# Calculate PV metrics
df_pv_metrics = calculate_pv_metrics_adv(df_pv, df_prices, monthly_self_cons_percentages)
df_pv_metrics_degradation = calculate_pv_metrics_adv(df_pv, df_prices, monthly_self_cons_percentages, incl_degradation=0.01)
df_pv_metrics_degradation_price = calculate_pv_metrics_adv(df_pv, df_prices, monthly_self_cons_percentages, incl_degradation=0.01, incl_price_increase=0.01)


#%% THE SAME FUNCTION FUNCTIONS TO calculate PV savings with degradation and price increase BUT APPLIES VECTORISED CALCULATIONS
def average_price_increase(column, price_increase, lifespan=20):
    """
    Calculate the average annual price over the lifespan for a DataFrame column.

    Parameters:
    - column (pd.Series): Column of initial prices in Euro.
    - price_increase (float): Annual price increase rate (e.g., 0.02 for 2%).
    - lifespan (int): Lifespan of the system in years.

    Returns:
    - pd.Series: Series with average price during lifespan considering price increase.
    """ 
    #total cumulative price over a defined lifespan, assuming the price increases by a constant percentage (price_increase) each time step
    total_price_increase = column.apply(
        lambda initial_price: sum(
            initial_price * (1 + price_increase) ** (t - 1) for t in range(1, lifespan + 1)
        )
    )
    return total_price_increase / lifespan

def calculate_pv_anual_average(column, degradation_rate=0.01, lifespan=20):
    """
    Calculate the average annual PV production for a DataFrame column considering degradation.

    Parameters:
    - column (pd.Series): Column of initial annual production in kWh.
    - degradation_rate (float): Annual PV degradation rate (e.g., 0.01 for 1%).
    - lifespan (int): Lifespan of the system in years.

    Returns:
    - pd.Series: Series with average annual PV production considering degradation.
    """
    #total cumulative production over a defined lifespan, assuming the production decreases by a constant percentage (degradation_rate) each time step
    total_production = column.apply(
        lambda initial_production: sum(
            initial_production * (1 - degradation_rate) ** (t - 1) for t in range(1, lifespan + 1)
        )
    )
    return total_production / lifespan

def calculate_pv_metrics_adv_df(df_pv, df_prices, monthly_self_cons_percentages, incl_degradation=0, incl_price_increase=0):
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
    df_results = pd.DataFrame()
    df_temp = df_pv.copy()
    
    for month in monthly_columns:
        df_temp[month] = df_temp[month]
        if incl_degradation != 0:
            # Calculate monthly average generation with degradation for the lifespan
            df_temp[month] = calculate_pv_anual_average(df_pv[month], degradation_rate=incl_degradation, lifespan=20)
        
        buy_price = df_prices.loc[month - 1, 'Electricity Buy']  # Match price by month
        sell_price = df_prices.loc[month - 1, 'Electricity Sell']
        if incl_price_increase != 0:
            # Calculate average monthly price considering price increase for the lifespan
            buy_price = average_price_increase(pd.Series([buy_price]), price_increase=incl_price_increase).iloc[0]
            sell_price = average_price_increase(pd.Series([sell_price]), price_increase=incl_price_increase).iloc[0]
        
        # Get the self-consumption percentage for the current month
        month_name = df_prices.loc[month - 1, 'Month']
        self_cons_percent = monthly_self_cons_percentages[month_name]
        
        # Self-consumed and exported energy
        df_temp[f"self_consumed_energy_kWh_{month}"] = df_temp[month] * self_cons_percent
        df_temp[f"exported_energy_kWh_{month}"] = df_temp[month] * (1 - self_cons_percent)
        df_temp[f'pv_self_cons_euro_{month}'] = df_temp[f"self_consumed_energy_kWh_{month}"] * buy_price
        df_temp[f'pv_to_grid_euro_{month}'] = df_temp[f"exported_energy_kWh_{month}"] * sell_price
        
    # Add to total kWh
    df_temp['PV_self_cons_kWh'] = df_temp[[f"self_consumed_energy_kWh_{month}" for month in monthly_columns]].sum(axis=1).round(3)
    df_temp['PV_to_grid_kWh'] = df_temp[[f"exported_energy_kWh_{month}" for month in monthly_columns]].sum(axis=1).round(3)
    
    # Monetary values
    df_temp['PV_self_cons_Euro'] = df_temp[[f"pv_self_cons_euro_{month}" for month in monthly_columns]].sum(axis=1).round(2)
    df_temp['PV_to_grid_Euro'] = df_temp[[f"pv_to_grid_euro_{month}" for month in monthly_columns]].sum(axis=1).round(2)
    
    df_results = df_temp[['build_id', 'census_id', 'PV_self_cons_kWh', 'PV_to_grid_kWh', 'PV_self_cons_Euro', 'PV_to_grid_Euro']]
    # Delete temporary dataframes to free up memory

    return pd.DataFrame(df_results)
#%% TESTING FUNCTION TO calculate EUAC with incorporating the degradation of PV systems, annual energy price increases, and different lifespans for PV and envelopes, you can create a function that calculates the EUAC for each component and then combines the results

def calculate_pv_metrics_adv_df(self,df_pv, df_prices, monthly_self_cons_percentages, incl_degradation=0, incl_price_increase=0):
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
    df_results = pd.DataFrame()
    df_temp = df_pv.copy()
    
    for month in monthly_columns:
        df_temp[month] = df_temp[month]
        if incl_degradation != 0:
            # Calculate monthly average generation with degradation for the lifespan
            df_temp[month] = self.calculate_pv_anual_average(df_pv[month], degradation_rate=incl_degradation, lifespan=20)
        
        buy_price = df_prices.loc[month - 1, 'Electricity Buy']  # Match price by month
        sell_price = df_prices.loc[month - 1, 'Electricity Sell']
        if incl_price_increase != 0:
            # Calculate average monthly price considering price increase for the lifespan
            buy_price = self.average_price_increase(pd.Series([buy_price]), price_increase=incl_price_increase).iloc[0]
            sell_price = self.average_price_increase(pd.Series([sell_price]), price_increase=incl_price_increase).iloc[0]
        
        # Get the self-consumption percentage for the current month
        month_name = df_prices.loc[month - 1, 'Month']
        self_cons_percent = monthly_self_cons_percentages[month_name]
        
        # Self-consumed and exported energy
        df_temp[f"self_consumed_energy_kWh_{month}"] = df_temp[month] * self_cons_percent
        df_temp[f"exported_energy_kWh_{month}"] = df_temp[month] * (1 - self_cons_percent)
        df_temp[f'pv_self_cons_euro_{month}'] = df_temp[f"self_consumed_energy_kWh_{month}"] * buy_price
        df_temp[f'pv_to_grid_euro_{month}'] = df_temp[f"exported_energy_kWh_{month}"] * sell_price
        
    # Add to total kWh
    df_temp['PV_self_cons_kWh'] = df_temp[[f"self_consumed_energy_kWh_{month}" for month in monthly_columns]].sum(axis=1).round(3)
    df_temp['PV_to_grid_kWh'] = df_temp[[f"exported_energy_kWh_{month}" for month in monthly_columns]].sum(axis=1).round(3)
    
    # Monetary values
    df_temp['PV_self_cons_Euro'] = df_temp[[f"pv_self_cons_euro_{month}" for month in monthly_columns]].sum(axis=1).round(2)
    df_temp['PV_to_grid_Euro'] = df_temp[[f"pv_to_grid_euro_{month}" for month in monthly_columns]].sum(axis=1).round(2)
    
    df_results = df_temp[['build_id', 'census_id', 'PV_self_cons_kWh', 'PV_to_grid_kWh', 'PV_self_cons_Euro', 'PV_to_grid_Euro']]

    #del df_temp
    return pd.DataFrame(df_results)

def calculate_euac(df, pv_degradation_rate, energy_price_increase, discount_rate, pv_lifespan, envelope_lifespan):
    """
    Calculate EUAC for PV and envelope systems with degradation, energy price increase, and separate lifespans.

    Parameters:
    - df (pd.DataFrame): Input data with columns for PV and envelope costs, savings, and investments.
    - pv_degradation_rate (float): Annual PV degradation rate (e.g., 0.005 for 0.5%).
    - energy_price_increase (float): Annual energy price increase rate (e.g., 0.02 for 2%).
    - discount_rate (float): Discount rate (e.g., 0.05 for 5%).
    - pv_lifespan (int): Lifespan of the PV system in years.
    - envelope_lifespan (int): Lifespan of the envelope system in years.

    Returns:
    - pd.DataFrame: DataFrame with an additional column for total EUAC per building.
    """
    def crf(r, n):
        """Calculate Capital Recovery Factor."""
        return r * (1 + r) ** n / ((1 + r) ** n - 1)
    
    def annuity_factor(r, n):
        """Calculate Annuinity Factor."""
        annuity_factor = ((1 + r) ** n - 1) / (r * (1 + r) ** n)
        return annuity_factor

    # Calculate CRFs
    pv_crf = crf(discount_rate, pv_lifespan)
    envelope_crf = crf(discount_rate, envelope_lifespan)

    # Initialize EUAC columns
    df['PV_EUAC'] = 0
    df['Envelope_EUAC'] = 0
    df['Total_EUAC'] = 0

    for index, row in df.iterrows():
        # PV EUAC calculation
        pv_initial_investment = row['PV_Initial_Ivestment_Euro']
        pv_annual_maintenance = row['PV_Annual_Maint_Euro']
        pv_self_cons_euro = row['PV_self_cons_Euro']

        pv_annual_savings = sum(
            pv_self_cons_euro * (1 - pv_degradation_rate) ** (t - 1) * (1 + energy_price_increase) ** (t - 1)
            / (1 + discount_rate) ** t
            for t in range(1, pv_lifespan + 1)
        )
        pv_euac = pv_initial_investment * pv_crf + pv_annual_maintenance - pv_annual_savings

        # Envelope EUAC calculation
        envelope_initial_investment = row['Envelope_Initial_Ivestment_Euro']
        envelope_annual_maintenance = row['Envelope_Annual_Maint_Euro']
        envelope_euac = envelope_initial_investment * envelope_crf + envelope_annual_maintenance

        # Combine EUAC
        total_euac = pv_euac + envelope_euac

        # Assign values to DataFrame
        df.at[index, 'PV_EUAC'] = pv_euac
        df.at[index, 'Envelope_EUAC'] = envelope_euac
        df.at[index, 'Total_EUAC'] = total_euac

    return df

# Example usage
data = {
    'build_id': [1, 2],
    'PV_self_cons_Euro': [2000, 2500],
    'PV_to_grid_Euro': [500, 600],
    'Envelope_Initial_Ivestment_Euro': [10000, 15000],
    'Envelope_Annual_Maint_Euro': [500, 600],
    'PV_Initial_Ivestment_Euro': [8000, 10000],
    'PV_Annual_Maint_Euro': [300, 400]
}
df = pd.DataFrame(data)

# Parameters
pv_degradation_rate = 0.005  # 0.5% degradation
energy_price_increase = 0.02  # 2% increase
discount_rate = 0.05  # 5% discount rate
pv_lifespan = 20  # PV lifespan in years
envelope_lifespan = 30  # Envelope lifespan in years

# Calculate EUAC
df_euac = calculate_euac(df, pv_degradation_rate, energy_price_increase, discount_rate, pv_lifespan, envelope_lifespan)


#%% Function Testing for PV savings with degradation
import pandas as pd

def calculate_pv_savings_df(initial_savings, degradation_rate, discount_rate, lifespan):
    """
    Calculate the PV savings with degradation and return a detailed DataFrame.

    Parameters:
    - initial_savings (float): Initial annual savings in the first year.
    - degradation_rate (float): Annual degradation rate (e.g., 0.005 for 0.5%).
    - discount_rate (float): Discount rate (e.g., 0.05 for 5%).
    - lifespan (int): Total lifespan of the PV system in years.

    Returns:
    - pd.DataFrame: A DataFrame with detailed calculations and total savings.
    """
    # Initialize lists to store year-by-year data
    years = []
    annual_savings_list = []
    discounted_savings_list = []
    
    # Loop through each year
    for year in range(1, lifespan + 1):
        # Calculate annual savings after degradation
        annual_savings = initial_savings * (1 - degradation_rate) ** (year - 1)
        # Calculate discounted savings
        discounted_savings = annual_savings / (1 + discount_rate) ** year
        
        # Append to lists
        years.append(year)
        annual_savings_list.append(annual_savings)
        discounted_savings_list.append(discounted_savings)
    
    # Create DataFrame
    data = {
        "Year": years,
        "Annual Savings": annual_savings_list,
        "Discounted Savings": discounted_savings_list
    }
    df = pd.DataFrame(data)
    
    # Add Total PV Savings row
    total_pv_savings = df["Discounted Savings"].sum()
    total_row = pd.DataFrame({
        "Year": ["Total"],
        "Annual Savings": [""],
        "Discounted Savings": [total_pv_savings]
    })
    df = pd.concat([df, total_row], ignore_index=True)
    
    return df

# Example usage
initial_savings = 1000  # Annual savings in year 1 ($)
degradation_rate = 0.005  # Degradation rate (0.5%)
discount_rate = 0.05  # Discount rate (5%)
lifespan = 20  # Lifespan of the PV system (20 years)

# Calculate savings and display DataFrame
df_pv_savings = calculate_pv_savings_df(initial_savings, degradation_rate, discount_rate, lifespan)

#%% CHARTING FUNCTION TESTING
#a. NPV Bar Chart
"""plt.figure(figsize=(10, 6))
sns.barplot(data=agg_df, x='census_id', y='NPV_per_dwelling', hue='census_id', palette='Blues_d')
plt.title('Total Net Present Value (NPV) per Census Section')
plt.xlabel('Census ID')
plt.ylabel('NPV (€)')
plt.tight_layout()
plt.show()"""
#b. EUAC Bar Chart
plt.figure(figsize=(10, 6))
plot_df = combined_df[combined_df["Filter"]=="Comb0ref_case"]
sns.barplot(data=plot_df, x='census_id', y='EUAC_per_m2', palette='Greens_d')#,hue='census_id')
plt.legend(loc='lower right', title= f'census_id_{plot_df["Filter"].unique()[0]}')  # Set legend location to lower right corner
plt.title(f'Total Equivalent Uniform Annual Cost (EUAC) per Census Section [{plot_df["Filter"].unique()[0]}]')
plt.xlabel('Census ID')
plt.ylabel('EUAC per dwelling (€)')
plt.tight_layout()
plt.show()
#%% COMBINING NPV AND EUAC IN ONE CHART
#c. Combined NPV and EUAC Bar Chart
# Melt the DataFrame for seaborn
melted_agg_df = agg_df.melt(id_vars='census_id', value_vars=['NPV', 'EUAC'],
                            var_name='Metric', value_name='Value')

plt.figure(figsize=(12, 7))
sns.barplot(data=melted_agg_df, x='census_id', y='Value', hue='Metric')
plt.title('NPV and EUAC per Census Section')
plt.xlabel('Census ID')
plt.ylabel('Value (€)')
plt.legend(title='Metric')
plt.tight_layout()
plt.show()

#%% SENSITIVITY ANALYSIS
# Calculate additional statistics if needed
summary_df = agg_df.copy()
summary_df['Average_NPV'] = summary_df['NPV'] / df.groupby('census_id')['build_id'].count().values
summary_df['Total_EUAC'] = summary_df['EUAC']

# Select relevant columns
summary_table = summary_df[['census_id', 'NPV', 'Average_NPV', 'EUAC']]

# Display the table
print(summary_table)

# Alternatively, display using matplotlib
fig, ax = plt.subplots(figsize=(8, 3))  # Adjust size as needed
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=summary_table.values,
                 colLabels=summary_table.columns,
                 loc='center',
                 cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title('Summary of Economic Indicators per Census Section')
plt.show()

#4. Sensitivity Analysis: NPV vs Discount Rates


# Define a range of discount rates
discount_rates = np.arange(0.01, 0.11, 0.01)  # 1% to 10%

# Function to calculate NPV for different rates
def compute_npv_sensitivity(df, rates, n):
    sensitivity_results = {}
    for r in rates:
        df['NPV_Sens'] = df.apply(lambda row: calculate_npv(row, r, n), axis=1)
        sensitivity_results[r] = df.groupby('census_id')['NPV_Sens'].sum()
    sensitivity_df = pd.DataFrame(sensitivity_results).reset_index()
    sensitivity_df = sensitivity_df.melt(id_vars='census_id', var_name='Discount_Rate', value_name='NPV')
    sensitivity_df['Discount_Rate'] = sensitivity_df['Discount_Rate'].astype(float)
    return sensitivity_df

# Calculate sensitivity
sensitivity_df = compute_npv_sensitivity(df, discount_rates, analysis_period)

# Plot Sensitivity Analysis
plt.figure(figsize=(12, 7))
sns.lineplot(data=sensitivity_df, x='Discount_Rate', y='NPV', hue='census_id', marker='o')
plt.title('Sensitivity Analysis: NPV vs Discount Rate per Census Section')
plt.xlabel('Discount Rate')
plt.ylabel('NPV (€)')
plt.legend(title='Census ID')
plt.tight_layout()
plt.show()

#HEATMAP

# Example: Heatmap of NPV per build_id and census_id
pivot_npv = df.pivot_table(index='build_id', columns='census_id', values='NPV')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_npv, annot=True, fmt=".0f", cmap='YlGnBu')
plt.title('Heatmap of NPV per Building and Census Section')
plt.xlabel('Census ID')
plt.ylabel('Build ID')
plt.tight_layout()
plt.show()

#Box Plot

plt.figure(figsize=(10, 6))
sns.boxplot(data=agg_df, x='census_id', y='NPV', palette='Pastel1')
plt.title('Distribution of NPV per Census Section')
plt.xlabel('Census ID')
plt.ylabel('NPV (€)')
plt.tight_layout()
plt.show()

#Scatter Plot

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Initial_Investment', y='NPV', hue='census_id', palette='Set2', s=100)
plt.title('Initial Investment vs NPV per Building')
plt.xlabel('Initial Investment (€)')
plt.ylabel('NPV (€)')
plt.legend(title='Census ID')
plt.tight_layout()
plt.show()


# %% PLOT FROM BRUCK et ALL article
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns

def plotNPV(df, column_mapping, filename):
    """
    Plots the NPV composition using data from a specific DataFrame.
    
    Parameters:
    - df: pandas DataFrame containing the data to plot.
    - column_mapping: dict with keys ["names", "NPV", "CAPEX", "fix_costs", "var_costs", "revenue"] 
      mapping to column names in the DataFrame.
    - filename: str, name of the file to save the plot.
    """
    labels = df[column_mapping["names"]].tolist()
    x = np.arange(len(labels))
    width = 0.35

    # Extract values and scale
    NPV = df[column_mapping["NPV"]] / 1e6
    inv = -df[column_mapping["CAPEX"]] / 1e6
    fix = -df[column_mapping["fix_costs"]] / 1e6
    #var = -df[column_mapping["var_costs"]] / 1e6
    rev = df[column_mapping["revenue"]] / 1e6

    sns.set_theme(style='darkgrid')
    fig, ax = plt.subplots()
    ax.grid()

    # Bar plots
    ax.bar(x - width/2, inv, width, label="CAPEX", color="#bc5090")
    ax.bar(x - width/2, fix, width, bottom=inv, label="Fix Costs", color="#58508d")
    #ax.bar(x - width/2, var, width, bottom=(fix + inv), label="Variable Costs", color="#ff6361")
    ax.bar(x - width/2, rev, width, bottom=0, label="Revenue", color="#20a9a6")
    ax.bar(x + width/2, NPV, width, label="NPV", color="#ffa600")

    # Labels and formatting
    ax.set_ylabel("NPV Composition [Mio €]")
    ax.set_title("NPV Composition by Scenario")
    ax.set_xticks(x)
    ax.tick_params(axis='x', rotation=45)
    ax.set_xticklabels(labels)
    ax.grid(linestyle='-', linewidth='0.5', color='white')

    # Annotate NPV bars
    for i in range(len(x)):
        ax.text(x[i] + width/2, NPV.iloc[i] - 0.1, f"{NPV.iloc[i]:.2f}", 
                color='black', ha="center", size="x-small")

    # Legend
    fontP = FontProperties()
    fontP.set_size('xx-small')
    ax.legend(prop=fontP)

    # Adjust layout and save
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(filename + ".png", dpi=300)
    plt.show()
    plt.close(fig)



data = {
    "Scenario": ["Scenario A", "Scenario B", "Scenario C"],
    "NPV": [5000000, 7000000, 6000000],  # Net Present Value in €
    "I": [2000000, 3000000, 2500000],  # CAPEX in €
    "fix_share": [1000000, 1200000, 1100000],  # Fixed costs in €
    "var_share": [500000, 800000, 600000],  # Variable costs in €
    "rev_share": [8500000, 10000000, 9500000],  # Revenue in €
}

# Create the DataFrame
df = pd.DataFrame(data)
column_mapping = {
    "names": "Scenario",
    "NPV": "NPV",
    "CAPEX": "I",
    "fix_costs": "fix_share",
    "var_costs": "var_share",
    "revenue": "rev_share"
}

plotNPV(df, column_mapping, "output_filename")

# %%

def plotNPV(df, column_mapping, filename):
    """
    Plots the NPV composition using data from a specific DataFrame.
    
    Parameters:
    - df: pandas DataFrame containing the data to plot.
    - column_mapping: dict with keys ["names", "NPV", "CAPEX", "fix_costs", "var_costs", "revenue"] 
      mapping to column names in the DataFrame.
    - filename: str, name of the file to save the plot.
    """
    labels = df[column_mapping["names"]].tolist()
    x = np.arange(len(labels))
    width = 0.35

    # Extract values and scale
    NPV = df[column_mapping["EUAC"]] / 1e6
    inv = -df[column_mapping["CAPEX"]] / 1e6
    fix = -df[column_mapping["fix_costs"]] / 1e6
    #var = -df[column_mapping["var_costs"]] / 1e6
    rev = df[column_mapping["revenue"]] / 1e6

    sns.set_theme(style='darkgrid')
    fig, ax = plt.subplots()
    ax.grid()

    # Bar plots
    ax.bar(x - width/2, inv, width, label="CAPEX", color="#bc5090")
    ax.bar(x - width/2, fix, width, bottom=inv, label="Fix Costs", color="#58508d")
    #ax.bar(x - width/2, var, width, bottom=(fix + inv), label="Variable Costs", color="#ff6361")
    ax.bar(x - width/2, rev, width, bottom=0, label="Revenue", color="#20a9a6")
    ax.bar(x + width/2, NPV, width, label="EUAC", color="#ffa600")

    # Labels and formatting
    ax.set_ylabel("EUAC Composition [Mio €]")
    ax.set_title("EUAC Composition by Scenario with rooftop PV")
    ax.set_xticks(x)
    ax.tick_params(axis='x', rotation=45)
    ax.set_xticklabels(labels)
    ax.grid(linestyle='-', linewidth='0.5', color='white')
    # Set y-axis with 5 step increments
    """y_min, y_max = ax.get_ylim()
    ax.set_yticks(np.arange(y_min, y_max, 5))"""
    # Annotate NPV bars
    for i in range(len(x)):
        ax.text(x[i] + width/2, NPV.iloc[i] - 0.1, f"{NPV.iloc[i]:.2f}", 
                color='black', ha="center", size="x-small")

    # Legend
    fontP = FontProperties()
    fontP.set_size('xx-small')
    ax.legend(prop=fontP)

    # Adjust layout and save
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(filename + ".png", dpi=300)
    plt.show()
    plt.close(fig)

combined_df_neighbourhood = combined_df.groupby('Filter').agg({
    'EUAC': 'sum',
    "Envelope_EUAC": 'sum',
    #'NRPE_kWh_per_dwelling': 'sum',
    #'Envelope_EUAC_per_dwelling': 'sum',
    #'EUAC_per_dwelling': 'sum',
    dwelling_col: 'sum',
    "Envelope_Initial_I": 'sum',
    "PV_I_C": 'sum',
    "Envelope_Annual_M": 'sum',
    "PV_M_C": 'sum',
    "Envelope_Annual_Savings": 'sum',
    "PV_Annual_Savings": 'sum'
}).reset_index()
combined_df_neighbourhood['CAPEX'] = combined_df_neighbourhood['Envelope_Initial_I'] + combined_df_neighbourhood['PV_I_C']
combined_df_neighbourhood['fix_share'] = combined_df_neighbourhood['Envelope_Annual_M'] + combined_df_neighbourhood['PV_M_C']
combined_df_neighbourhood['revenue'] = combined_df_neighbourhood['Envelope_Annual_Savings'] + combined_df_neighbourhood['PV_Annual_Savings']


column_mapping = {
    "names": "Filter",
    "EUAC": "Envelope_EUAC",# "Envelope_EUAC", "EUAC"
    "CAPEX": "CAPEX",
    "fix_costs": "fix_share",
    "var_costs": "var_share",
    "revenue": "revenue"
}

plotNPV(combined_df_neighbourhood, column_mapping, "output_filename")
# %%
