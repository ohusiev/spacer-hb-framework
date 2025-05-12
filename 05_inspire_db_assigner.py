#%%
import pandas as pd
import geopandas as gpd
import os 
import numpy as np

PATH = 'data/'
#%%
# #Sample data
data = pd.DataFrame({
    's_v_ratio': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'year_of_construction': [1920, 1950, 1975, 1985, 1995, 2005]
})
#%%
# Real data 
data = gpd.read_file(os.path.join(PATH, '03_footprint_subtracted_facades_and_s_v_volume_area.geojson'))
data = data.rename(columns={'Ano_Constr': 'year_constr'})
#%%
"""

#Recalculate a undetached surface area
cols=['fa_area_N', 'fa_area_NE', 'fa_area_E', 'fa_area_SE',
       'fa_area_S', 'fa_area_SW', 'fa_area_W', 'fa_area_NW']
data['Total_fa_area'] = data[cols].sum(axis=1).round(2)
#Put average w2wall ratios for each orientation depending on df_average_w2wall age of construction
data["Tot_w2wall"] = 0.24
data["Tot_window_area"] = data["Total_fa_area"] * data["Tot_w2wall"]

data['f_v_ratio'] = data.apply(calculate_f_v_ratio, axis=1).round(4)


data['HDem_iNSPiRE'] = data.apply(lambda x: calculate_heating_value(x.f_v_ratio, x.year_constr, 'demand'), axis=1) # kWh/m2y 
data['HCons_iNSPiRE'] = data.apply(lambda x: calculate_heating_value(x.f_v_ratio, x.year_constr, 'consumption'), axis=1) # kWh/m2y

data['CDem_iNSPiRE'] = data.apply(lambda x: calculate_cooling_value(x.f_v_ratio, x.year_constr, 'demand'), axis=1) # kWh/m2y
data['CCons_iNSPiRE'] = data.apply(lambda x: calculate_cooling_value(x.f_v_ratio, x.year_constr, 'consumption'), axis=1) # kWh/m2y

data['H_CO2_iNSPiRE'] = data.apply(lambda x: calculate_heating_value(x.f_v_ratio, x.year_constr,'co2'), axis=1) # kgCO2/m2y
data.to_file(os.path.join(PATH, '05_buildings_with_energy_and_co2_values.geojson'), driver='GeoJSON')

"""
def calculate_f_v_ratio(row, surface_area_column='Total_fa_area', volume_column='volume'):
    surface_area = row[surface_area_column]
    volume = row[volume_column]
    return round(surface_area/volume, 4)
""" FOR EXTRA
#4. Calculate area and perimeter of buildings footprint polygons
file_dir = os.getcwd() +"/vector/osm_buildings/"

height_data_path = os.path.join(file_dir, 'height.geojson')
gdf_height = gpd.read_file(height_data_path).round(4)
data = data.merge(gdf_height[['CodEdifici', 'h_mean', 'h_stdev', 'h_min', 'h_max']], on='CodEdifici', how='left')

data['f_area'] = round(data['geometry'].area,4)#.round(2)
data['f_perimeter'] = round(data['geometry'].length,4)#.round(2)
data['surface_area'] = data.apply(calculate_surface_area, axis=1) # Simplification: assuming the building is a right rectangulasr prism
data['volume'] = data.apply(calculate_volume, axis=1) # Simplification: assuming the building is a right rectangular prism
data['f_v_ratio'] = data.apply(calculate_f_v_ratio, axis=1).round(4)

"""
#%%
def get_period_from_year(year_of_construction):
    period_mapping = {
        "pre45": range(1800, 1945),  # Pre-1945
        "45_70": range(1945, 1971),  # 1945-1970
        "70_80": range(1971, 1981),  # 1971-1980
        "80_90": range(1981, 1991),  # 1981-1990
        "90_00": range(1991, 2001),  # 1991-2000
        "00_10": range(2001, 2015)   # 2001-2010
    }

    for period, years in period_mapping.items():
        if year_of_construction in years:
            return period
#%% EQUATIONS

def calculate_heating_value(s_v_ratio, year, value_type):
    """Calculate the heating demand or consumption of a building based on its surface-to-volume ratio and year of construction.
        Parameters
    ----------
    s_v_ratio : float
        The ratio of the surface area of the building to its volume.
    year : int
        The year of construction of the building.
    value_type : str
        The type of value to calculate. Can be either 'demand' or 'consumption'.
    Returns
    -------
    float
        The calculated value.
        Raises
        ------
        ValueError  
        If an invalid period or value type is specified.
    """

    period = get_period_from_year(year) # get the period from the year of construction
    """ 
        coefficients = {#Mediterranean Climate
        "demand": {
            "pre45": (289.63,-36.906),
            "45_70": (206.79,-10.533),
            "70_80": (185.44,-8.8008),
            "80_90": (97.112,8.7629),
            "90_00": (101.98,2.4171),
            "00_10": (90.444,4.5508)
        },
        "consumption": {#Mediterranean Climate
            "pre45": (362.03,-46.132),
            "45_70": (258.49,-13.166),
            "70_80": (231.8,-11.001),
            "80_90": (121.39,10.954),
            "90_00": (127.47,3.0214),
            "00_10": (113.06,5.6885)
        }
    }    
    """

    coefficients = {#Southern Dry Climate
        "demand": {
            "pre45": (302.83,6.7161),
            "45_70": (285.23,-0.0956),
            "70_80": (282.82,-0.0412),
            "80_90": (209.13,5.5313),
            "90_00": (209.49,-8.7779),
            "00_10": (89.112,2.5945)
        },
        "consumption": {#Southern Dry Climate
            "pre45": (378.53,8.3952),
            "45_70": (356.53,-0.1195),
            "70_80": (353.53,-0.0515),
            "80_90": (261.41,-6.9141),
            "90_00": (261.87,-10.972),
            "00_10": (111.39,3.2431)
        },
        "co2": {#Southern Dry Climate
            "pre45": (83.278,1.8469), #83.278x + 1.8469
            "45_70": (78.437, -0.0263), #78.437x - 0.0263
            "70_80": (77.777, -0.0113),#77.777x - 0.0113
            "80_90": (57.51, -1.5211),#= 57.51x - 1.5211
            "90_00": (57.611, -2.4139),#57.611x - 2.4139
            "00_10": (24.506, 0.7135) #24.506x + 0.7135
        }
    }    
    if value_type in coefficients and period in coefficients[value_type]:
        a, b = coefficients[value_type][period]
        return round(a * s_v_ratio + b, 4)
    else:
        raise ValueError("Invalid period or value type specified. Value types: demand, consumption. Periods: pre45, 45_70, 70_80, 80_90, 90_00, 00_10.")
#%% 
def calculate_cooling_value(s_v_ratio, year, value_type):
    """Calculate the cooling demand or consumption of a building based on its surface-to-volume ratio and year of construction.
        Parameters
    ----------
    s_v_ratio : float
        The ratio of the surface area of the building to its volume.
    year : int
        The year of construction of the building.
    value_type : str
        The type of value to calculate. Can be either 'demand' or 'consumption'.
    Returns
    -------
    float
        The calculated value.
        Raises
        ------
        ValueError  
        If an invalid period or value type is specified.
    """

    period = get_period_from_year(year) # get the period from the year of construction
    coefficients = {
        "demand":{ # Southern Dry Climate
        "00_10": (4.9195,22.673), # period: 2000-2010 4.9195x + 22.673
        "90_00": (7.0068,21.126), # period: 1990-2000 7.0068x + 21.126
        "80_90": (6.7815,21.171), # period: 1980-1990 6.7815x + 21.171
        "70_80": (7.2643,18.706), # period: 1970-1980 7.2643x + 18.706
        "45_70": (7.6174,18.718), # period: 1945-1970 7.6174x + 18.718
        "pre45": (13.927,17.514) # period: pre 1945 13.927x + 17.514
            },
        "consumption":{ # Southern Dry Climate
        "00_10": (1.9678,9.0691), # period: 2000-2010 1.9678x + 9.0691
        "90_00": (2.8027,8.4502), # period: 1990-2000 2.8027x + 8.4502
        "80_90": (2.7126,8.4686), # period: 1980-1990 2.7126x + 8.4686
        "70_80": (2.9057,7.4823), # period: 1970-1980 2.9057x + 7.4823
        "45_70": (3.047,7.4873), # period: 1945-1970 3.047x + 7.4873
        "pre45": (5.5709,7.0055)} # period: pre 1945  5.5709x + 7.0055
    }
    
    """ 
    coefficients = {
        "demand":{ #%% Oceanic Climate
        "00_10": (-2.3128, 8.0613), # period: 2000-2010
        "90_00": (-3.951, 5.8427), # period: 1990-2000
        "80_90": (-2.7872, 2.6775), # period: 1980-1990
        "70_80": (-2.3416, 1.6599), # period: 1970-1980
        "45_70": (-1.6627, 1.2095), # period: 1945-1970
        "pre45": (-2.0134, 1.4633) # period: 1945-1970
            },
        "consumption":{ #%% Oceanic Climate
        "00_10": (-0.9251,3.2245), # period: 2000-2010 -0.9251x + 3.2245
        "90_00": (-0.8054, 0.5853), # period: 1990-2000 -0.8054x + 0.5853
        "80_90": (-1.1149, 1.071), # period: 1980-1990 -1.1149x + 1.071
        "70_80": (-0.9367, 0.664), # period: 1970-1980 -0.9367x + 0.664
        "45_70": (-0.6651, 0.4838), # period: 1945-1970 y = -0.6651x + 0.4838
        "pre45": (1.5804, 2.3371)} # period: 1945-1970 1.5804x + 2.3371
    }
    """
    if value_type in coefficients and period in coefficients[value_type]:
        a, b = coefficients[value_type][period]
        return round(a * s_v_ratio + b, 4)
    else:
        raise ValueError("Invalid period or value type specified. Value types: demand, consumption. Periods: pre45, 45_70, 70_80, 80_90, 90_00, 00_10.")
        return 'Error'
#%% 
def co2_emissions(s_v_ratio, year):
    """Calculate the CO2 emissions of a building based on its surface-to-volume ratio and year of construction.
        Parameters
    ----------
    s_v_ratio : float
        The ratio of the surface area of the building to its volume.
    year : int
        The year of construction of the building.
    Returns
    -------
    float
        The calculated CO2 emissions.
        Raises
        ------
        ValueError  
        If an invalid period is specified.
    """
    period = get_period_from_year(year) # get the period from the year of construction
    coefficients = { #Southern Dry CO2 emissions
        "pre45": (), #83.278x + 1.8469
        "45_70": (), #78.437x - 0.0263
        "70_80": (),#77.777x - 0.0113
        "80_90": (),#= 57.51x - 1.5211
        "90_00": (),#57.611x - 2.4139
        "00_10": () #24.506x + 0.7135
    }
    coefficients = { #Oceanic CO2 emissions
        "pre45": (96.541, 2.147),
        "45_70": (99.545, -0.2382),
        "70_80": (81.778, -1.8032),
        "80_90": (62.5, -3.541),
        "90_00": (135.29, -8.1396),
        "00_10": (23.451, -1.5582)
    }
    
    if period in coefficients:
        a, b = coefficients[period]
        return round(a * s_v_ratio + b, 4)
    else:
        raise ValueError("Invalid period specified.")
#%%