#%%
import pandas as pd
import geopandas as gpd
import os 
import numpy as np

class InspireDBAssigner:
    def __init__(self, path='data/'):
        self.PATH = path
        self.data = None

    def load_data(self, filename='03_footprint_subtracted_facades_and_s_v_volume_area.geojson'):
        # Real data 
        self.data = gpd.read_file(os.path.join(self.PATH, filename))
        self.data = self.data.rename(columns={'Ano_Constr': 'year_constr'})

    def recalculate_surface_area(self):
        # Recalculate a undetached surface area
        cols = ['fa_area_N', 'fa_area_NE', 'fa_area_E', 'fa_area_SE',
                'fa_area_S', 'fa_area_SW', 'fa_area_W', 'fa_area_NW']
        self.data['Total_fa_area'] = self.data[cols].sum(axis=1).round(2)

    def assign_window_areas(self, windows_to_wall_ratio=0.24):
        # Put average w2wall ratios for each orientation depending on df_average_w2wall age of construction
        self.data["Tot_w2wall"] = windows_to_wall_ratio
        self.data["Tot_window_area"] = self.data["Total_fa_area"] * self.data["Tot_w2wall"]

    def calculate_f_v_ratio(self, row, surface_area_column='Total_fa_area', volume_column='volume'):
        surface_area = row[surface_area_column]
        volume = row[volume_column]
        return round(surface_area / volume, 4)

    def get_period_from_year(self, year_of_construction):
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

    #EQUATIONS

    def calculate_heating_value(self, s_v_ratio, year, value_type):
        """Calculate the heating demand or consumption of a building based on its surface-to-volume ratio and year of construction."""
        period = self.get_period_from_year(year) # get the period from the year of construction
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
                "pre45": (83.278,1.8469),
                "45_70": (78.437, -0.0263),
                "70_80": (77.777, -0.0113),
                "80_90": (57.51, -1.5211),
                "90_00": (57.611, -2.4139),
                "00_10": (24.506, 0.7135)
            }
        }
        if value_type in coefficients and period in coefficients[value_type]:
            a, b = coefficients[value_type][period]
            return round(a * s_v_ratio + b, 4)
        else:
            raise ValueError("Invalid period or value type specified. Value types: demand, consumption. Periods: pre45, 45_70, 70_80, 80_90, 90_00, 00_10.")

    def calculate_cooling_value(self, s_v_ratio, year, value_type):
        """Calculate the cooling demand or consumption of a building based on its surface-to-volume ratio and year of construction."""
        period = self.get_period_from_year(year) # get the period from the year of construction
        coefficients = {
            "demand":{ # Southern Dry Climate
                "00_10": (4.9195,22.673),
                "90_00": (7.0068,21.126),
                "80_90": (6.7815,21.171),
                "70_80": (7.2643,18.706),
                "45_70": (7.6174,18.718),
                "pre45": (13.927,17.514)
            },
            "consumption":{ # Southern Dry Climate
                "00_10": (1.9678,9.0691),
                "90_00": (2.8027,8.4502),
                "80_90": (2.7126,8.4686),
                "70_80": (2.9057,7.4823),
                "45_70": (3.047,7.4873),
                "pre45": (5.5709,7.0055)
            }
        }
        if value_type in coefficients and period in coefficients[value_type]:
            a, b = coefficients[value_type][period]
            return round(a * s_v_ratio + b, 4)
        else:
            raise ValueError("Invalid period or value type specified. Value types: demand, consumption. Periods: pre45, 45_70, 70_80, 80_90, 90_00, 00_10.")

    def co2_emissions(self, s_v_ratio, year):
        """Calculate the CO2 emissions of a building based on its surface-to-volume ratio and year of construction."""
        period = self.get_period_from_year(year) # get the period from the year of construction
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

    def process(self):
        # Calculate f_v_ratio
        self.data['f_v_ratio'] = self.data.apply(self.calculate_f_v_ratio, axis=1).round(4)

        # Calculate heating and cooling values
        self.data['HDem_iNSPiRE'] = self.data.apply(
            lambda x: self.calculate_heating_value(x.f_v_ratio, x.year_constr, 'demand'), axis=1)
        self.data['HCons_iNSPiRE'] = self.data.apply(
            lambda x: self.calculate_heating_value(x.f_v_ratio, x.year_constr, 'consumption'), axis=1)
        self.data['CDem_iNSPiRE'] = self.data.apply(
            lambda x: self.calculate_cooling_value(x.f_v_ratio, x.year_constr, 'demand'), axis=1)
        self.data['CCons_iNSPiRE'] = self.data.apply(
            lambda x: self.calculate_cooling_value(x.f_v_ratio, x.year_constr, 'consumption'), axis=1)
        self.data['H_CO2_iNSPiRE'] = self.data.apply(
            lambda x: self.calculate_heating_value(x.f_v_ratio, x.year_constr, 'co2'), axis=1)
        print(f"cols: {self.data.columns}")

    def save(self, filename='05_buildings_with_energy_and_co2_values.geojson'):
        self.data.to_file(os.path.join(self.PATH, filename), driver='GeoJSON')
        print(f"Data saved to {filename}")

    

# Example usage:
# assigner = InspireDBAssigner()
# assigner.load_data()
# assigner.recalculate_surface_area()
# assigner.assign_window_areas(windows_to_wall_ratio=0.24)
# assigner.process()
# assigner.save()

# %%
