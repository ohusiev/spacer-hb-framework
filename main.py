#%%
import importlib
import pandas as pd


class DefaultPathsAndFileNames:
    """
    A class to hold a set of default paths for a standardized and replicable approach.
    """

    def __init__(self, excel_path='files_directories_naming.xlsx', sheet_name="Sheet1"):
        """
        Initialize the DefaultPaths class.

        Args:
            excel_path (str, optional): Path to the Excel file containing variable names.
            sheet_name (str, optional): Name of the sheet to read from the Excel file.
        """
        self.BASE_DIR = "your_custom_name"  # Replace with your base directory
        self.data_dir = f"{self.BASE_DIR}/data"
        self.lidar_dir = f"{self.BASE_DIR}/lidar"
        self.vector_dir = f"{self.BASE_DIR}/vector"
        self.raster_dir = f"{self.BASE_DIR}/raster"

        # Define file names
        self.LIDAR_FILE = "lidar.las"
        self.BUILDING_FOOTPRINT_FILE = "building_footprint.shp"
        self.STATISTICAL_CENSUS_FILE = "statistical_census.shp"
        self.WHITEBOX_RT_ANALYSIS_FILE = "wb_rt_analysis.shp"

        # Extract file names from Excel and set them as attributes
        if excel_path and sheet_name:
            df=self._set_file_names_from_excel(excel_path, sheet_name)
            print(df)
            self.PATH = os.getcwd() if pd.isna(df.loc['PATH', 'File']) else f"{df.loc['PATH', 'File']}"
            self.LIDAR_FILE = f"{df.loc['LIDAR_FILE', 'File']}"
            self.BUILDING_FOOTPRINT_FILE = f"{df.loc['BUILDING_FOOTPRINT_FILE','File']}"
            self.STATISTICAL_CENSUS_FILE = f"{df.loc['STATISTICAL_CENSUS_FILE','File']}"
            self.WHITEBOX_RT_ANALYSIS_FILE = f"{df.loc['WHITEBOX_RT_ANALYSIS_FILE','File']}"

            #Print the file names for demonstration
            print("File names from Excel:\n")
            print(f"PATH: {self.PATH}")
            print(f"LIDAR_FILE: {self.LIDAR_FILE}")
            print(f"BUILDING_FOOTPRINT_FILE: {self.BUILDING_FOOTPRINT_FILE}")
            print(f"STATISTICAL_CENSUS_FILE: {self.STATISTICAL_CENSUS_FILE}")
            print(f"WHITEBOX_RT_ANALYSIS_FILE: {self.WHITEBOX_RT_ANALYSIS_FILE}")

    def get(self, key):
        """
        Retrieve the path associated with a given key.

        Args:
            key (str): The key for the desired path.

        Returns:
            str: The corresponding path, or None if the key does not exist.
        """
        return getattr(self, key, None)

    def _set_file_names_from_excel(self, excel_path, sheet_name):
        """
        Extract file names from an Excel sheet and set them as attributes.

        Args:
            excel_path (str): Path to the Excel file.
            sheet_name (str): Name of the sheet to read.
        """
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name, index_col=0)
            #print(df.head())
            #print(df.loc['LIDAR_FILE','File'])  # Print the row corresponding to 'LIDAR_FILE' for debugging
            # Assuming the file names are in the first column
            #file_names = df.iloc[:, 0].dropna().tolist()
            return df
        except Exception as e:
            print(f"Error reading Excel file: {e}")
    
#%%
import importlib
import os
import geopandas as gpd
import python.mod_02_pv_power_month as pv_rooftop_analysis
import python.mod_03_geopandas_facade_analyser as facade_analyser
import python.mod_04_ener_consum_profile_assigner as ener_cons_profile_assigner
import python.mod_04_energy_profile_aggregation as profile_aggregation
import python.mod_04_self_consump_estimation as self_cons_estimator
import python.mod_05_inspire_db_assigner as inspire_db_assigner

""" 
pivot_rooftop_analysis = importlib.import_module("01_pivot_rooftop_data")
pv_rooftop_analysis = importlib.import_module("02_pv_calc_rooftop.pv_power_month")

facade_analyser = importlib.import_module("03_geopandas_facade_analyser")
ener_cons_profile_assigner = importlib.import_module("04_ener_consum_profile_assigner")
profile_aggregation = importlib.import_module("04_energy_profile_aggregation")
self_cons_estimator = importlib.import_module("04_self_consump_estimation")

inspire_db_assigner = importlib.import_module("05_inspire_db_assigner")
"""
#%%
# Example usage
default_paths = DefaultPathsAndFileNames(excel_path='files_directories_naming.xlsx', sheet_name="Sheet1")
print(default_paths.get("BASE_DIR"))  # Access the base directory
#get a dataframe with the file names
print(default_paths.get("LIDAR_FILE"))  # Access the LIDAR file name

#%% 01_pivot_rooftop_data

rt = pivot_rooftop_analysis.PivotRooftopAnalysis(file_dir=os.getcwd())

df_segments_wb_rooftop_analysis, gdf_building_footprint = rt.process_DataFrames()
df_segments_wb_rooftop_analysis, building_footprint = rt.pivot_whitebox_rooftop_analysis(df_segment=df_segments_wb_rooftop_analysis, df_buildings_footprint=gdf_building_footprint, col_name="s_area", key_ids=["build_id", "census_id"])
#%% 02_pv_calc_rooftop
pv_calculator = pv_rooftop_analysis.PVMonthCalculator('241211_econom_data.xlsx')
#pv_calculator.calculate()
#%% 03_geopandas_facade_analyser
# from 03_geopandas_facade_analyser import FacadeAnalyser

# Initialize the analyser (optionally provide base_dir or file_path)
analyser = facade_analyser.FacadeAnalyser()

# Load polygons
analyser.load_polygons()
# (Optional) Merge height data if available
height_data_path = os.path.join(analyser.base_dir, 'height.geojson')
if os.path.exists(height_data_path):
    gdf_height = gpd.read_file(height_data_path).round(4)
    analyser.polygons_gdf = analyser.polygons_gdf.merge(
        gdf_height[['build_id', 'h_mean', 'h_stdev', 'h_min', 'h_max']],
        on='build_id', how='left'
    )
# Calculate area and perimeter
analyser.polygons_gdf['f_area'] = round(analyser.polygons_gdf['geometry'].area, 4)
analyser.polygons_gdf['f_perimeter'] = round(analyser.polygons_gdf['geometry'].length, 4)

# Calculate facade lengths per orientation
facades_per_orientation_len_df = analyser.length_per_orientation()

# Find neighbors and their lengths per orientation
analyser.list_neighboring_polygons()
adjusted_facades_len_df = analyser.length_of_neighbors_per_orientation()

# Calculate surface area, volume, and s/v ratio
analyser.polygons_gdf['surface_area'] = analyser.polygons_gdf.apply(analyser.calculate_surface_area, axis=1)
analyser.polygons_gdf['volume'] = analyser.polygons_gdf.apply(analyser.calculate_volume, axis=1)
analyser.polygons_gdf['s_v_ratio'] = analyser.polygons_gdf.apply(analyser.calculate_s_v_ratio, axis=1).round(4)

# Subtract facade lengths
result_df = analyser.subtract_facade_len_from_adjusted_sides(facades_per_orientation_len_df, adjusted_facades_len_df)

# Calculate facade area per orientation
fadace_length_cols = {'len_N': 'N', 'len_NE': 'NE', 'len_E': 'E',
                        'len_SE': 'SE', 'len_S': 'S', 'len_SW': 'SW', 'len_W': 'W', 'len_NW': 'NW'}
for key, value in fadace_length_cols.items():
    result_df[f"fa_area_{value}"] = [0 if x < 0.1 else x for x in result_df[key]] * result_df['h_mean']

# Save results if needed
result_df.to_file("data/03_footprint_subtracted_facades_and_s_v_volume_area.geojson", driver="GeoJSON", index=False)
result_df.drop(columns=['geometry']).to_csv("data/03_footprint_subtracted_facades_and_s_v_volume_area.csv", index=False)


#%% 04_ener_consum_profile_assigner
assigner = ener_cons_profile_assigner.EnergyConsumptionProfileAssigner()
assigner.process()
#%% 04_energy_profile_aggregation
PATH = r"C:\\Users\\Oleksandr-MSI\\Documentos\\GitHub\\spacer-hb-framework\\LoadProGen\\Bilbao"
# Instantiate and use the class
#pv_pct_list = [0.25, 0.5, 0.75]
#for pv_pct in pv_pct_list:
profile_mapping = {
    #"1P_Occup": "ND1 Single Occupied Dwellings",
    "1P_Work": "CHR07 Single with work",
    "Stu_Work": "CHR13 Student with Work",
    "1P_Ret": "CHR30 Single, Retired Man/Woman", #redo profile, it has 1 min time step - DONE
    #"2P_Occup": "ND2 Two People Occupied Dwellings",
    "Couple_Work": "CHR01 Couple both at Work",
    "Couple_65+": "CHR16 Couple over 65 years",
    "1P_1Ch_Work": "CHR22 Single woman, 1 child, with work",
    #"3-5P_Occup": "ND3-5 Three to Five People Occupied Dwellings",
    "Fam_1Ch_Work": "CHR03 Family, 1 child, both at work",
    "Stu_Share": "CHR52 Student Flatsharing",
    "Fam_2Ch_Work": "CHR27 Family both at work, 2 children",
    "Fam_3Ch_Work": "CHR41 Family with 3 children, both at work",
    "Fam_1Ch_1Wrk1Hm": "CHR45 Family with 1 child, 1 at work, 1 at home",
    "Fam_3Ch_1Wrk1Hm": "CHR20 Family one at work, one work home, 3 children",
    "Fam_3Ch_HmWrk": "CHR59 Family, 3 children, parents without work/work at home",
    #"6-9P_Occup": "ND6-9 Six to Nine People Occupied Dwellings",
    "6-9P_Occup_id_1": "id_1", #CHR15 Multigenerational Home: working couple, 2 children, 2 seniors 
    #"6-9P_Occup_id_2": "id_1",
    "6-9P_Occup_id_3": "id_1",
    #"6-9P_Occup_id_4": "id_1",
    #"10+P_Occup": "ND10 (Ten or more People Occupied Dwellings)",
    "10+P_Occup_id_1": "id_1",
    "10+P_Occup_id_2": "id_1"
}
aggregator = profile_aggregation.EnergyProfileAggregator(PATH, profile_mapping, pv_pct=0.25)
#aggregator.plot_profiles()
result_pv_df, result_no_pv_df = aggregator.save_results()
#%% 04_self_consump_estimation
#pv_pct_list = [0.25, 0.5, 0.75]
estimator = self_cons_estimator.SelfConsumptionEstimator(
    data_struct_file='02_pv_calc_from_bond_rooftop/pv_energy.yml',
    rayon="otxarkoaga",
    pv_pct=0.25
)
estimator.time_alignment()
estimator.run_self_consumption()
monthly_agg, monthly_agg_no_pv = estimator.save_aggregated_profiles()
pv_census_aggreg_df = estimator.save_pv_profiles()
estimator.save_net_balance(monthly_agg, pv_census_aggreg_df)

#%% 05_inspire_db_assigner
heating_db_assigner = inspire_db_assigner.InspireDBAssigner()
heating_db_assigner.load_data()
heating_db_assigner.recalculate_surface_area()
heating_db_assigner.assign_window_areas(windows_to_wall_ratio=0.24)
heating_db_assigner.process()
heating_db_assigner.save()