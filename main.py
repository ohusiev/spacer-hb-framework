#%%
import os
import pandas as pd
import geopandas as gpd
import python.mod_01_0_wb_automator as wb_automator
import python.mod_01_1_pivot_rooftop_data as pivot_rooftop_analysis
import python.mod_02_pv_power_month as pv_rooftop_analysis
import python.mod_03_geopandas_facade_analyser as facade_analyser
import python.mod_04_1_ener_consum_profile_assigner as ener_cons_profile_assigner
import python.mod_04_2_energy_profile_aggregation as profile_aggregation
import python.mod_04_3_self_consump_estimation as self_cons_estimator
import python.mod_05_0_inspire_db_assigner as inspire_db_assigner
import python.mod_05_1_simple_kpi_calc as kpi_calc
import python.mod_05_2_test_economic_analysis as economic_analysis
import python.mod_07_self_cons_scenarios_calc as self_cons_scenarios_calc 
import python.mod_07_geo_visualization as geo_visualization



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
default_paths = DefaultPathsAndFileNames(excel_path='241211_econom_data.xlsx', sheet_name="file_names")
print(default_paths.get("BASE_DIR"))  # Access the base directory
#get a dataframe with the file names
print(default_paths.get("LIDAR_FILE"))  # Access the LIDAR file name

#%% MODULE 1 00 wb_automatiser
wb_automator = wb_automator.RooftopAnalysisAutomatiserPython(
    path="C:/Users/Oleksandr-MSI/Documentos/GitHub/spacer-hb-framework",
    root_dir="pyqgis_wb_automator",
    case_study_name="bilbao_otxarkoaga",
    suffix="_v2",
    crs="epsg:25830",
    census_id="census_id",
    building_ids=["build_id"]
    )
wb_automator.run()
# 
# 
#%% 
# MODULE 1 01_pivot_rooftop_data

rt = pivot_rooftop_analysis.PivotRooftopAnalysis(file_dir=os.getcwd(),path_to_wb_rooftop_analysis='pyqgis_wb_automator\\bilbao_otxarkoaga_v2\\00_wb_rt_analysis_bilbao_otxarkoaga_v2_segments_xy_coord.geojson', path_to_buildings_footprint='vector\\buildings_footprint\\etrs_25830\\buildings_inspire_clip_oxarkoaga+census.shp')

df_segments_wb_rooftop_analysis, gdf_building_footprint = rt.process_DataFrames()
df_segments_wb_rooftop_analysis, building_footprint = rt.pivot_whitebox_rooftop_analysis(df_segment=df_segments_wb_rooftop_analysis, df_buildings_footprint=gdf_building_footprint, col_name="s_area", key_ids=["build_id", "census_id"])
#%% 
# MODULE 2 02_pv_calc_rooftop
pv_calculator = pv_rooftop_analysis.PVMonthCalculator('241211_econom_data.xlsx')
pv_calculator.calculate()
#%% 
# MODULE 3 03_geopandas_facade_analyser
# Initialize the analyser (optionally provide base_dir or file_path)
analyser = facade_analyser.FacadeAnalyser()

# Load polygons
analyser.load_polygons()
# (Optional) Merge height data if available
#height_data_path = os.path.join(analyser.base_dir, 'height.geojson')
height_data_path = os.path.join(os.getcwd(), "vector","buildings_footprint", "height.geojson")
if os.path.exists(height_data_path):
    gdf_height = gpd.read_file(height_data_path).round(4)
    analyser.polygons_gdf = analyser.polygons_gdf.merge(
        gdf_height[['build_id', 'h_mean', 'h_stdev', 'h_min', 'h_max']],
        on='build_id', how='left'
    )
# Calculate area and perimeter
analyser.polygons_gdf['f_area'] = round(analyser.polygons_gdf['geometry'].area, 4)
analyser.polygons_gdf['f_perimeter'] = round(analyser.polygons_gdf['geometry'].length, 4)
analyser.polygons_gdf['n_floorsEstim'] = (analyser.polygons_gdf['h_mean'] / 3.0).round(0)
analyser.polygons_gdf['h_estim'] = analyser.polygons_gdf['n_floorsEstim'] * 3.0 + 1


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


#%% 
# MUDULE 4.1 04_ener_consum_profile_assigner
assigner = ener_cons_profile_assigner.EnergyConsumptionProfileAssigner()
assigner.process()
#%%
# MUDULE 4.2 04_energy_profile_aggregation
PATH = r"C:\\Users\\Oleksandr-MSI\\Documentos\\GitHub\\spacer-hb-framework\\LoadProGen\\Bilbao"
# Instantiate and use the class
#pv_pct_list = [0.25, 0.5, 0.75, 1]
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
aggregator = profile_aggregation.EnergyProfileAggregator(PATH, profile_mapping, pv_pct=1)
#aggregator.plot_profiles()
result_pv_df, result_no_pv_df = aggregator.save_results()
#%% 
# MUDULE 4.3 04_self_consump_estimation
pv_pct_list = [0.25, 0.5,0.75,1]
for pv_pct in pv_pct_list:
    estimator = self_cons_estimator.SelfConsumptionEstimator(
        data_struct_file='python/pv_energy.yml',
        district="otxarkoaga",
        pv_pct=pv_pct
    )
    #estimator.time_alignment()
    estimator.run_self_consumption()
    monthly_agg, monthly_agg_no_pv = estimator.save_aggregated_profiles()
    pv_census_aggreg_df = estimator.save_pv_profiles()
    estimator.save_net_balance(monthly_agg, pv_census_aggreg_df)

#%% 
# MODULE 5.0 05_inspire_db_assigner
heating_db_assigner = inspire_db_assigner.InspireDBAssigner()
heating_db_assigner.load_data()
heating_db_assigner.recalculate_surface_area()
heating_db_assigner.assign_window_areas(windows_to_wall_ratio=0.24)
heating_db_assigner.process()
heating_db_assigner.save()

# Load layers
buildings = gpd.read_file('C:/Users/Oleksandr-MSI/Documentos/GitHub/spacer-hb-framework/data/05_buildings_with_energy_and_co2_values.geojson')
h_dem = gpd.read_file('C:/Users/Oleksandr-MSI/Documentos/GitHub/spacer-hb-framework/vector/hm_raster_25830.shp')

# Perform spatial join (predicate = intersects or within)
joined = gpd.sjoin(buildings, h_dem[['HDemProj', 'geometry']], how='left', predicate='intersects')

# Drop duplicates by index to mimic METHOD=2 (take first match only)
joined = joined[~joined.index.duplicated(keep='first')]

# Export to GeoJSON
joined.to_file('C:/Users/Oleksandr-MSI/Documentos/GitHub/spacer-hb-framework/data/05_buildings_with_energy_and_co2_values+HDemProj.geojson', driver='GeoJSON')


#%% 
# MODULE 5.1 mod_05_1_simple_kpi_calc.py

root = r"C:\Users\Oleksandr-MSI\Documentos\GitHub\spacer-hb-framework"
pv_file_name = "01_footprint_s_area_wb_rooftop_analysis_pv_month_pv.xlsx"
cost_file_name = "241211_econom_data.xlsx"

analyzer = kpi_calc.EconomicKPIAnalyzer(root, pv_file_name, cost_file_name)
df_pv_filt = analyzer.prepare_facades()
analyzer.plot_pv_generation()
#%%
analyzer.plot_census_stackplot()

SCENARIO = ["S1","S2"]
REGION = 'Pais_Vasco'
COMBINATION_ID = [1,2]#2
intervention_combinations = {
    1: ["facade", "roof"],
    2: ["facade", "windows", "roof"],
}
heating_energy_price_euro_per_kWh = 0.243
energy_price_growth_rate = 0
pv_degradation_rate = 0
ESTIM_AVG_DWE_SIZE =55 # Average dwelling size in m2, used for heating demand calculation

for scenario in SCENARIO:
    for combination_id in COMBINATION_ID:
        analyzer.calculate_costs(scenario)
        analyzer.calculate_heating_demand(REGION, scenario, combination_id, ESTIM_AVG_DWE_SIZE, intervention_combinations, heating_energy_price_euro_per_kWh)

analyzer.save_facades("data/05_buildings_with_energy_and_co2_values+HDemProj_facade_costs_with_HDem_corr.geojson")

analyzer.calculate_pv_costs()
analyzer.save_pv_to_excel()
analyzer.join_pv_costs_to_facades()
analyzer.save_facades_with_pv("data/05_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV.geojson")
analyzer.calculate_pv_metrics(pv_degradation_rate, energy_price_growth_rate)
analyzer.save_economic_analysis(
    "data/05_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV_economic.geojson",
    "data/05_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV_economic_otxarkpaga_orginal_price_incr+pv_degrad.xlsx"
)

#%% 
# MODULE 5.2 mod_05_2_pv_power_hourly.py
root = os.getcwd()  # or specify your root directory
analysis = economic_analysis.EconomicAnalysis(root)
analysis.run_analysis(CAL_REFERENCE_CASE=True)
for scenario in SCENARIO:
    for combination_id in COMBINATION_ID:
        analysis.run_analysis(SCENARIO=scenario, COMBINATION_ID=combination_id)

analysis.save_to_excel("06_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV_economic+EUAC_ORIGIN")
analysis.plot_neighbourhood("output_filename",euac_col="Envelope_EUAC")
analysis.plot_neighbourhood("output_filename",euac_col="EUAC")

#%% 
# MODULE 07 mod_07_self-cons_scenarios_calc
self_cons_analysis = self_cons_scenarios_calc.SelfConsumptionAnalysis(root)
self_cons_analysis.calculate_self_consumption()
self_cons_analysis.calculate_per_dwelling()
CENSUS_ID = 4802003011
df_self_cons_calc_per_dwelling_filter, df_self_cons_pct_filer = self_cons_analysis.filter_by_census_id(CENSUS_ID)
self_cons_analysis.plot_self_consumption_trends(
    df_self_cons_calc_per_dwelling_filter,
    df_self_cons_pct_filer,
    xlabel='Months',
    ylabel='Total Self-Consumed Electricity (kWh)',
    header=f'Monthly Average PV Electricity Self-Consumption in {CENSUS_ID} section per dwelling Percentage by Scenario',
    y_min=0, y_max=200
)
self_cons_analysis.load_consumption_data()
self_cons_analysis.filter_per_dwelling_percentage()
self_cons_analysis.create_heatmap()
self_cons_analysis.calculate_self_sufficiency()
#%%
# MODULE 07 mod_07_geo_visualization
census_file = os.path.join(os.getcwd(), "vector", "stat_census", "Otxarkoaga.shp")
data_file = os.path.join(os.getcwd(), "data", "06_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV_economic+EUAC_ORIGIN_ADD_SHEET.xlsx")

# Pass the desired filter value as an argument
visualizer = geo_visualization.GeoHeatmapVisualizer(census_file, data_file, filter_value='Comb0ref_case', sheet_name="Sheet0")
visualizer.plot_energy_co2_heatmaps()
visualizer.geo_heatmap(column_1='NRPE_Envelope_kWh_per_m2', column_2='Envelope_EUAC_per_m2', vmin1=0, vmax1=200, vmin2=0, vmax2=50)
visualizer.geo_heatmap_no_colorbar(column_1='NRPE_kWh_per_dwelling', column_2='EUAC_per_dwelling')


#%%
#HOURLY\
from python.pv_power_hourly import PVHourlyCalculator

# Example usage
params_file = '241211_econom_data.xlsx'  # Replace with your actual file path
calculator = PVHourlyCalculator(params_file)
calculator.calculate_hourly_roofs()


# %%
""" 
processing.run("wbt:LidarRooftopAnalysis", {'input':'C:\\Users\\Oleksandr-MSI\\Documentos\\GitHub\\spacer-hb-framework\\lidar\\otxarkoaga_lidar_cliped.las','buildings':'C:/Users/Oleksandr-MSI/Documentos/GitHub/spacer-hb-framework/vector/buildings_footprint/etrs_25830/buildings_inspire_clip_oxarkoaga+census.shp','radius':2,'num_iter':50,'num_samples':10,'threshold':0.15,'model_size':15,'max_slope':65,'norm_diff':10,'azimuth':180,'altitude':30,'output':'TEMPORARY_OUTPUT'})

def lidar_rooftop_analysis(self, lidar_inputs: List[Lidar], building_footprints: Vector, search_radius: float = 2.0, num_iterations: int = 50, num_samples: int = 10, inlier_threshold: float = 0.15, acceptable_model_size: int = 30, max_planar_slope: float = 75.0, norm_diff_threshold: float = 2.0, azimuth: float = 180.0, altitude: float = 30.0)
"""

from whitebox_workflows import WbEnvironment, Lidar, Vector
wbe = WbEnvironment()

# Define file paths
lidar_file = r'C:\Users\Oleksandr-MSI\Documentos\GitHub\spacer-hb-framework\lidar\otxarkoaga_lidar_cliped.las'
building_footprints_file = r'C:\Users\Oleksandr-MSI\Documentos\GitHub\spacer-hb-framework\vector\buildings_footprint\etrs_25830\buildings_inspire_clip_oxarkoaga+census.shp'

# Open the lidar and vector data properly using the environment
lidar_inputs = [wbe.read_lidar(lidar_file)]
building_footprints = wbe.read_vector(building_footprints_file)

# Run the tool
output_vector = wbe.lidar_rooftop_analysis(
    lidar_inputs=lidar_inputs,
    building_footprints=building_footprints,
    search_radius=2.0,
    num_iterations=50,
    num_samples=10,
    inlier_threshold=0.15,
    acceptable_model_size=15,
    max_planar_slope=65.0,
    norm_diff_threshold=10.0,
    azimuth=180.0,
    altitude=30.0
)

# Save the output to a new shapefile
wbe.write_vector(output_vector,r'C:\Users\Oleksandr-MSI\Documentos\GitHub\spacer-hb-framework\data\rooftop_analysis_result.shp')