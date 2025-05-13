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
            self.PATH = df.loc['PATH', 'File']
            self.LIDAR_FILE = f"{df.loc['LIDAR_FILE', 'File']}"
            self.BUILDING_FOOTPRINT_FILE = f"{df.loc['BUILDING_FOOTPRINT_FILE','File']}"
            self.STATISTICAL_CENSUS_FILE = f"{df.loc['STATISTICAL_CENSUS_FILE','File']}"
            self.WHITEBOX_RT_ANALYSIS_FILE = f"{df.loc['WHITEBOX_RT_ANALYSIS_FILE','File']}"

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
# Example usage
default_paths = DefaultPaths()
print(default_paths.get_path("data_dir"))  # Prints the data directory path

module_name = "00_set_up_directories.set_up_directories"
mk_dir = importlib.import_module(module_name)

# Define the base directory and create the directory structure
base_dir = "your_custom_name"  # Replace "your_custom_name" with any directory name you want

create_directories= mk_dir.CustomDirectoryStructure(base_dir)
create_directories.create_directories()
"""
#%%
import importlib
import os
import geopandas as gpd
rooftop_analysis = importlib.import_module("01_pivot_rooftop_data")
rt = rooftop_analysis.PivotRooftopAnalysis(file_dir=os.getcwd())
#%%

df_segments_wb_rooftop_analysis, gdf_building_footprint = rt.process_DataFrames()
#%%
df_segments_wb_rooftop_analysis, building_footprint = rt.pivot_whitebox_rooftop_analysis(df_segment=df_segments_wb_rooftop_analysis, df_buildings_footprint=gdf_building_footprint, col_name="s_area", key_ids=["build_id", "census_id"])

# %%
