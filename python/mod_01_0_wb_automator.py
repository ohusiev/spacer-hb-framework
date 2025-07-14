#%%
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pathlib import Path
from whitebox_workflows import WbEnvironment



class RooftopAnalysisAutomatiserPython:
    def __init__(self, path, root_dir, case_study_name, suffix, crs, filter_area, census_id, building_ids,building_footprint_path,statistical_census_path, lidar_path):
        self.path = Path(path)
        self.root_dir = root_dir
        self.case_study_name = case_study_name + suffix if suffix else case_study_name
        self.crs = crs
        self.census_id = census_id
        self.building_ids = building_ids

        self.i_folder = self.path
        self.o_folder = self.path / self.root_dir / self.case_study_name
        self.temp_files_folder = self.o_folder / "temp_files"
        self.i_folder.mkdir(parents=True, exist_ok=True)
        self.o_folder.mkdir(parents=True, exist_ok=True)
        self.temp_files_folder.mkdir(parents=True, exist_ok=True)
        self.filter_area = filter_area

        # Input paths
        self.lidar_path = lidar_path #os.path.join(self.i_folder, "lidar", "otxarkoaga_lidar_cliped.las")
        self.building_footprint_path = building_footprint_path#os.path.join(self.i_folder, "vector", "buildings_footprint", "etrs_25830", "buildings_inspire_clip_oxarkoaga+census.shp")
        self.statistical_census_path = statistical_census_path #os.path.join(self.i_folder, "vector", "stat_census", "Otxarkoaga.shp")
        #self.rooftop_analysis_path = self.i_folder / f"vector/whitebox_tool/wb_rt_analysis_{self.case_study_name}_segments.shp"

        # Output paths
        self.rooftop_output_path = os.path.join(self.o_folder, f"wb_rt_analysis_{self.case_study_name}_segments.shp")
        self.output_building_census_join = self.temp_files_folder / "joined_building_census.shp"
        self.output_rooftop_clean = self.temp_files_folder / "rooftop_clean.shp"
        self.output_rooftop_xy = self.temp_files_folder / "rooftop_with_xy.shp"
        self.output_final_geojson = self.o_folder / f"00_wb_rt_analysis_{self.case_study_name}_segments_xy_coord.geojson"
        self.output_final_shp = self.o_folder / f"00_wb_rt_analysis_{self.case_study_name}_segments_xy_coord.shp"
        self.output_final_csv = self.o_folder / f"00_wb_rt_analysis_{self.case_study_name}_segments.csv"
    
    def run_lidar_analysis(self):
        wbe = WbEnvironment()

        # Define file paths
        #lidar_file = r'C:\Users\Oleksandr-MSI\Documentos\GitHub\spacer-hb-framework\lidar\otxarkoaga_lidar_cliped.las'
        #building_footprints_file = r'C:\Users\Oleksandr-MSI\Documentos\GitHub\spacer-hb-framework\vector\buildings_footprint\etrs_25830\buildings_inspire_clip_oxarkoaga+census.shp'

        # Open the lidar and vector data properly using the environment
        lidar_inputs = [wbe.read_lidar(self.lidar_path)]
        building_footprints = wbe.read_vector(self.building_footprint_path)

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
        wbe.write_vector(output_vector, self.rooftop_output_path)
        print(f"Rooftop analysis completed and saved to {self.rooftop_output_path}")

    def join_buildings_with_census(self):
        buildings = gpd.read_file(self.building_footprint_path)
        census = gpd.read_file(self.statistical_census_path)[[self.census_id, 'geometry']]
        joined = gpd.sjoin(buildings, census, how='left', predicate='intersects')
        joined.to_file(self.output_building_census_join)

    def clean_rooftop_segments(self):
        gdf = gpd.read_file(self.rooftop_output_path)
        gdf_clean = gdf[gdf.is_valid]
        gdf_clean.to_file(self.output_rooftop_clean)

    def add_xy_fields(self):
        gdf = gpd.read_file(self.rooftop_output_path)#(self.output_rooftop_clean)
        #filter the output to only include values of column "AREA" greater than 1
        if self.filter_area is not None:
            gdf = gdf[gdf['AREA'] > self.filter_area]
            print(f"Filtered rooftops to include only those with area greater than {self.filter_area} square meters.")
        gdf["geometry"] = gdf.representative_point()
        gdf["x"] = gdf.geometry.x
        gdf["y"] = gdf.geometry.y

        # Fallback for nulls: use centroids
        nulls = gdf["x"].isna() | gdf["y"].isna()
        if nulls.any():
            centroids = gdf.geometry.centroid
            gdf.loc[nulls, "x"] = centroids.x[nulls]
            gdf.loc[nulls, "y"] = centroids.y[nulls]

        gdf.to_file(self.output_rooftop_xy)

    def join_rooftops_with_building_ids(self):
        rooftops = gpd.read_file(self.output_rooftop_xy)
        buildings = gpd.read_file(self.output_building_census_join)

        joined = gpd.sjoin(rooftops, buildings[self.building_ids + [self.census_id, 'geometry']], how='left', predicate='within')
        #add to x and y columns name corresponding name of crm f"{self.crs}_x" and f"{self.crs}_y"
        joined = joined.rename(columns={"x": f"{self.crs}_x", "y": f"{self.crs}_y"})
        joined.to_file(self.output_final_geojson, driver='GeoJSON')
        joined.to_file(self.output_final_shp)
        joined.drop(columns='geometry').to_csv(self.output_final_csv, index=False)

    def run(self):
        print("Starting rooftop analysis...")
        self.run_lidar_analysis()
        print("Joining buildings with census...")
        self.join_buildings_with_census()
        #print("Cleaning rooftop geometries...")
        #self.clean_rooftop_segments()
        print("Adding XY coordinates to rooftops...")
        self.add_xy_fields()
        print("Joining rooftops with building and census IDs...")
        self.join_rooftops_with_building_ids()
        print("All tasks completed.")

#%%
# Example usage (disabled here to avoid execution in this environment)
if __name__ == "__main__":
    path="C:/Users/Oleksandr-MSI/Documentos/GitHub/spacer-hb-framework"
    lidar_path = os.path.join(path, "lidar", "otxarkoaga_lidar_cliped.las")
    building_footprint_path = os.path.join(path, "vector", "buildings_footprint", "etrs_25830", "buildings_inspire_clip_oxarkoaga+census.shp")
    statistical_census_path = os.path.join(path, "vector", "stat_census", "Otxarkoaga.shp")
    automator = RooftopAnalysisAutomatiserPython(
        path=path,
        root_dir="pyqgis_wb_automatiser",
        case_study_name="bilbao_otxarkoaga",
        suffix="_v2",
        crs="epsg:25830",
        census_id="census_id",
        building_ids=["build_id"],
        building_footprint_path=building_footprint_path,
        statistical_census_path=statistical_census_path,
        lidar_path=lidar_path
     )
    automator.run()


# %%
