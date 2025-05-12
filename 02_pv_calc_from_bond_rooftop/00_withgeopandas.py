# %%
import geopandas as gpd
import os

# Set input/output paths
PATH = "H:\\My Drive\\01_PHD_CODE\\chapter_5\\spain\\bilbao_otxarkoaga\\"
root_dir = "pyqgis_wb_automatiser"
suffix = "_v2"
case_study_name = "bilbao_otxarkoaga"
if suffix:
    case_study_name += suffix
census_id = "census_id"
building_ids = ['build_id', 'osm_id']

# Create directories for input/output files
output_folder = os.path.join(PATH, root_dir, case_study_name)
os.makedirs(output_folder, exist_ok=True)
temp_files_folder = os.path.join(output_folder, "temp_files")
os.makedirs(temp_files_folder, exist_ok=True)

# Set paths for input layers
building_footprint_path = os.path.join(PATH, "vector\\osm_buildings\\etrs_25830\\buildings_inspire_clip_oxarkoaga+census.shp")
statistical_census_path = os.path.join(PATH, "vector\\stat_census\\Otxarkoaga.shp")
rooftop_analysis_path = os.path.join(PATH, f"vector\\whitebox_tool\\wb_rt_analysis_bilbao_otxarkoaga_segments.shp")

# Output paths
output_join_1_path = os.path.join(temp_files_folder, "join_1_buildings_census.shp")
output_join_2_path = os.path.join(temp_files_folder, "join_2_rooftop_mean_coords.shp")
output_join_3_path = os.path.join(temp_files_folder, "join_3_final_join.shp")
final_output_file = f"00_wb_rt_analysis_{case_study_name}_segments.shp"
final_output_path = os.path.join(output_folder, final_output_file)

# Load input layers
building_footprint = gpd.read_file(building_footprint_path)
statistical_census = gpd.read_file(statistical_census_path)
rooftop_analysis = gpd.read_file(rooftop_analysis_path)

# Step 1: Spatial join between buildings and census data
join_1 = gpd.sjoin(building_footprint, statistical_census, how="left", predicate="intersects")
join_1.to_file(output_join_1_path)

# Step 2: Calculate mean coordinates for rooftop analysis
rooftop_analysis['mean_x'] = rooftop_analysis.geometry.centroid.x
rooftop_analysis['mean_y'] = rooftop_analysis.geometry.centroid.y
rooftop_analysis.to_file(output_join_2_path)

# Step 3: Spatial join between mean coordinates and joined buildings
join_2 = gpd.read_file(output_join_2_path)
join_2 = join_2.drop(columns=['index_left', 'index_right'], errors='ignore')
join_1 = join_1.drop(columns=['index_left', 'index_right'], errors='ignore')
join_3 = gpd.sjoin(join_2, join_1, how="left", predicate="within")
join_3.to_file(output_join_3_path)

# Step 4: Final join between rooftop analysis and previous results
join_3 = join_3.drop(columns=['index_left', 'index_right'], errors='ignore')
final_join = gpd.sjoin(rooftop_analysis, join_3, how="left", predicate="intersects")
final_join = final_join.drop(columns=['index_left', 'index_right'], errors='ignore')
final_join.to_file(final_output_path)

print(f"Processing completed successfully! Final output saved at: {final_output_path}")
# %%
import geopandas as gpd
import os

# Set input/output paths
PATH = "H:\\My Drive\\01_PHD_CODE\\chapter_5\\spain\\bilbao_otxarkoaga\\"
root_dir = "pyqgis_wb_automatiser"
suffix = "_v2"
case_study_name = "bilbao_otxarkoaga"
if suffix:
    case_study_name += suffix

# Create directories for input/output files
output_folder = os.path.join(PATH, root_dir, case_study_name)
os.makedirs(output_folder, exist_ok=True)
temp_files_folder = os.path.join(output_folder, "temp_files")
os.makedirs(temp_files_folder, exist_ok=True)

# Set paths for input layers
building_footprint_path = os.path.join(PATH, "vector\\osm_buildings\\etrs_25830\\buildings_inspire_clip_oxarkoaga+census.shp")
statistical_census_path = os.path.join(PATH, "vector\\stat_census\\Otxarkoaga.shp")
rooftop_analysis_path = os.path.join(PATH, f"vector\\whitebox_tool\\wb_rt_analysis_bilbao_otxarkoaga_segments.shp")

# Output paths
output_join_1_path = os.path.join(temp_files_folder, "join_1_buildings_census.shp")
output_join_2_path = os.path.join(temp_files_folder, "join_2_rooftop_buildings.shp")
final_output_file = f"00_wb_rt_analysis_{case_study_name}_segments.shp"
final_output_path = os.path.join(output_folder, final_output_file)

# Load input layers
building_footprint = gpd.read_file(building_footprint_path)
statistical_census = gpd.read_file(statistical_census_path)
rooftop_analysis = gpd.read_file(rooftop_analysis_path)
# Step 0 filter out polygons with area less than 2 m2 as distorted polygons
rooftop_analysis = rooftop_analysis[rooftop_analysis['AREA'] > 2]
print("Distorted polygons with area less than 2 m2 removed")
building_footprint=building_footprint.drop(columns=['census_id'], errors='ignore')

# Step 1: Spatial join - Building footprints within statistical census
# Perform spatial join with 'within' predicate
def spatial_join_buildings_census(building_footprint, statistical_census, output_path):
    join_1_within = gpd.sjoin(building_footprint, statistical_census, how="left", predicate="within")

    # For polygons not completely within, perform spatial join with 'intersects' predicate
    join_1_intersects = gpd.sjoin(building_footprint, statistical_census, how="left", predicate="intersects")

    # Identify polygons not completely within
    not_within = join_1_within[join_1_within['index_right'].isna()]

    # For these polygons, find the largest overlap
    for idx, row in not_within.iterrows():
        building_geom = row.geometry
        overlaps = join_1_intersects[join_1_intersects.index_left == idx]
        if not overlaps.empty:
            overlaps['overlap_area'] = overlaps.geometry.intersection(building_geom).area
            largest_overlap = overlaps.loc[overlaps['overlap_area'].idxmax()]
            join_1_within.loc[idx, 'index_right'] = largest_overlap['index_right']

    # Combine results
    join_1 = join_1_within
    join_1 = join_1.drop(columns=['index_right'])  # Remove unnecessary index column
    join_1.to_file(output_path)
    return join_1

# Call the function
join_1 = spatial_join_buildings_census(building_footprint, statistical_census, output_join_1_path)

# Step 2: Spatial join - Rooftop analysis within joined buildings (including census info)
#drop all columns except ['geometry', 'census_id', 'build_id']
building_with_census = join_1[['geometry', 'census_id', 'build_id']]
 #gpd.read_file(output_join_1_path, columns=["census_id", "build_id"])
join_2 = gpd.sjoin(rooftop_analysis, building_with_census, how="left", predicate="within")
join_2 = join_2.drop(columns=['index_right'])  # Remove unnecessary index column
join_2.to_file(output_join_2_path)

# Step 3: Save the final output
#final_output = gpd.read_file(output_join_2_path)
#final_output.to_file(final_output_path)
join_2.to_file(final_output_path)
print(f"Processing completed successfully! Final output saved at: {final_output_path}")

# %%
# %% NOT WORKING
import geopandas as gpd
import os

# Set input/output paths
PATH = "H:\\My Drive\\01_PHD_CODE\\chapter_5\\spain\\bilbao_otxarkoaga\\"
root_dir = "pyqgis_wb_automatiser"
suffix = "_v2"
case_study_name = "bilbao_otxarkoaga"
if suffix:
    case_study_name += suffix

# Create directories for input/output files
output_folder = os.path.join(PATH, root_dir, case_study_name)
os.makedirs(output_folder, exist_ok=True)
temp_files_folder = os.path.join(output_folder, "temp_files")
os.makedirs(temp_files_folder, exist_ok=True)

# Set paths for input layers
building_footprint_path = os.path.join(PATH, "vector\\osm_buildings\\etrs_25830\\buildings_inspire_clip_oxarkoaga+census.shp")
statistical_census_path = os.path.join(PATH, "vector\\stat_census\\Otxarkoaga.shp")
rooftop_analysis_path = os.path.join(PATH, f"vector\\whitebox_tool\\wb_rt_analysis_bilbao_otxarkoaga_segments.shp")

# Output paths
output_join_1_path = os.path.join(temp_files_folder, "join_1_buildings_census.shp")
output_join_2_path = os.path.join(temp_files_folder, "join_2_rooftop_buildings.shp")
final_output_file = f"00_wb_rt_analysis_{case_study_name}_segments.shp"
final_output_path = os.path.join(output_folder, final_output_file)

# Load input layers
building_footprint = gpd.read_file(building_footprint_path)
statistical_census = gpd.read_file(statistical_census_path)
rooftop_analysis = gpd.read_file(rooftop_analysis_path)
# Step 0 filter out polygons with area less than 2 m2 as distorted polygons
rooftop_analysis = rooftop_analysis[rooftop_analysis['AREA'] > 2]
print("Distorted polygons with area less than 2 m2 removed")
building_footprint=building_footprint.drop(columns=['census_id'], errors='ignore')

# Step 1: Spatial join - Building footprints within statistical census
# Perform spatial join with 'within' predicate
def spatial_join_buildings_census(building_footprint, statistical_census, output_path):
    join_1_within = gpd.sjoin(building_footprint, statistical_census, how="left", predicate="within")

    # For polygons not completely within, perform spatial join with 'intersects' predicate
    join_1_intersects = gpd.sjoin(building_footprint, statistical_census, how="left", predicate="intersects")

    # Identify polygons not completely within
    not_within = join_1_within[join_1_within['index_right'].isna()]

    # For these polygons, find the largest overlap
    for idx, row in not_within.iterrows():
        building_geom = row.geometry
        overlaps = join_1_intersects[join_1_intersects.index == idx]
        if not overlaps.empty:
            overlaps['overlap_area'] = overlaps.geometry.intersection(building_geom).area
            largest_overlap = overlaps.loc[overlaps['overlap_area'].idxmax()]
            join_1_within.loc[idx, 'index_right'] = largest_overlap['index_right']

    # Combine results
    join_1 = join_1_within
    join_1 = join_1.drop(columns=['index_right'], errors='ignore')  # Remove unnecessary index column
    join_1.to_file(output_path)
    return join_1

# Call the function
join_1 = spatial_join_buildings_census(building_footprint, statistical_census, output_join_1_path)

# Step 2: Spatial join - Rooftop analysis within joined buildings (including census info)
#drop all columns except ['geometry', 'census_id', 'build_id']
building_with_census = join_1[['geometry', 'census_id', 'build_id']]
def spatial_join_buildings_roofs(building_footprint, statistical_census, output_path):
    join_1_within = gpd.sjoin(building_footprint, statistical_census, how="left", predicate="within")

    # For polygons not completely within, perform spatial join with 'intersects' predicate
    join_1_intersects = gpd.sjoin(building_footprint, statistical_census, how="left", predicate="intersects")

    # Identify polygons not completely within
    not_within = join_1_within[join_1_within['index_right'].isna()]

    # For these polygons, find the largest overlap
    for idx, row in not_within.iterrows():
        building_geom = row.geometry
        overlaps = join_1_intersects[join_1_intersects.index == idx]
        if not overlaps.empty:
            overlaps['overlap_area'] = overlaps.geometry.intersection(building_geom).area
            largest_overlap = overlaps.loc[overlaps['overlap_area'].idxmax()]
            join_1_within.loc[idx, 'index_right'] = largest_overlap['index_right']

    # Combine results
    join_1 = join_1_within
    join_1 = join_1.drop(columns=['index_right'])  # Remove unnecessary index column
    join_1.to_file(output_path)
    return join_1
join_2 = spatial_join_buildings_roofs(rooftop_analysis, building_with_census, output_join_2_path)
#join_2.to_file(output_join_2_path)
# Convert rooftop_analysis polygons to centroids (points)
rooftop_centroids = rooftop_analysis.copy()
rooftop_centroids['geometry'] = rooftop_centroids.centroid

# Perform spatial join between rooftop centroids and buildings with census info
join_2 = gpd.sjoin(rooftop_centroids, building_with_census, how="left", predicate="within")

# Merge the join results back to the original rooftop_analysis polygons
join_2 = rooftop_analysis.merge(join_2.drop(columns='geometry'), left_index=True, right_index=True)

# Step 3: Save the final output
#final_output = gpd.read_file(output_join_2_path)
#final_output.to_file(final_output_path)
join_2.to_file(final_output_path)
print(f"Processing completed successfully! Final output saved at: {final_output_path}")
