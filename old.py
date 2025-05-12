from qgis.core import (QgsVectorLayer, QgsVectorFileWriter, QgsVectorLayerJoinInfo, QgsApplication, QgsCoordinateReferenceSystem, QgsCoordinateTransformContext)
from qgis import processing
import os
import inspect
#%%
#LIST a QGIS unique command–line name
"""
for alg in QgsApplication.processingRegistry().algorithms():
        print(alg.id(), "->", alg.displayName())
"""

# SET INPUT/OUTPUT PATH and FILES NAMES
PATH = "C:\\Users\\Oleksandr-MSI\\Documentos\\GitHub\\spacer-hb-framework"
root_dir = "pyqgis_wb_automatiser"
suffix = "_v2"#"wb_rt_analysis_"
case_study_name = "bilbao_otxarkoaga"
if suffix and suffix != "":
    case_study_name = "bilbao_otxarkoaga" + suffix
census_id = "census_id"
building_ids = ['build_id', 'osm_id'] #'CodEdifici'
# Coordinate Reference System
crs = "epsg:25830"
#short name for the crs
crs_str = crs.replace("epsg:", "crs")

#Filter polygons by selection ones with area more than 2 m2
filter_expression = ''#'"AREA" > 2'

# Create Folders for input/output files if they do not exist
i_folder = PATH
o_folder = os.path.join(PATH, root_dir, case_study_name)
os.makedirs(i_folder, exist_ok=True)
os.makedirs(o_folder, exist_ok=True)

# SET PATHS for input layers
building_footprint_path = os.path.join(i_folder, "vector\\buildings_footprint\\etrs_25830\\buildings_inspire_clip_oxarkoaga+census.shp")#inspire_db_Lasarte_clip.shp")
statistical_census_path = os.path.join(i_folder, "vector\\stat_census\\Otxarkoaga.shp")
rooftop_analysis_path = os.path.join(i_folder, f"vector\\whitebox_tool\\wb_rt_analysis_bilbao_otxarkoaga_segments.shp")
lidar_path = os.path.join(i_folder, f"lidar\\otxarkoaga_lidar_cliped.las")

processing.run("wbt:LidarRooftopAnalysis", {
    'input': lidar_path,
    'buildings': building_footprint_path,
    'radius': 2,
    'num_iter': 50,
    'num_samples': 10,
    'threshold': 0.15,
    'model_size': 15,
    'max_slope': 65,
    'norm_diff': 10,
    'azimuth': 180,
    'altitude': 30,
    'output': rooftop_analysis_path
})

# Directory for working files and (temporary files)
if not os.path.exists(o_folder):
    os.makedirs(o_folder)
    print("Output folder created: ", o_folder)
temp_files_folder = os.path.join(o_folder, "temp_files")
os.makedirs(temp_files_folder, exist_ok=True)
print("Temporary files folder created: ", temp_files_folder)

# Temporary Files output paths
output_join_1_path = os.path.join(temp_files_folder, "join_1_buildings_census.shp")
output_corrected_geom_path = os.path.join(temp_files_folder, "corrected_geom_.shp")
output_join_2_path = os.path.join(temp_files_folder, "join_2_rooftop_mean_coords.shp")
output_join_2_1_path = os.path.join(temp_files_folder, "join_2_rooftop_add_xy.geojson")

# FINAL OUTPUT FILE
final_output_file = f"00_wb_rt_analysis_{case_study_name}_segments.shp"#.geojson"
final_output_path = os.path.join(o_folder, final_output_file)
output_join_3_path = os.path.join(o_folder, f"00_wb_rt_analysis_{case_study_name}_segments_xy_coord.geojson")


# Load input layers
building_footprint_layer = QgsVectorLayer(building_footprint_path, "Building_footprint", "ogr")
statistical_census_layer = QgsVectorLayer(statistical_census_path, "Statistical_census", "ogr")
rooftop_analysis_layer = QgsVectorLayer(rooftop_analysis_path, "RooftopAnalysis", "ogr")

# Apply filter
if filter_expression and filter_expression != "":
    rooftop_analysis_layer.setSubsetString(filter_expression)
    # Check if the filter is applied
    if rooftop_analysis_layer.subsetString() == filter_expression:
        print(f"Filter applied: Only polygons with {filter_expression} m² are selected.")
    else:
        print("Failed to apply filter.")

# Step 1: Join Attributes by Location
processing.run("native:joinattributesbylocation", {
    'INPUT': building_footprint_path,
    'JOIN': statistical_census_path,
    'PREDICATE': [0],
    'JOIN_FIELDS': [census_id],
    'METHOD': 2,
    'DISCARD_NONMATCHING': False,
    'PREFIX': '',
    'OUTPUT': output_join_1_path
})

# Check if layer has invalid geometries
if rooftop_analysis_layer.isValid():
    processing.run("native:fixgeometries", {
        'INPUT': rooftop_analysis_layer,
        'METHOD':1,
        'OUTPUT':output_corrected_geom_path})

# Step 2: Join Building_footprint.shp with the output of step 1
building_footprint_layer_joined = QgsVectorLayer(output_join_1_path, "Building_footprint_joined", "ogr")
join_info_1 = QgsVectorLayerJoinInfo()
join_info_1.setJoinFieldName(building_ids[1])
join_info_1.setTargetFieldName(building_ids[1])
join_info_1.setUsingMemoryCache(True)
join_info_1.setJoinLayerId(building_footprint_layer_joined.id())
join_info_1.setJoinLayer(building_footprint_layer_joined)
building_footprint_layer_joined.addJoin(join_info_1)

# Step 3: Point of surface coordinates for RooftopAnalysis layer
"""processing.run("native:meancoordinates", {
    'INPUT': rooftop_analysis_layer,
    'UID': 'FID',
    'OUTPUT': output_join_2_path
})"""
if rooftop_analysis_layer.isValid():
    processing.run("native:pointonsurface", {
        'INPUT': output_corrected_geom_path,
        'ALL_PARTS':True,
        'OUTPUT': output_join_2_path
    })
else:
    processing.run("native:meancoordinates", {
    'INPUT': rooftop_analysis_layer,
    'UID': 'FID',
    'OUTPUT': output_join_2_path
    })

rooftop_analysis_to_x_y_layer = QgsVectorLayer(output_join_2_path, "RooftopAnalysis_joined", "ogr")

processing.run("native:addxyfields", {
    'INPUT':rooftop_analysis_to_x_y_layer,
    'CRS':QgsCoordinateReferenceSystem(crs_str),
    'PREFIX':f'{crs_str}_',
    'OUTPUT':output_join_2_1_path
})


# Step 4: Join Attributes by Location between output of step 3 and step 2 output file
joined_fields =[f'{crs_str}_x',f'{crs_str}_y'] + building_ids + [census_id]
rooftop_analysis_mean_coord_layer = QgsVectorLayer(output_join_2_1_path, "RooftopAnalysis_joined", "ogr")

# Filter to select only unique values from "FID" column
unique_fid_expression = 'array_length(array_distinct(array_agg("FID"))) = 1'
rooftop_analysis_mean_coord_layer.setSubsetString(unique_fid_expression)

# joined_fields = [f for f in joined_fields if f != "FID"]
processing.run("native:joinattributesbylocation", {
    'INPUT': rooftop_analysis_mean_coord_layer,
    'JOIN': building_footprint_layer_joined,
    'PREDICATE': [5],
    'JOIN_FIELDS': joined_fields,
    'METHOD': 2,
    'DISCARD_NONMATCHING': False,
    'PREFIX': '',
    'OUTPUT': output_join_3_path#final_output_path #
})
print ('Geojon file created with x and y coordinates: ', output_join_3_path)#final_output_path) #

# Step 5: Join the resulting layer with RooftopAnalysis layer by FID attribute
final_output_layer = QgsVectorLayer(output_join_3_path, "Final_output", "ogr")
join_info_2 = QgsVectorLayerJoinInfo()
join_info_2.setJoinFieldName("FID")
join_info_2.setTargetFieldName("FID")
join_info_2.setJoinLayerId(final_output_layer.id())
join_info_2.setUsingMemoryCache(True)
join_info_2.setJoinLayer(final_output_layer)
join_info_2.setPrefix('')
join_info_2.setJoinFieldNamesSubset(joined_fields)
rooftop_analysis_layer.addJoin(join_info_2)

# Save resulting shapefile
writer = QgsVectorFileWriter.writeAsVectorFormat(rooftop_analysis_layer, final_output_path, "UTF-8", final_output_layer.crs(), "ESRI Shapefile")
if writer[0] != QgsVectorFileWriter.NoError:
    print("Error when saving layer:", writer)
else:
    print("Layer saved successfully! ", writer)

# Save resulting CSV file without multipolygons
options = QgsVectorFileWriter.SaveVectorOptions()
options.driverName = "CSV"
options.layerOptions = ['GEOMETRY=AS_XY']
writer = QgsVectorFileWriter.writeAsVectorFormatV2(rooftop_analysis_layer, final_output_path.replace(".shp", ".csv"), QgsCoordinateTransformContext(), options)
if writer[0] != QgsVectorFileWriter.NoError:
    print("Error when saving layer:", writer)
else:
    print("Layer saved successfully as CSV! ", writer)
#%%
