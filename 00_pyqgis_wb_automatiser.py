from qgis.core import (
    QgsVectorLayer, QgsVectorFileWriter, QgsVectorLayerJoinInfo,
    QgsCoordinateReferenceSystem, QgsCoordinateTransformContext,
    QgsProcessingFeatureSourceDefinition, QgsFeatureRequest
)
from qgis import processing
import os
import geopandas as gpd



class RooftopAnalysisAutomatiser:
    def __init__(self, path, root_dir, case_study_name, suffix, crs, census_id, building_ids, filter_expression=''):
        self.path = path
        self.root_dir = root_dir
        self.case_study_name = case_study_name + suffix if suffix else case_study_name
        self.crs = crs
        self.crs_str = crs.replace("epsg:", "crs")
        self.census_id = census_id
        self.building_ids = building_ids
        self.filter_expression = filter_expression

        self.i_folder = self.path
        self.o_folder = os.path.join(self.path, self.root_dir, self.case_study_name)
        self.temp_files_folder = os.path.join(self.o_folder, "temp_files")
        os.makedirs(self.i_folder, exist_ok=True)
        os.makedirs(self.o_folder, exist_ok=True)
        os.makedirs(self.temp_files_folder, exist_ok=True)

        # Paths
        self.building_footprint_path = os.path.join(self.i_folder, "vector\\buildings_footprint\\etrs_25830\\buildings_inspire_clip_oxarkoaga+census.shp")
        self.statistical_census_path = os.path.join(self.i_folder, "vector\\stat_census\\Otxarkoaga.shp")
        self.rooftop_analysis_path = os.path.join(self.i_folder, f"vector\\whitebox_tool\\wb_rt_analysis_{case_study_name}_segments.shp")
        self.lidar_path = os.path.join(self.i_folder, f"lidar\\otxarkoaga_lidar_cliped.las")

        self.output_join_1_path = os.path.join(self.temp_files_folder, "join_1_buildings_census.shp")
        self.output_corrected_geom_path = os.path.join(self.temp_files_folder, "corrected_geom_.shp")
        self.output_join_2_path = os.path.join(self.temp_files_folder, "join_2_rooftop_mean_coords.shp")
        self.output_join_3_path = os.path.join(self.temp_files_folder, "join_3_rooftop_add_xy.geojson")
        self.output_join_4_path = os.path.join(self.o_folder, f"00_wb_rt_analysis_{self.case_study_name}_segments_xy_coord.geojson")
        self.final_output_file = f"00_wb_rt_analysis_{self.case_study_name}_segments.shp"
        self.final_output_path = os.path.join(self.o_folder, self.final_output_file)

    def run_lidar_analysis(self):
        processing.run("wbt:LidarRooftopAnalysis", {
            'input': self.lidar_path,
            'buildings': self.building_footprint_path,
            'radius': 2,
            'num_iter': 50,
            'num_samples': 10,
            'threshold': 0.15,
            'model_size': 15,
            'max_slope': 65,
            'norm_diff': 10,
            'azimuth': 180,
            'altitude': 30,
            'output': self.rooftop_analysis_path
        })

    def join_building_with_census(self):
        processing.run("native:joinattributesbylocation", {
            'INPUT': self.building_footprint_path,
            'JOIN': self.statistical_census_path,
            'PREDICATE': [0],
            'JOIN_FIELDS': [self.census_id],
            'METHOD': 2,
            'DISCARD_NONMATCHING': False,
            'PREFIX': '',
            'OUTPUT': self.output_join_1_path
        })

    def fix_rooftop_geometries(self):
        layer = QgsVectorLayer(self.rooftop_analysis_path, "RooftopAnalysis", "ogr")
        if layer.isValid():
            processing.run("native:fixgeometries", {
                'INPUT': layer,
                'METHOD': 1,
                'OUTPUT': self.output_corrected_geom_path
            })

    def calculate_surface_points_and_xy(self):
        processing.run("native:pointonsurface", {
            'INPUT': self.output_corrected_geom_path,
            'ALL_PARTS': True,
            'OUTPUT': self.output_join_2_path
        })

        processing.run("native:addxyfields", {
            'INPUT': self.output_join_2_path,
            'CRS': QgsCoordinateReferenceSystem(self.crs_str),
            'PREFIX': f'{self.crs_str}_',
            'OUTPUT': self.output_join_3_path
        })

    def fallback_mean_coords_for_null_xy(self):
        """
        For features with NULL XY in the surface output, run meancoordinates and patch values.
        Saves results to a new shapefile to avoid overwrite issues.
        """
        x_field = f"{self.crs_str}_x"
        y_field = f"{self.crs_str}_y"
        input_path = self.final_output_path  # self.output_join_3_path
        patch_prefix = "mean_"

        # Load layer and select features with NULL x_field or y_field
        xy_layer = QgsVectorLayer(input_path, "XY_Layer", "ogr")
        expr = f'"{x_field}" IS NULL OR "{y_field}" IS NULL'
        xy_layer.selectByExpression(expr)
        count_selected = xy_layer.selectedFeatureCount()
        
        if count_selected==0:
            print("No NULL XY values found. Fallback not needed.")
            self.patched_xy_output_path = input_path  # proceed with unmodified file
            return

        print(f"Selected {count_selected} features with NULL XY")

        # Save selected features to a new shapefile
        selected_features = xy_layer.selectedFeatures()
        if selected_features:
            fields = xy_layer.fields()
            crs = xy_layer.crs()
            output_path = os.path.join(self.temp_files_folder, "null_xy_features.shp")
            writer = QgsVectorFileWriter(
            output_path,
            "UTF-8",
            fields,
            xy_layer.wkbType(),
            crs,
            "ESRI Shapefile"
            )
            for feat in selected_features:
                writer.addFeature(feat)
            del writer  # Finalize writing
            print(f"Saved {len(selected_features)} features with NULL XY to {output_path}")
        else:
            print("No features with NULL XY found.")

        patched_path = os.path.join(self.temp_files_folder, "joined_surface_xy_patched.shp")

        result = processing.run("native:meancoordinates", {
            'INPUT': QgsProcessingFeatureSourceDefinition(
            output_path,
            featureLimit=-1,
            geometryCheck=QgsFeatureRequest.GeometryAbortOnInvalid
            ),
            'UID': 'FID',
            'OUTPUT': patched_path
        })
        if result and 'OUTPUT' in result:
            print(f"Patched XY coordinates saved to {result['OUTPUT']}")
        else:
            print("Failed to save patched XY coordinates.")

    def fill_null_xy_with_mean_coords(self):
        """
        Fill NULL XY values with mean coordinates from the patched shapefile.
        """
        xy_layer = QgsVectorLayer(self.patched_xy_output_path, "XY_Layer", "ogr")
        x_field = f"{self.crs_str}_x"
        y_field = f"{self.crs_str}_y"
        mean_x_field = f"{self.crs_str}_{self.patch_prefix}x"
        mean_y_field = f"{self.crs_str}_{self.patch_prefix}y"

        for feature in xy_layer.getFeatures():
            if feature[x_field] is None or feature[y_field] is None:
                feature[x_field] = feature[mean_x_field]
                feature[y_field] = feature[mean_y_field]
                xy_layer.updateFeature(feature)

        # Save the updated layer
        writer = QgsVectorFileWriter.writeAsVectorFormat(
            xy_layer, self.final_output_path, "UTF-8",
            xy_layer.crs(), "ESRI Shapefile"
        )
        if writer[0] != QgsVectorFileWriter.NoError:
            print("Error when saving shapefile:", writer)
        else:
            print("Shapefile saved successfully:", self.final_output_path)

    def map_and_fill_null_xy(self):
        """
        Fill NULL x/y in final output with MEAN_X/MEAN_Y from patched file by FID using geopandas.
        """
        x_field = f"{self.crs_str}_x"
        y_field = f"{self.crs_str}_y"
        final_path = self.final_output_path
        patched_path = os.path.join(self.temp_files_folder, "joined_surface_xy_patched.shp")
        output_path = self.final_output_path #os.path.join(self.o_folder, "final_layer_with_filled_xy.shp")

        final_gdf = gpd.read_file(final_path)
        patched_gdf = gpd.read_file(patched_path)[["FID", "MEAN_X", "MEAN_Y"]]

        # Merge on FID, suffixes to distinguish columns
        merged = final_gdf.merge(patched_gdf, on="FID", how="left", suffixes=("", "_mean"))

        # Fill nulls in x/y with MEAN_X/MEAN_Y
        merged[x_field] = merged[x_field].fillna(merged["MEAN_X"])
        merged[y_field] = merged[y_field].fillna(merged["MEAN_Y"])

        # Drop MEAN_X/MEAN_Y columns before saving
        merged = merged.drop(columns=["MEAN_X", "MEAN_Y"])

        merged.to_file(output_path)
        print("Shapefile with filled XY overwritten:", output_path)
        # Save as CSV without geometry
        merged = merged.drop(columns=["geometry"])
        csv_output_path = os.path.join(self.o_folder, f"00_wb_rt_analysis_{self.case_study_name}_segments.csv")
        merged.to_csv(csv_output_path, index=False)
        print("CSV with filled XY saved:", csv_output_path)


    def join_surface_with_buildings(self):
        joined_fields = [f'{self.crs_str}_x', f'{self.crs_str}_y'] + self.building_ids + [self.census_id]
        surface_layer = QgsVectorLayer(self.output_join_3_path, "RooftopMeanCoords", "ogr")
        building_layer_joined = QgsVectorLayer(self.output_join_1_path, "BuildingsJoinCensus", "ogr")

        # Apply uniqueness constraint (optional, as per your original script)
        unique_fid_expression = 'array_length(array_distinct(array_agg("FID"))) = 1'
        surface_layer.setSubsetString(unique_fid_expression)

        processing.run("native:joinattributesbylocation", {
            'INPUT': surface_layer,
            'JOIN': building_layer_joined,
            'PREDICATE': [5],
            'JOIN_FIELDS': joined_fields,
            'METHOD': 2,
            'DISCARD_NONMATCHING': False,
            'PREFIX': '',
            'OUTPUT': self.output_join_4_path
        })

    def final_fid_join_and_save(self):
        joined_fields = [f'{self.crs_str}_x', f'{self.crs_str}_y'] + self.building_ids + [self.census_id]
        final_output_layer = QgsVectorLayer(self.output_join_3_path, "FinalOutput", "ogr")
        rooftop_layer = QgsVectorLayer(self.rooftop_analysis_path, "RooftopAnalysis", "ogr")

        join_info = QgsVectorLayerJoinInfo()
        join_info.setJoinFieldName("FID")
        join_info.setTargetFieldName("FID")
        join_info.setJoinLayerId(final_output_layer.id())
        join_info.setUsingMemoryCache(True)
        join_info.setJoinLayer(final_output_layer)
        join_info.setPrefix('')
        join_info.setJoinFieldNamesSubset(joined_fields)
        rooftop_layer.addJoin(join_info)

        writer = QgsVectorFileWriter.writeAsVectorFormat(
            rooftop_layer, self.final_output_path, "UTF-8",
            final_output_layer.crs(), "ESRI Shapefile"
        )
        if writer[0] != QgsVectorFileWriter.NoError:
            print("Error when saving shapefile:", writer)
        else:
            print("Shapefile saved successfully:", self.final_output_path)
        """
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "CSV"
        options.layerOptions = ['GEOMETRY=AS_XY']
        writer = QgsVectorFileWriter.writeAsVectorFormatV2(
            rooftop_layer,
            self.final_output_path.replace(".shp", ".csv"),
            QgsCoordinateTransformContext(), options
        )
        if writer[0] != QgsVectorFileWriter.NoError:
            print("Error when saving CSV:", writer)
        else:
            print("CSV saved successfully:", self.final_output_path.replace(".shp", ".csv"))
        """
    def run(self):
        self.run_lidar_analysis()
        self.join_building_with_census()
        self.fix_rooftop_geometries()
        self.calculate_surface_points_and_xy()
        self.join_surface_with_buildings()
        self.final_fid_join_and_save()
        self.fallback_mean_coords_for_null_xy()
        self.map_and_fill_null_xy()


# Example usage
if __name__ == "__main__":
    automator = RooftopAnalysisAutomatiser(
        path="C:\\Users\\Oleksandr-MSI\\Documentos\\GitHub\\spacer-hb-framework",
        root_dir="pyqgis_wb_automatiser",
        case_study_name="bilbao_otxarkoaga",
        suffix="_v2",
        crs="epsg:25830",
        census_id="census_id",
        building_ids=["build_id"],#, "osm_id"],
        filter_expression=''  # e.g. '"AREA" > 2'
    )
    automator.run()