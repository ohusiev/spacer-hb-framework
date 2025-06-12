#%%
import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import MultiPolygon, Polygon
import numpy as np
from shapely.geometry import LineString

class FacadeAnalyser:
    def __init__(self, base_dir=None, file_path=None):
        if base_dir is None:
            #self.base_dir = os.path.join(os.getcwd(), "vector", "buildings_footprint")
            self.base_dir = os.path.join(os.getcwd(), "data")
        else:
            self.base_dir = base_dir
        if file_path is None:
            self.file_path = os.path.join(self.base_dir,"01_footprint_s_area_wb_rooftop_analysis.geojson")# "etrs_25830/buildings_inspire_clip_oxarkoaga+census.shp")
        else:
            self.file_path = file_path
        self.polygons_gdf = None
        self.side_angles = {
            'N': (337.5, 22.5),
            'NW': (300, 337.5), 
            'E': (60, 111), 
            'SW': (198, 249), 
            'S': (162, 198), 
            'SE': (111, 162), 
            'W': (249, 300), 
            'NE': (22.5, 60)
        }

    def load_polygons(self, build_id='build_id'):
        col_of_interest = [build_id, 'census_id', 'Codigo_Pol', 'Codigo_Par', 'building', 'Numero_Alt', 'Ano_Constr', 'Ano_Rehabi','r_area','geometry']
        self.polygons_gdf = gpd.read_file(self.file_path, index_col="ID")
        print(self.polygons_gdf.head())
        self.polygons_gdf = self.polygons_gdf[col_of_interest]

    def calculate_segment_length_and_orientation(self, start_x, start_y, end_x, end_y):
        dx = end_x - start_x
        dy = end_y - start_y
        length = np.round(np.sqrt(dx**2 + dy**2), decimals=2)
        angle = (180 / np.pi) * np.arctan2(dy, dx)
        if angle < 0:
            angle += 360
        angle = (360 - angle) % 360
        return length, angle

    def classify_orientation(self, angle):
        for side, (start, end) in self.side_angles.items():
            if start > end:
                if angle >= start or angle < end:
                    return side
            else:
                if start <= angle < end:
                    return side
        return None

    def length_per_orientation(self):
        sides = list(self.side_angles.keys())
        for side in sides:
            self.polygons_gdf[f'len_{side}'] = 0.0
        for idx, row in self.polygons_gdf.iterrows():
            geometry = row['geometry']
            if isinstance(geometry, Polygon):
                polygons = [geometry]
            elif isinstance(geometry, MultiPolygon):
                polygons = [poly for poly in geometry.geoms]
            else:
                continue
            for poly in polygons:
                polygon_coords = np.array(poly.exterior.coords)
                for i in range(len(polygon_coords) - 1):
                    start_x, start_y = polygon_coords[i]
                    end_x, end_y = polygon_coords[i + 1]
                    segment_length, segment_orientation = self.calculate_segment_length_and_orientation(start_x, start_y, end_x, end_y)
                    orientation_class_name = self.classify_orientation(segment_orientation)
                    if orientation_class_name is not None:
                        self.polygons_gdf.at[idx, f'len_{orientation_class_name}'] += round(segment_length, 2)
        return self.polygons_gdf

    def classify_side(self, intersection):
        if intersection.geom_type == 'LineString':
            line = LineString(list(intersection.coords))
            if line.is_empty:
                return None
            x1, y1, x2, y2 = line.xy[0][0], line.xy[1][0], line.xy[0][-1], line.xy[1][-1]
            length, angle = self.calculate_segment_length_and_orientation(x1, y1, x2, y2)
            direction = self.classify_orientation(angle)
            return direction, length
        elif intersection.geom_type == 'MultiLineString':
            total_length = 0
            direction_lengths = {side: 0 for side in self.side_angles.keys()}
            for line in intersection.geoms:
                if line.is_empty:
                    continue
                x1, y1, x2, y2 = line.xy[0][0], line.xy[1][0], line.xy[0][-1], line.xy[1][-1]
                length, angle = self.calculate_segment_length_and_orientation(x1, y1, x2, y2)
                direction = self.classify_orientation(angle)
                if direction:
                    direction_lengths[direction] += length
                    total_length += length
            return direction_lengths, total_length
        else:
            return None

    def list_neighboring_polygons(self, unique_id_column='build_id', proximity_threshold=0.1):
        self.polygons_gdf['neighboring_polygons'] = None
        for idx, polygon in self.polygons_gdf.iterrows():
            neighbors = self.polygons_gdf[self.polygons_gdf.geometry.distance(polygon.geometry) <= proximity_threshold]
            neighbors = neighbors[neighbors.index != idx]
            neighbor_ids = neighbors.index.tolist()
            neighbor_ids = self.polygons_gdf.loc[neighbor_ids, unique_id_column].tolist()
            neighbor_ids_str = ",".join(map(str, neighbor_ids))
            self.polygons_gdf.at[idx, 'neighboring_polygons'] = neighbor_ids_str
        return self.polygons_gdf

    def length_of_neighbors_per_orientation(self, unique_id_column='build_id'):
        sides = ['N', 'NW', 'E', 'SW', 'S', 'SE', 'W', 'NE']
        result_df = pd.DataFrame(index=self.polygons_gdf.index, columns=[unique_id_column] + [f'len_{side}' for side in sides])
        result_df[unique_id_column] = self.polygons_gdf[unique_id_column]
        result_df = result_df.fillna(0.0).infer_objects()
        for idx, row in self.polygons_gdf.iterrows():
            neighboring_ids = row['neighboring_polygons']
            if neighboring_ids:
                neighboring_ids = neighboring_ids.split(',')
                for neighbor_id in neighboring_ids:
                    neighbor_polygon = self.polygons_gdf.loc[self.polygons_gdf[unique_id_column] == neighbor_id, 'geometry'].values[0]
                    intersection = row['geometry'].intersection(neighbor_polygon)
                    side_info = self.classify_side(intersection)
                    if side_info:
                        if isinstance(side_info, tuple):
                            side, common_side_length = side_info
                            if isinstance(side, dict):
                                for key, value in side.items():
                                    result_df.at[idx, f'len_{key}'] += round(value, 2)
                            else:
                                result_df.at[idx, f'len_{side}'] += round(common_side_length, 2)
        return result_df

    def subtract_facade_len_from_adjusted_sides(self, df1_facade_len, df2_adjusted_edges_len, key_column='build_id', columns_to_subtract=None):
        if columns_to_subtract is None:
            columns_to_subtract = ['len_N', 'len_NW', 'len_E', 'len_SW', 'len_S', 'len_SE', 'len_W', 'len_NE']
        merged_df = pd.merge(df1_facade_len, df2_adjusted_edges_len, on=key_column, suffixes=('_df1', '_df2'))
        result_df = df1_facade_len.copy()
        for col in columns_to_subtract:
            result_df[col] = merged_df[f'{col}_df1'] - merged_df[f'{col}_df2']
        return result_df

    def calculate_surface_area(self, row, heigth_column='h_mean', area_column='f_area', perimeter_column='f_perimeter'):
        perimeter = row[perimeter_column]
        height = row[heigth_column]
        area = row[area_column]
        return round(perimeter * height + 2 * area, 4)

    def calculate_volume(self, row, heigth_column='h_mean', area_column='f_area'):
        area = row[area_column]
        height = row[heigth_column]
        return round(area * height, 4)

    def calculate_s_v_ratio(self, row, surface_area_column='surface_area', volume_column='volume'):
        surface_area = row[surface_area_column]
        volume = row[volume_column]
        return round(surface_area / volume, 4)
#%%
# Example execution steps:
if __name__ == "__main__":
    analyser = FacadeAnalyser()
    analyser.load_polygons()
    # Merge height data if available
    #height_data_path = os.path.join(analyser.base_dir, 'height.geojson')
    height_data_path = os.path.join(os.getcwd(), "vector","buildings_footprint")
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
    adjusted_facades_len_df = analyser.list_neighboring_polygons()
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
        print(f"The length of facade less than 1m assigned to 0, for orientation {key} number of records is: {result_df[key].loc[result_df[key] < 1].count()}")
    print(f"Facade area per orientation calculated successfully.")

    # Save results if needed
    result_df.to_file("data/03_footprint_subtracted_facades_and_s_v_volume_area.geojson", driver="GeoJSON", index=False)
    result_df.drop(columns=['geometry']).to_csv("data/03_footprint_subtracted_facades_and_s_v_volume_area.csv", index=False)

    # Example of how to use the FacadeAnalyser class from another script

    # from 03_geopandas_facade_analyser import FacadeAnalyser

    # Initialize the analyser (optionally provide base_dir or file_path)
    analyser = FacadeAnalyser()

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

#%% Functions format and round the numerical columns to 2 decimal places and convert them to strings with formatting
# Round the numerical columns to 2 decimal places and convert them to strings with formatting
def format_cell(x):
    # Check if the value is None
    if x is None:
        return None
    # Convert the value to float, format it, and return as string
    return "{:.2f}".format(float(x))
def round_and_convert(x):
    if x is not None:
        return "{:.2f}".format(round(x, 2))
    else:
        return None
""" 
If needed, apply the formatting function to the specified columns, handling None values
# Apply the formatting function to the specified columns, handling None values
polygons_gdf[['perim_gpd', 'area_gpd', 'len_N', 'len_NE', 'len_E', 'len_SE', 'len_S', 'len_SW', 'len_W', 'len_NW']] = polygons_gdf[['perim_gpd', 'area_gpd', 'len_N', 'len_NE', 'len_E', 'len_SE', 'len_S', 'len_SW', 'len_W', 'len_NW']].applymap(format_cell)

# Add comma separators for thousands and two decimal places
polygons_gdf[['perim_gpd', 'area_gpd', 'len_N', 'len_NE', 'len_E', 'len_SE', 'len_S', 'len_SW', 'len_W', 'len_NW']] = polygons_gdf[['perim_gpd', 'area_gpd', 'len_N', 'len_NE', 'len_E', 'len_SE', 'len_S', 'len_SW', 'len_W', 'len_NW']].applymap(format_cell)
"""