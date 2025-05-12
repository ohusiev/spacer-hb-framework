#%%
import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import MultiPolygon, Polygon
import numpy as np
from shapely.geometry import LineString

# Define the base directory for input files
base_dir = os.path.join(os.getcwd(), "vector", "osm_buildings")

# Define the file paths
file_path = os.path.join(base_dir, "etrs_25830/buildings_inspire_clip_oxarkoaga+census.shp")

# Load your polygons into a GeoDataFrame

# Load your polygons into GeoDataFrames
polygons_gdf = gpd.read_file(file_path,index_col="ID")
# select only the columns of interest
build_id = 'build_id'# ['build_id', 'osm_id', 'CodEdifici']
col_of_interest = [build_id,'census_id','Codigo_Pol','Codigo_Par', 'building',  'Numero_Alt',
       'Ano_Constr', 'Ano_Rehabi',  'geometry'] #'Codigo_Uso',
polygons_gdf = polygons_gdf[col_of_interest]
#%%
"""CALCULATE LENGTH OF POLYGON SIDE PER ORIENTATION"""
# Define a function to calculate the length and orientation of a line segment
def calculate_segment_length_and_orientation(start_x, start_y, end_x, end_y):
    dx = end_x - start_x
    dy = end_y - start_y
    length = np.round(np.sqrt(dx**2 + dy**2), decimals=2)
    angle = (180 / np.pi) * np.arctan2(dy, dx)
    if angle < 0:
        angle += 360  # Convert angle to range [0, 360)
    angle = (360 - angle) % 360  # Convert to clockwise angle
    return length, angle

side_angles = {
    'N': (337.5, 22.5),
    'NW': (300, 337.5), 
    'E': (60, 111), 
    'SW': (198, 249), 
    'S': (162, 198), 
    'SE': (111, 162), 
    'W': (249, 300), 
    'NE': (22.5, 60)}

# Function to classify orientation based on angle
def classify_orientation(angle, side_angles):
    for side, (start, end) in side_angles.items():
        # Handle the case where the range wraps around (e.g., N: 337.5 to 22.5)
        if start > end:
            if angle >= start or angle < end:
                return side
        else:
            if start <= angle < end:
                return side
    return None  # Return None if no match is found (shouldn't happen with valid input)
# Function to calculate the length of each side of the polygon per orientations (N, NE, E, SE, S, SW, W, NW)
def length_per_orientation(polygons_gdf):
    sides = list(side_angles.keys())
    
    # Initialize columns for each side orientation
    for side in sides:
        polygons_gdf[f'len_{side}'] = 0.0  # Initialize with 0.0 for ease of addition
    
    
    # Iterate over each polygon
    for idx, row in polygons_gdf.iterrows():
        geometry = row['geometry']
        
        # Ensure geometry is either Polygon or MultiPolygon
        if isinstance(geometry, Polygon):
            polygons = [geometry]  # Treat as a single polygon list
        elif isinstance(geometry, MultiPolygon):
            polygons = [poly for poly in geometry.geoms]  # Access individual polygons within the MultiPolygon
        else:
            continue  # Skip if not a Polygon or MultiPolygon
        
        # Process each polygon in the (Multi)Polygon
        for poly in polygons:
            # Extract polygon coordinates
            polygon_coords = np.array(poly.exterior.coords)
            
            # Iterate over each vertex of the polygon
            for i in range(len(polygon_coords) - 1):
                start_x, start_y = polygon_coords[i]
                end_x, end_y = polygon_coords[i + 1]
                
                # Calculate the length and orientation of the line segment
                segment_length, segment_orientation = calculate_segment_length_and_orientation(start_x, start_y, end_x, end_y)
                orientation_class_name= classify_orientation(segment_orientation, side_angles)
                
                # Classify the segment based on orientation
                #segment_orientation_class = round(segment_orientation / 45) % 8  # Divide by 45 degrees and modulo 8 to get a value between 0 and 7
                #orientation_class_name = sides[segment_orientation_class]
                
                # Add the length of the segment to the corresponding column
                if orientation_class_name is not None:
                    polygons_gdf.at[idx, f'len_{orientation_class_name}'] += round(segment_length, 2)  # Accumulate length

    return polygons_gdf
#%% # Define the functions to calculate the length of neighboring polygons per orientation
def classify_side(intersection):
    if intersection.geom_type == 'LineString':
        line = LineString(list(intersection.coords))
        if line.is_empty:
            return None
        x1, y1, x2, y2 = line.xy[0][0], line.xy[1][0], line.xy[0][-1], line.xy[1][-1]
        length, angle = calculate_segment_length_and_orientation(x1, y1, x2, y2)
        direction = classify_orientation(angle, side_angles)
        return direction, length
    elif intersection.geom_type == 'MultiLineString':
        total_length = 0
        direction_lengths = {side: 0 for side in side_angles.keys()}
        for line in intersection.geoms:
            if line.is_empty:
                continue
            x1, y1, x2, y2 = line.xy[0][0], line.xy[1][0], line.xy[0][-1], line.xy[1][-1]
            length, angle = calculate_segment_length_and_orientation(x1, y1, x2, y2)
            direction = classify_orientation(angle, side_angles)
            if direction:
                direction_lengths[direction] += length
                total_length += length
        return direction_lengths, total_length
    else:
        return None

def list_neighboring_polygons(polygons_gdf, unique_id_column='build_id', proximity_threshold=0.1):
    # Create a new column to store the neighboring polygons
    polygons_gdf['neighboring_polygons'] = None
    
    # Iterate over each polygon
    for idx, polygon in polygons_gdf.iterrows():
        # Find polygons within the proximity threshold
        neighbors = polygons_gdf[polygons_gdf.geometry.distance(polygon.geometry) <= proximity_threshold]
        
        # Remove the polygon itself from the list of neighbors
        neighbors = neighbors[neighbors.index != idx]
        
        # Get the IDs of neighboring polygons
        neighbor_ids =  neighbors.index.tolist()
        neighbor_ids = polygons_gdf.loc[neighbor_ids,unique_id_column].tolist()
        
        # Convert the list of IDs to a string representation
        neighbor_ids_str = ",".join(map(str, neighbor_ids))
        
        # Update the 'neighboring_polygons' column with the string representation of neighbor IDs
        polygons_gdf.at[idx, 'neighboring_polygons'] = neighbor_ids_str
    
    return polygons_gdf
    
def length_of_neighbors_per_orientation(polygons_gdf, unique_id_column='build_id'):
    sides = ['N', 'NW', 'E', 'SW', 'S', 'SE', 'W', 'NE']
    unique_id_column = unique_id_column
    result_df = pd.DataFrame(index=polygons_gdf.index, columns=[unique_id_column] +[f'len_{side}' for side in sides])
    result_df[unique_id_column] = polygons_gdf[unique_id_column]
    result_df = result_df.fillna(0.0).infer_objects()  # Initialize with 0.0 for ease of addition and infer correct data types

    for idx, row in polygons_gdf.iterrows():
        neighboring_ids = row['neighboring_polygons']
        if neighboring_ids:
            neighboring_ids = neighboring_ids.split(',')  # Convert string of IDs to list
            for neighbor_id in neighboring_ids:
                neighbor_polygon = polygons_gdf.loc[polygons_gdf[unique_id_column] == neighbor_id, 'geometry'].values[0]
                #neighbor_polygon = polygons_gdf.loc[int(neighbor_id), 'geometry']
                intersection = row['geometry'].intersection(neighbor_polygon)
                side_info = classify_side(intersection)
                if side_info:
                    if isinstance(side_info, tuple):
                        side, common_side_length = side_info
                        if isinstance(side, dict):
                            for key, value in side.items():
                                result_df.at[idx, f'len_{key}'] += round(value, 2)
                        else:
                            result_df.at[idx, f'len_{side}'] += round(common_side_length, 2)
                    #elif isinstance(side_info, dict):
                        #for side, length in side_info.items():
                            #result_df.at[idx, f'len_{side}'] += round(length, 2)

                #ATTEMPT TO CALCULATE THE LENGTH OF CLOSELY LOCATED POLYGONS
                """ 
                buffered_polygon = row['geometry'].buffer(proximity_threshold)
                #intersection = buffered_polygon.intersection(neighbor_polygon)
                # Check if the intersection is significant
                
                if not intersection.is_empty:
                    side = classify_side(intersection)
                    if side:
                        common_side_length = round(intersection.length, 2)
                        result_df.at[idx, f'len_{side}'] += common_side_length
                """
                
    return result_df
#%% # Define the functions and corresponding columns to calculate the surface area, volume, and surface-to-volume ratio of each building
def calculate_surface_area(row, heigth_column='h_mean', area_column='f_area', perimeter_column='f_perimeter'): 
    perimeter = row[perimeter_column]
    height = row[heigth_column]
    area = row[area_column]
    return round(perimeter*height + 2* area, 4) # Simplification: assuming the building is a right rectangular prism
def calculate_volume(row,heigth_column='h_mean', area_column='f_area'):
    area = row[area_column]
    height = row[heigth_column]
    return round(area * height, 4) # Simplification: assuming the building is a right rectangular prism
def calculate_s_v_ratio(row, surface_area_column='surface_area', volume_column='volume'):
    surface_area = row[surface_area_column]
    volume = row[volume_column]
    return round(surface_area/volume, 4)
#%% 
#JOIN FIELDS 'Codigo_Mun', 'Codigo_Pol', 'Codigo_Par', 'Codigo_Sub', 'Codigo_Edi' TO CREATE A id OF BUILDING
#polygons_gdf['build_id']= polygons_gdf['Codigo_Mun'].astype(str) + polygons_gdf['Codigo_Pol'].astype(str) + polygons_gdf['Codigo_Par'].astype(str) + polygons_gdf['Codigo_Sub'].astype(str) + polygons_gdf['Codigo_Edi'].astype(str)
""" 
# Execution steps:
#1. Run the file to initialize the functions
#2. Load the GeoDataFrame with building footprint polygons 
#polygons_gdf = gpd.read_file(file_path)# index_col="ID")

#3. Merge the height data with the building footprint data
height_data_path = os.path.join(base_dir, 'height.geojson')
gdf_height = gpd.read_file(height_data_path).round(4)
polygons_gdf = polygons_gdf.merge(gdf_height[['build_id', 'h_mean', 'h_stdev', 'h_min', 'h_max']], on='build_id', how='left')

#4. Calculate area and perimeter of buildings footprint polygons
polygons_gdf['f_area'] = round(polygons_gdf['geometry'].area,4)#.round(2)
polygons_gdf['f_perimeter'] = round(polygons_gdf['geometry'].length,4)#.round(2)
polygons_gdf['n_floorsEstim'] = (polygons_gdf['h_mean'] / 3.0).round(0) # Simplification: assuming the building is a right rectangular prism
polygons_gdf['h_estim'] = polygons_gdf['n_floorsEstim'] * 3.0 + 1 # Simplification: assuming the building is a right rectangular prism
#5. Calculate the length of each side of the polygon per orientations and save to a GeoJSON file
facades_per_orientation_len_df =  length_per_orientation(polygons_gdf)
#facades_per_orientation_len_df.to_file("data/03_geopandas_facade_analyser/03_length_facades_per_orientation.geojson", driver='GeoJSON', index = False)

#6. Calculate the length of neighboring polygons per orientations and save to a CSV file
adjusted_facades_len_df = list_neighboring_polygons(polygons_gdf)
adjusted_facades_len_df = length_of_neighbors_per_orientation(adjusted_facades_len_df)
#adjusted_facades_len_df.to_csv("data/03_geopandas_facade_analyser/03_length_of_adjusted_facades.csv", index = False)

"""
""" 
# Suntetically generate data of year of construction for testing
#polygons_gdf['year_constr'] = np.random.randint(1900, 2010, polygons_gdf.shape[0])

# Calculate the surface area of each building
polygons_gdf['surface_area'] = polygons_gdf.apply(calculate_surface_area, axis=1) # Simplification: assuming the building is a right rectangulasr prism

# Calculate the volume of each building
polygons_gdf['volume'] = polygons_gdf.apply(calculate_volume, axis=1) # Simplification: assuming the building is a right rectangular prism

# Calculate the surface-to-volume ratio of each building
polygons_gdf['s_v_ratio'] = polygons_gdf.apply(calculate_s_v_ratio, axis=1).round(4)
#polygons_gdf.to_file("data/03_geopandas_facade_analyser/03_merged_buildings_with_s_v_volume_area.geojson", driver='GeoJSON', index = False)

# Calculate subtracted length of facades
result_df = subtract_facade_len_from_adjusted_sides(facades_per_orientation_len_df, adjusted_facades_len_df)
# result_df.to_file("data/03_subtracted_length_of_facades.geojson", driver='GeoJSON', index = False)
#Calculate facade area per orientation
fadace_length_cols = {'len_N':'N', 'len_NE':'NE', 'len_E':'E', 
            'len_SE':'SE', 'len_S':'S', 'len_SW':'SW', 'len_W':'W', 'len_NW':'NW'}
for key, value in fadace_length_cols.items():
    #Consider all values
    #result_df[f"fa_area_{value}"] = result_df[key] * result_df['h_mean']
    #Don't consider values less then 1m
    result_df[f"fa_area_{value}"] = [0 if x < 0.1 else x for x in result_df[key]] * result_df['h_mean'] 
    print(f"The lenght of facede less then 1m assigned to 0, for orientation {key} number of records is: {result_df[key].loc[result_df[key] < 1].count()}")
    #result_df[f"fa_area_{key}"] = result_df[key] * result_df['h_mean']
result_df.to_file("data/03_geopandas_facade_analyser/03_footprint_subtracted_facades_and_s_v_volume_area.geojson", driver = "GeoJSON", index=False)
result_df = result_df.drop(columns=['geometry'])
result_df.to_csv("data/03_footprint_subtracted_facades_and_s_v_volume_area.csv", index=False)
"""

#%%
def subtract_facade_len_from_adjusted_sides(df1_facade_len, df2_adjusted_edges_len, key_column='build_id', columns_to_subtract=None):
    # Default columns to subtract if not provided
    if columns_to_subtract is None:
        columns_to_subtract = ['len_N', 'len_NW', 'len_E', 'len_SW', 'len_S', 'len_SE', 'len_W', 'len_NE']
    
    # Merge the dataframes on the key_column
    merged_df = pd.merge(df1_facade_len, df2_adjusted_edges_len, on=key_column, suffixes=('_df1', '_df2'))
    
    # Create a new dataframe to store the results of the subtractions
    result_df = df1_facade_len.copy()
    
    # Subtract the corresponding columns
    for col in columns_to_subtract:
        result_df[col] = merged_df[f'{col}_df1'] - merged_df[f'{col}_df2']
    
    # Return the result dataframe
    return result_df

# Example usage
#result_df = subtract_facade_len_from_adjusted_sides(len_per_orientation_df, neighbor_lengths_df)
#result_df.to_excel("data/03_subtracted_length_of_facades.xlsx", index=False)

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

# %%
##FOR TEST
""" 
polygons_gdf = adjusted_facades_len_df.loc[adjusted_facades_len_df['CodEdifici'].isin(["201309500114",'201309500115', '201309500116','201309500117'])]
sides = ['N', 'NW', 'E', 'SW', 'S', 'SE', 'W', 'NE']
unique_id_column = ["CodEdifici"]
result_df = pd.DataFrame(index=polygons_gdf.index, columns=unique_id_column +[f'len_{side}' for side in sides])
result_df[unique_id_column] = polygons_gdf[unique_id_column]
result_df = result_df.fillna(0.0).infer_objects()  # Initialize with 0.0 for ease of addition and infer correct data types
for idx, row in polygons_gdf.iterrows():
    neighboring_ids = row['neighboring_polygons']
    if neighboring_ids:
        neighboring_ids = neighboring_ids.split(',') 
        neighbor_polygon = polygons_gdf.loc[polygons_gdf['CodEdifici'] == neighboring_ids, 'geometry'].values[0]
        #neighbor_polygon = polygons_gdf.loc[int(neighbor_id), 'geometry']
        intersection = row['geometry'].intersection(neighbor_polygon)
        print(intersection)
        side = classify_side(intersection)
        print(side)
        if side:
            common_side_length = round(intersection.length, 2)
            result_df.at[idx, f'len_{side}'] += common_side_length
"""