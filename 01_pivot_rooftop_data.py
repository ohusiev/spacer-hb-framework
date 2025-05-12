# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 17:57:45 2024

@author: Oleksandr-MSI
"""
#%%
import pandas as pd
import geopandas as gpd
import os
import numpy as np
#%%
if __name__ == "__main__": # If the script is run from the main file
    print("Run from the main file")
    # Load the dataset
    file_dir = os.getcwd() #+'\\data\\'
    #All buildings 
    wb_rt_analysis_file_path = os.path.join(file_dir,'pyqgis_wb_automatiser\\bilbao_otxarkoaga_v2\\00_wb_rt_analysis_bilbao_otxarkoaga_v2_segments.shp') # Import of polygon data of whitebox rooftop analysis
    buildings_footprint_path = os.path.join(file_dir,'vector\\osm_buildings\\etrs_25830\\buildings_inspire_clip_oxarkoaga+census.shp') # Import of polygon data of segm_id footprint
    # Only buildings match from Reference Case
    #wb_rt_analysis_file_path = os.path.join(file_dir,'pyqgis_wb_automatiser\\\wb_otxarkoaga_pv\\\wb_otxarkoaga_pv.shp') # Import of polygon data of whitebox rooftop analysis
    #buildings_footprint_path = os.path.join(file_dir,'vector\\osm_buildings\\otxarkoaga_pv.shp') # Import of polygon data of segm_id footprint
    df_buildings_footprint = gpd.read_file(buildings_footprint_path)
    print(df_buildings_footprint.head())
        # Check if the file exists
    if not os.path.exists(wb_rt_analysis_file_path):
        print(f"File not found: {wb_rt_analysis_file_path}")
    else:
        gdf = gpd.read_file(wb_rt_analysis_file_path, col_index="FID") # Read the file
        df = gdf.drop(columns='geometry')
        print(gdf.head())  # Display the first few rows of the dataframe

azimuth_categories = {
    'roof_N': (337.5, 22.5),
    'roof_NE': (22.5, 60),
    'roof_E': (60, 111),
    'roof_SE': (111, 162),
    'roof_S': (162, 198),
    'roof_SW': (198, 249),
    'roof_W': (249, 300),
    'roof_NW': (300, 337.5),
    'flat_roof': (0, 0)  # Add flat_roof to the dictionary
}

#%%  
# Function to pivot the rooftop analysis data by slope and azimuth categories.
def pivot_whitebox_rooftop_analysis(file_dir=file_dir, df_segment=None, df_buildings_footprint=None, col_name="s_area", key_ids=["build_id", "census_id"]):
    if df_segment is None:
        df_segment = df
    if df_buildings_footprint is None:
        df_buildings_footprint = gpd.read_file(buildings_footprint_path)
    """
        df_segment: GeoDataFrame of the Whitebox Rooftop_analysis, default is `gdf`
    Parameters
    ----------
        file_dir: root where to save output pivot file, default is `file_dir`
        df: GeoDataFrame of the Whitebox Rooftop_analysis, default is `gdf`
        df_buildings_footprint: GeoDataFrame of the building footprint, default is `df_buildings_footprint`
        col_name: column name to pivot the data by (s_area, ppv, pv_gen) default is `s_area`
        key_ids: list of key columns to use for pivoting and merging, default is ["build_id", "census_id"]
    Returns
    -------   
    return: df, pivot_df, building_footprint 
    explanation: 
    df is the dataframe with the slope and azimuth categories    
    #pivot_df is the pivot table with sum of s_area, classified by slope and azimuth in *.csv format
    building_footprint is the merged data of pivot_df with the segm_id footprint in *.geojson format

    """
    if type(df_segment) != gpd.GeoDataFrame:
        print("The input data is not a GeoDataFrame")
    else:
        df_segment=df_segment.drop(columns='geometry')
    df_segment=df_segment.round(4)

    col_rename = {"AREA":"s_area", "SLOPE": 'slope',"FID": "segm_id", "BUILDING": "wb_build_id","MAX_ELEV": "wb_max_elev", "HILLSHADE": "hillshade","ASPECT": "aspect"}
    df_segment = df_segment.rename(columns=col_rename) # Rename the columns
    print(f"0. Renamed the columns: {col_rename}")   
    #if df_segment['build_id'] is has None value check
    if df_segment['build_id'].isnull().any():
        print("The 'build_id' column has None values")
        # Print the rows with missing values in the wb_build_id column
        mask = df_segment['build_id'].isnull() | df_segment['census_id'].isnull()
        print(df_segment.loc[mask,['build_id', 'census_id', 'wb_build_id']])
        print('Checking the missing values ... ')
        df_segment=check_nan_records(df_segment) # Check and fill the missing values in the rooftop analysis data
        #df_segment = fix_impurities(df_segment) # Check and fix the misidentification of whitebox
        #print(df_segment[mask])

    df_segment=df_segment[df_segment['s_area'] > 1] # Remove the rows with s_area less than 1
    print("1. Removed the rows with s_area less than 1 m2")

    # Applying the slope classification
    df_segment['slope_cat'] = df_segment['slope'].apply(classify_slope)
    df_segment[["slope", "aspect"]]=df_segment[["slope", "aspect"]].round(0).astype(int)
    # Applying the azimuth categorization based on the azimuth_categories dictionary
    df_segment["aspect_cat"] = df_segment["aspect"].apply(categorize_aspect) 
    print("3. Applied the slope classification and azimuth categorization")
    # Calculate the projected area of the roof segment
    df_segment['slope_rad'] = df_segment.slope/180*np.pi # Convert the slope to radians
    df_segment[f'{col_name}_proj'] = df_segment['s_area'] / np.cos(df_segment['slope_rad']) 
    # Pivot the table with sum of `col_name`, classified by azimuth and slope
    pivot_df_sloped= df_segment[df_segment['slope_cat']=="sloped_roof"].pivot_table(
        index=key_ids+["wb_build_id"], 
        columns=['aspect_cat'], 
        values= f'{col_name}_proj', # 's_area', 'ppv','pv_gen' 
        aggfunc='sum', 
        fill_value=0
    ).round(4)
    pivot_df_flat= df_segment[df_segment['slope_cat']=="flat_roof"].pivot_table(
        index=key_ids+["wb_build_id"], 
        columns=['slope_cat'], 
        values= f'{col_name}_proj', # 's_area', 'ppv','pv_gen' 
        aggfunc='sum', 
        fill_value=0
    ).round(4)

    pivot_df_flat["plain_roof"]=1
    
    # If the two DataFrames have different indexes, 
    # the resulting DataFrame will contain all unique indexes from both, 
    # with NaN values where there is no matching data for a specific index from either DataFrame.
    pivot_df = pd.concat([pivot_df_sloped, pivot_df_flat], axis=1)
    #print(pivot_df.columns)
    pivot_df["r_area"]=pivot_df[list(azimuth_categories.keys())].sum(axis=1)#+["flat_roof"]].sum(axis=1)

    print(f"4. Pivot the table with sum of '{col_name}' column, classified by azimuth and slope")

    df_segment['exception']=None
    reorder_cols = key_ids + ["wb_build_id","segm_id","exception", "wb_max_elev","hillshade",
            "slope","aspect","s_area","slope_cat","aspect_cat"]
    df_segment = df_segment[reorder_cols]
    pivot_df = pivot_df.reset_index()
    #print(pivot_df.columns)
    #print(df_buildings_footprint.head())
    #df_buildings_footprint= df_buildings_footprint.merge(pivot_df, on='build_id', how="inner") # Merge the pivot table with the segm_id footprint data
    df_buildings_footprint = df_buildings_footprint.merge(
    pivot_df.loc[:, ~pivot_df.columns.isin(df_buildings_footprint.columns) | (pivot_df.columns == 'build_id')], 
    on='build_id', 
    how="inner") # Merge the pivot table with the segm_id footprint data, ignoring columns already in the left dataframe
    df_buildings_footprint['exception']=None

    # Check if the "building" exist in the dataframe
    if "building" not in df_buildings_footprint.columns:
        df_buildings_footprint['building'] = None
    else:
        print("Residential buildings has to be filtered")
        #df_buildings_footprint=df_buildings_footprint[df_buildings_footprint['building']=="V"]
    #re-order columns
    # Ensure the 'build_id' column in df_segment has unique values
    df_segment_unique = df_segment.drop_duplicates(subset='build_id')
    #merge "wb_max_elev" to df_buildings_footprint by 'build_id'
    df_segment_unique = df_segment_unique[['build_id', 'wb_max_elev']]
    df_buildings_footprint = df_buildings_footprint.merge(df_segment_unique, on='build_id', how="inner")
    reorder_cols = ["build_id", "census_id", "wb_build_id"] + [v for v in list(df_buildings_footprint.columns) if v not in ["build_id", "census_id", "wb_build_id"] and v != "segm_id"]
    df_buildings_footprint = df_buildings_footprint[reorder_cols]
    #df_segment = df_segment.rename(columns={"CodEdifici":"build_id"})
    df_segment.to_excel(f"{file_dir}/data/01_segments_{col_name}_wb_rooftop_analysis.xlsx", sheet_name="Sheet1", index=False)
    df_buildings_footprint=df_buildings_footprint.rename(columns={"s_area":"r_area"})
    #print(df_buildings_footprint.columns)
    df_buildings_footprint.to_file(f"{file_dir}/data/01_footprint_{col_name}_wb_rooftop_analysis.geojson", driver='GeoJSON')
    df_buildings_footprint.to_excel(f"{file_dir}/data/01_footprint_{col_name}_wb_rooftop_analysis.xlsx",sheet_name="Sheet1", index=False)
    print(f"File(s) saved to: {file_dir}/data")
    return df_segment, df_buildings_footprint
#%%
# Re-defining the slope classification function
def classify_slope(slope, flat_threshold=10):
    if 0 < slope <= flat_threshold:
        return 'flat_roof'
    elif slope > flat_threshold:
        return 'sloped_roof'
    else:
        return None

def categorize_aspect(azimuth):
    for category, (start, end) in azimuth_categories.items():
        if start > end:
            if azimuth >= start or azimuth <= end:
                return category
        else:
            if start <= azimuth <= end:
                return category
    return None

#%% Function to check and fill the missing values in the rooftop analysis data
def check_nan_records(whitebox_rooftop_analysis, columns_to_check=['build_id', 'census_id']):
    rooftop_df_concat = pd.DataFrame() # Initialize an empty DataFrame to store the final result
    try:
        # Step 1: Identify rows in rooftop_df where any of the specified columns is null
        mask = whitebox_rooftop_analysis[columns_to_check].isnull().any(axis=1)
        empty_records = whitebox_rooftop_analysis[mask]

        # Step 2: Make a reference table of all unique 'segm_id' 
        wb_build_ids_unique = whitebox_rooftop_analysis.dropna(subset=columns_to_check)
        wb_build_ids_unique = wb_build_ids_unique.drop_duplicates(subset=['wb_build_id']).set_index('wb_build_id')

        # Step 3: For each column in rooftop_df you want to update, check if it's null and update based on the mapping
        for index, row in empty_records.iterrows():
            for col in columns_to_check:
                empty_records.at[index, col] = wb_build_ids_unique.at[row['wb_build_id'], col]

        # Step 4: Drop all the null values of original dataframe, and add corresponding filled data
        rooftop_df_clean = whitebox_rooftop_analysis.dropna(subset=columns_to_check)
        rooftop_df_concat = pd.concat([rooftop_df_clean, empty_records], axis=0)

        check = rooftop_df_concat[columns_to_check].isnull().any(axis=1)
        if check.any():
            rooftop_df_concat = pd.concat([rooftop_df_clean, empty_records], axis=0)
            print(f"NOT all {columns_to_check} NULL values mapped. \n Next records had to be dropped: \n {rooftop_df_concat[check]}")
        else:
            print(f"All {columns_to_check} NULL values have been mapped!")
    except KeyError as e:
        print(f"KeyError: {e}. Please check if the required columns exist in the dataframe.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return rooftop_df_concat
#%%
#Function that manually check misidentification of whitebox

def fix_impurities(df, building_col='wb_build_id', area_col='s_area', id_cols=['build_id', 'census_id']):
    # Group the GeoDataFrame by the specified building column
    grouped = df.groupby(building_col)
    
    # Iterate through each group
    for building, group in grouped:
        if group[id_cols].nunique().max() > 0:
            # Find the row with the largest area
            max_area_idx = group[area_col].idxmax()
            
            # Extract the values of the specified id columns from the row with the largest area
            max_area_row = group.loc[max_area_idx, id_cols]
            
            # Reassign the values of the specified id columns for all rows in this group
            df.loc[group.index, id_cols] = max_area_row.values
            
            # Print the corrections made
            print(f"Corrected {building_col} {building}: Set {id_cols} to {max_area_row.values}")
    
    return df

#%% 

"""
# Execution steps:
df_segments_wb_rooftop_analysis, building_footprint = pivot_whitebox_rooftop_analysis(file_dir=file_dir, df_segment=gdf,col_name="s_area")

"""

