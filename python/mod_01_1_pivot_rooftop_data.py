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
class PivotRooftopAnalysis:
    def __init__(self, file_dir, path_to_wb_rooftop_analysis='pyqgis_wb_automatiser\\bilbao_otxarkoaga_v2\\00_wb_rt_analysis_bilbao_otxarkoaga_v2_segments_xy_coord.geojson', path_to_buildings_footprint='vector\\buildings_footprint\\etrs_25830\\buildings_inspire_clip_oxarkoaga+census.shp'):
        self.file_dir = file_dir
        self.azimuth_categories = {
            'roof_N': (337.5, 22.5),
            'roof_NE': (22.5, 60),
            'roof_E': (60, 111),
            'roof_SE': (111, 162),
            'roof_S': (162, 198),
            'roof_SW': (198, 249),
            'roof_W': (249, 300),
            'roof_NW': (300, 337.5),
            'flat_roof': (0, 0)
        }
        self.path_to_wb_rooftop_analysis = path_to_wb_rooftop_analysis
        self.path_to_buildings_footprint = path_to_buildings_footprint

    def process_DataFrames(self):
        """
        Process rooftop analysis by reading and validating required files.

        Args:
            file_dir (str, optional): Directory path where the files are located. Defaults to current working directory.

        Returns:
            tuple: DataFrames for buildings footprint and rooftop analysis.
        """

        file_dir = self.file_dir or os.getcwd()
        
        # Define file paths
        wb_rt_analysis_file_path = os.path.join(
            file_dir, self.path_to_wb_rooftop_analysis
        )
        buildings_footprint_path = os.path.join(
            file_dir, self.path_to_buildings_footprint
        )
        
        # Read buildings footprint
        df_buildings_footprint = gpd.read_file(buildings_footprint_path)
        #print(df_buildings_footprint.head())
        
        # Validate and read rooftop analysis file
        if not os.path.exists(wb_rt_analysis_file_path):
            print(f"File not found: {wb_rt_analysis_file_path}")
        elif not os.path.isfile(wb_rt_analysis_file_path):
            print(f"File not found: {wb_rt_analysis_file_path}")
        else:
            gdf = gpd.read_file(wb_rt_analysis_file_path, col_index="FID")
            df = gdf.drop(columns='geometry')
            #print(gdf.head())
            return df, df_buildings_footprint

    def classify_slope(self, slope, flat_threshold=10):
        if 0 < slope <= flat_threshold:
            return 'flat_roof'
        elif slope > flat_threshold:
            return 'sloped_roof'
        else:
            return None

    def categorize_aspect(self, azimuth):
        for category, (start, end) in self.azimuth_categories.items():
            if start > end:
                if azimuth >= start or azimuth <= end:
                    return category
            else:
                if start <= azimuth <= end:
                    return category
        return None

    def check_nan_records(self, whitebox_rooftop_analysis, columns_to_check=['build_id', 'census_id']):
        rooftop_df_concat = pd.DataFrame()
        try:
            mask = whitebox_rooftop_analysis[columns_to_check].isnull().any(axis=1)
            empty_records = whitebox_rooftop_analysis[mask]
            wb_build_ids_unique = whitebox_rooftop_analysis.dropna(subset=columns_to_check)
            wb_build_ids_unique = wb_build_ids_unique.drop_duplicates(subset=['wb_build_id']).set_index('wb_build_id')

            for index, row in empty_records.iterrows():
                for col in columns_to_check:
                    empty_records.at[index, col] = wb_build_ids_unique.at[row['wb_build_id'], col]

            rooftop_df_clean = whitebox_rooftop_analysis.dropna(subset=columns_to_check)
            rooftop_df_concat = pd.concat([rooftop_df_clean, empty_records], axis=0)

            check = rooftop_df_concat[columns_to_check].isnull().any(axis=1)
            if check.any():
                print(f"NOT all {columns_to_check} NULL values mapped. \n Next records had to be dropped: \n {rooftop_df_concat[check]}")
            else:
                print(f"All {columns_to_check} NULL values have been mapped!")
        except KeyError as e:
            print(f"KeyError: {e}. Please check if the required columns exist in the dataframe.")
        except Exception as e:
            print(f"An error occurred: {e}")
        return rooftop_df_concat

    def fix_impurities(self, df, building_col='wb_build_id', area_col='s_area', id_cols=['build_id', 'census_id']):
        grouped = df.groupby(building_col)
        for building, group in grouped:
            if group[id_cols].nunique().max() > 0:
                max_area_idx = group[area_col].idxmax()
                max_area_row = group.loc[max_area_idx, id_cols]
                df.loc[group.index, id_cols] = max_area_row.values
                print(f"Corrected {building_col} {building}: Set {id_cols} to {max_area_row.values}")
        return df

    def pivot_whitebox_rooftop_analysis(self, df_segment=None, df_buildings_footprint=None, col_name="s_area", key_ids=["build_id", "census_id"]):
        if df_segment is None:
            raise ValueError("df_segment cannot be None")
        if df_buildings_footprint is None:
            raise ValueError("df_buildings_footprint cannot be None")

        df_segment = df_segment.drop(columns='geometry', errors='ignore').round(4)
        col_rename = {"AREA": "s_area", "SLOPE": 'slope', "FID": "segm_id", "BUILDING": "wb_build_id",
                      "MAX_ELEV": "wb_max_elev", "HILLSHADE": "hillshade", "ASPECT": "aspect"}
        df_segment = df_segment.rename(columns=col_rename)

        if df_segment['build_id'].isnull().any():
            df_segment = self.check_nan_records(df_segment)

        df_segment = df_segment[df_segment['s_area'] > 1]
        df_segment['slope_cat'] = df_segment['slope'].apply(self.classify_slope)
        df_segment[["slope", "aspect"]] = df_segment[["slope", "aspect"]].round(0).astype(int)
        df_segment["aspect_cat"] = df_segment["aspect"].apply(self.categorize_aspect)
        df_segment['slope_rad'] = df_segment.slope / 180 * np.pi
        df_segment[f'{col_name}_proj'] = df_segment['s_area'] / np.cos(df_segment['slope_rad'])

        pivot_df_sloped = df_segment[df_segment['slope_cat'] == "sloped_roof"].pivot_table(
            index=key_ids + ["wb_build_id"],
            columns=['aspect_cat'],
            values=f'{col_name}_proj',
            aggfunc='sum',
            fill_value=0
        ).round(4)

        pivot_df_flat = df_segment[df_segment['slope_cat'] == "flat_roof"].pivot_table(
            index=key_ids + ["wb_build_id"],
            columns=['slope_cat'],
            values=f'{col_name}_proj',
            aggfunc='sum',
            fill_value=0
        ).round(4)

        pivot_df_flat["plain_roof"] = 1
        pivot_df = pd.concat([pivot_df_sloped, pivot_df_flat], axis=1)
        pivot_df["r_area"] = pivot_df[list(self.azimuth_categories.keys())].sum(axis=1)

        df_segment['exception'] = None
        reorder_cols = key_ids + ["wb_build_id", "segm_id", "exception", "wb_max_elev", "hillshade",
                                  "slope", "aspect", "s_area", "slope_cat", "aspect_cat"]
        df_segment = df_segment[reorder_cols]
        pivot_df = pivot_df.reset_index()

        df_buildings_footprint = df_buildings_footprint.merge(
            pivot_df.loc[:, ~pivot_df.columns.isin(df_buildings_footprint.columns) | (pivot_df.columns == 'build_id')],
            on='build_id',
            how="inner"
        )
        df_buildings_footprint['exception'] = None

        if "building" not in df_buildings_footprint.columns:
            df_buildings_footprint['building'] = None

        df_segment_unique = df_segment.drop_duplicates(subset='build_id')
        df_segment_unique = df_segment_unique[['build_id', 'wb_max_elev']]
        df_buildings_footprint = df_buildings_footprint.merge(df_segment_unique, on='build_id', how="inner")

        reorder_cols = ["build_id", "census_id", "wb_build_id"] + \
                       [v for v in list(df_buildings_footprint.columns) if
                        v not in ["build_id", "census_id", "wb_build_id"] and v != "segm_id"]
        df_buildings_footprint = df_buildings_footprint[reorder_cols]

        df_segment.to_excel(f"{self.file_dir}/data/01_segments_{col_name}_wb_rooftop_analysis.xlsx", sheet_name="Sheet1", index=False)
        df_buildings_footprint = df_buildings_footprint.rename(columns={"s_area": "r_area"})
        df_buildings_footprint.to_file(f"{self.file_dir}/data/01_footprint_{col_name}_wb_rooftop_analysis.geojson", driver='GeoJSON')
        df_buildings_footprint.to_excel(f"{self.file_dir}/data/01_footprint_{col_name}_wb_rooftop_analysis.xlsx", sheet_name="Sheet1", index=False)
        print(f"File(s) saved to: {self.file_dir}/data")
        return df_segment, df_buildings_footprint

#%% 

"""
# Execution steps:
df_segments_wb_rooftop_analysis, building_footprint = pivot_whitebox_rooftop_analysis(file_dir=file_dir, df_segment=gdf,col_name="s_area")

"""

