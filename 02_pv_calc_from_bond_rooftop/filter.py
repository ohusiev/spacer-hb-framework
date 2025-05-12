import numpy as np
import pandas as pd
import os
#%%
class Filter:
    def __init__(self, segment=None):
        """
        Initialize the Filter class with optional input dataframe of segment.

        Args:
            segment: Optional input data. Default is None.
        """
        self.segment = segment

    def calculate_filtered_area(self, segment, panel_pars, nominal_power=0.3, k_panel_density=None):
        """
        Filters and calculates the maximum available area for PV installation
        per building based on panel parameters and installation limits.

        Args:
            segment (pd.DataFrame): DataFrame with building and segment information.
            panel_pars (dict): Dictionary containing panel parameters and limits.
            nominal_power (float): Nominal power output of each panel in kWp.

        Returns:
            pd.DataFrame: DataFrame with maximum PV area and corresponding data for each building.
        """
        if k_panel_density is None:
            k_panel_density = pd.Series({'plain': 0.75, 'slopy': 0.65})
        # Filter segments with slope > 0 and calculate area
        # Calculate area limits for PV installations
        s_panel = panel_pars['pv']['s_panel']
        kWp_limit = panel_pars['pv']['kWp_limit']
        n_panel_limit = kWp_limit / nominal_power#np.floor(kWp_limit / nominal_power)
        s_panel_limit_plain = n_panel_limit * s_panel #* 1/k_panel_density['plain']

        s_panel_limit_slopy = n_panel_limit * s_panel #* 1/k_panel_density['slopy']
        segment["scale_factor"]=0
        segment['s_area_rad'] = segment['s_area'] / np.cos(segment['slope']/180*np.pi) # м²

        # Adjust segment area based on building limits
        segment_idx = ((segment.aspect > 85) & (segment.aspect < 275) & (segment.slope > 0)) | (segment.slope == 0)
        for build_id, group in segment[segment_idx].groupby('build_id'):
            total_area_plain = group[group['slope'] == 0]['s_area'].sum()
            total_area_slopy = group[group['slope'] > 0]['s_area_rad'].sum()
            #if build_id == 201303200512:
            #    print("slopy tot area: ", total_area_slopy)

            scale_factor_plain = min(1, s_panel_limit_plain / (total_area_plain * k_panel_density['plain'])) if total_area_plain > 0 else 0
            scale_factor_slopy = min(1, s_panel_limit_slopy / (total_area_slopy * k_panel_density['slopy'])) if total_area_slopy > 0 else 0
            #if build_id == 201303200512:
            #    print("slopy tot factor: ", scale_factor_slopy)

            # Apply scale factors to the group
            segment.loc[group.index, 'scale_factor'] = (
                scale_factor_plain if group['slope'].iloc[0] == 0 else scale_factor_slopy
            )
            segment.loc[group.index.intersection(segment[segment['slope'] == 0].index), 's_area'] *= scale_factor_plain
            segment.loc[group.index.intersection(segment[segment['slope'] > 0].index), 's_area'] *= scale_factor_slopy
    
        return segment
    
    def save_filtered_area(self, path, data):
        """
        Save the filtered area data to an Excel file.

        Args:
            path (str): Path to the Excel file.
            data (pd.DataFrame): DataFrame containing the filtered area data.
        """
        if not os.path.exists(os.path.join(root, f"data\Otxarkoaga\\filter_{panel_pars['pv']['kWp_limit']}kWp_limit")):
            os.makedirs(os.path.join(root, f"data\Otxarkoaga\\filter_{panel_pars['pv']['kWp_limit']}kWp_limit"))
        data_filtered.to_excel(os.path.join(root, f"data\Otxarkoaga\\filter_{panel_pars['pv']['kWp_limit']}kWp_limit\\02_segments_s_area_wb_rooftop_analysis_filtered.xlsx"), index=False)
        print(f"Filtered area data saved to {path}")

