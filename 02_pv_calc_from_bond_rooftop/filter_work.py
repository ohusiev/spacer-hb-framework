#%%
import pandas as pd
import numpy as np
import os
root = os.getcwd()
path = os.path.join(root, "data\Otxarkoaga\\filter_10kWp_limit\\01_segments_s_area_wb_rooftop_analysis.xlsx")
data = pd.read_excel(path)

# %%
panel_pars = {'pv': {}, 'heat': {}} # panel parameters
panel_pars['pv']['s_panel'] = 0.99 * 1.95 # 1.93 m2
nominal_power = 0.3 # kWp
# limit of the kWp of panels on the roof considering regulatory disadvantages
panel_pars['pv']['kWp_limit'] = 10

# calculate limit number of panels on the roof considering regulatory disadvantages
panel_pars['pv']['n_panel_limit'] = np.floor(panel_pars['pv']['kWp_limit'] / nominal_power)
# calculate limit area for panels on the roof considering regulatory disadvantages
panel_pars['pv']['s_panel_limit'] = panel_pars['pv']['n_panel_limit'] * panel_pars['pv']['s_panel']
filter = Filter(data)
data_filtered = filter.calculate_filtered_area(data, panel_pars, nominal_power=0.3)
# %%
if not os.path.exists(os.path.join(root, f"data\Otxarkoaga\\filter_{panel_pars['pv']['kWp_limit']}kWp_limit")):
    os.makedirs(os.path.join(root, f"data\Otxarkoaga\\filter_{panel_pars['pv']['kWp_limit']}kWp_limit"))
data_filtered.to_excel(os.path.join(root, f"data\Otxarkoaga\\filter_{panel_pars['pv']['kWp_limit']}kWp_limit\\02_segments_s_area_wb_rooftop_analysis_filtered.xlsx"), index=False)
# %%
# WORKING FILTERING FUNCTION BUT FOR ALL THE SEGMENTS OF THE BUILDING (without initial filtering by aspect)
def calculate_filtered_area(self, segment, panel_pars, nominal_power=0.3, k_panel_density=None):
    """
    Filters and calculates the maximum available area for PV installation
    per building based on panel parameters and installation limits.

    Args:
        segment (pd.DataFrame): DataFrame containing building and segment information.
        panel_pars (dict): Dictionary with panel parameters and installation limits.
        nominal_power (float, optional): Nominal power output of each panel in kWp. Default is 0.3 kWp.
        k_panel_density (pd.Series, optional): Panel density factors for plain and slopy areas. Default is None.

    Returns:
        pd.DataFrame: DataFrame with updated segment areas and scale factors for each building.
    """
    if k_panel_density is None:
        k_panel_density = pd.Series({'plain': 0.75, 'slopy': 0.65})
    # Filter segments with slope > 0 and calculate area
    # Calculate area limits for PV installations
    s_panel = panel_pars['pv']['s_panel']
    kWp_limit = panel_pars['pv']['kWp_limit']
    n_panel_limit = np.floor(kWp_limit / nominal_power)
    s_panel_limit_plain = n_panel_limit * s_panel #* 1/k_panel_density['plain']

    s_panel_limit_slopy = n_panel_limit * s_panel #* 1/k_panel_density['slopy']
    segment["scale_factor"]=0
    segment['s_area_rad'] = segment['s_area'] / np.cos(segment['slope']/180*np.pi) # м²

    # Adjust segment area based on building limits
    for build_id, group in segment.groupby('build_id'):
        total_area_plain = group[group['slope'] == 0]['s_area'].sum()
        total_area_slopy = group[group['slope'] > 0]['s_area_rad'].sum()

        scale_factor_plain = min(1, s_panel_limit_plain / (total_area_plain* k_panel_density['plain'])) if total_area_plain > 0 else 0

        scale_factor_slopy = min(1, s_panel_limit_slopy / (total_area_slopy* k_panel_density['slopy'])) if total_area_slopy > 0 else 0
        #group['scale_factor'] = scale_factor_slopy

        #group.loc[group['slope'] == 0, 's_area'] = group.loc[group['slope'] == 0, 's_area'] * scale_factor_plain
        #group.loc[group['slope'] > 0, 's_area'] = group.loc[group['slope'] > 0, 's_area'] * scale_factor_slopy

        # Apply scale factors to the group
        segment.loc[group.index, 'scale_factor'] = (
            scale_factor_plain if group['slope'].iloc[0] == 0 else scale_factor_slopy
        )
        segment.loc[group.index.intersection(segment[segment['slope'] == 0].index), 's_area'] *= scale_factor_plain
        segment.loc[group.index.intersection(segment[segment['slope'] > 0].index), 's_area'] *= scale_factor_slopy

    return segment